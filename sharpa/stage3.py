"""Stage 3 networks: a bounded actor and twin critics around the frozen policy.

The observation may start with the frozen token (the pooled decision slots of the four-expert policy, `token` = 256, or 0
when it is left out); the rest are the frozen policy's read-outs, proprioception, fingertip force, the reference motion
and the last correction (sim/env.py). Fixed normalisation of the non-token part, as in Facet-0's online adaptation. The
actor's last layer starts at zero: before any update the correction is zero and the system is the frozen policy.
"""
from __future__ import annotations

import torch
import torch.nn as nn

TOKEN = 256


class _Features(nn.Module):
    def __init__(self, obs_dim: int, proj: int, token: int = TOKEN):
        super().__init__()
        self.token = token
        self.register_buffer("mean", torch.zeros(obs_dim - token))
        self.register_buffer("std", torch.ones(obs_dim - token))
        self.tok = nn.Linear(token, proj) if token else None
        self.out_dim = (proj if token else 0) + obs_dim - token

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean.copy_(mean)
        self.std.copy_(std.clamp(min=1e-3))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        rest = ((obs[:, self.token:] - self.mean) / self.std).clamp(-8, 8)
        return torch.cat([torch.relu(self.tok(obs[:, :self.token])), rest], 1) if self.token else rest


def _mlp(i: int, h: int, o: int, norm: bool = False) -> nn.Sequential:
    n = (lambda: nn.LayerNorm(h)) if norm else nn.Identity
    return nn.Sequential(nn.Linear(i, h), n(), nn.ReLU(), nn.Linear(h, h), n(), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    """obs [B, obs_dim] -> correction in [-1, 1]^act_dim (before the environment's rate limit)."""

    def __init__(self, obs_dim: int, act_dim: int, proj: int = 64, hidden: int = 256, token: int = TOKEN):
        super().__init__()
        self.f = _Features(obs_dim, proj, token)
        self.net = _mlp(self.f.out_dim, hidden, act_dim)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(self.f(obs)))


class Critic(nn.Module):
    """Twin Q(s, a). obs is the critic observation: the actor's observation followed by the simulator's object state."""

    def __init__(self, obs_dim: int, act_dim: int, proj: int = 64, hidden: int = 256, token: int = TOKEN):
        super().__init__()
        self.f1, self.f2 = _Features(obs_dim, proj, token), _Features(obs_dim, proj, token)
        # layer norm keeps Q from running away for corrections that were never tried
        self.q1, self.q2 = _mlp(self.f1.out_dim + act_dim, hidden, 1, True), _mlp(self.f2.out_dim + act_dim, hidden, 1, True)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        return (self.q1(torch.cat([self.f1(obs), act], 1)).squeeze(1), self.q2(torch.cat([self.f2(obs), act], 1)).squeeze(1))


def limit(action: torch.Tensor, prev: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """The environment's rate limit: the executed correction stays within `rate` of the previous one."""
    return prev + torch.maximum(torch.minimum(action - prev, rate), -rate)
