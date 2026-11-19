"""Force-aware chunked policy with decision slots and a mixture-of-experts action head.

    tokens:  vision (cams x 17)  +  state history (K)  +  tactile history (K)  +  torque history (K)  +  task (1)
    encoder: bidirectional transformer  ->  "state cache" (encoded once per step)
    readout: learned query slots cross-attend into the cache
               H chunk slots   -> action chunk (joint-target deltas) through K gated experts
                               -> expected fingertip force for the same steps
               3 decision slots-> contact-soon (yes/no), phase (choice), force level (ordinal)
             the decision slots also produce the expert gate.

Force enters in four switchable ways (config `force.*`): tactile input, torque input, force
output, and force-derived supervision of the decision slots. With an input switched off its
tokens are replaced by one learned "not available" token, so the network and its size are
identical with and without force. This is different from feeding a measured zero
(`eval.zero_force`), which goes through the normal input path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sphere(z: torch.Tensor) -> torch.Tensor:
    """Every token scaled to root-mean-square 1 (EBR keeps the latent on this sphere)."""
    return z * torch.rsqrt(z.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(z.dtype)


class _Attention(nn.Module):
    """Plain attention written out, so that it can be differentiated twice (the fused kernels cannot)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.q, self.kv, self.out = heads, nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        q = self.q(x).view(B, N, self.heads, -1).transpose(1, 2)
        k, v = self.kv(mem).view(B, mem.shape[1], 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) * (d // self.heads) ** -0.5
        return self.out((a.softmax(-1) @ v).transpose(1, 2).reshape(B, N, d))


class EnergyNet(nn.Module):
    """E(z; x) of Energy-Based Reasoners (Chen et al. 2026): slot latents z [B, S, d] and the state cache x -> one
    energy per slot, summed. With `direct` the same network returns an update of z instead (control without an energy)."""

    def __init__(self, d: int, heads: int, layers: int, slots: int, direct: bool):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(slots, d))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(nn.ModuleDict({
            "n1": nn.LayerNorm(d), "self": _Attention(d, heads), "n2": nn.LayerNorm(d), "cross": _Attention(d, heads),
            "n3": nn.LayerNorm(d), "mlp": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))}) for _ in range(layers))
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, d if direct else 1)
        self.direct = direct

    def forward(self, z: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        h = z + self.pos
        for l in self.layers:
            a = l["n1"](h)
            h = h + l["self"](a, a)
            h = h + l["cross"](l["n2"](h), cache)
            h = h + l["mlp"](l["n3"](h))
        h = self.head(self.norm(h))
        return h if self.direct else h.squeeze(-1).sum(1)                 # update [B, S, d] or energy [B]


class ForcePolicy(nn.Module):
    def __init__(self, cfg: dict, stats: dict, n_joints: int, n_skills: int, n_tasks: int,
                 vis_tokens: int, vis_dim: int):
        super().__init__()
        m, f = cfg["model"], cfg["force"]
        d, K, H = m["d_model"], m["obs_history"], m["chunk"]
        self.use_tactile, self.use_torque = f["tactile_input"], f["torque_input"]
        self.H, self.J, self.K = H, n_joints, K
        self.force_scale = cfg["train"]["force_scale_n"]
        # predict the change from the force measured now (only possible when the model sees that force)
        self.force_residual = bool(m.get("force_residual", False)) and self.use_tactile
        # tiny world model: the force change is predicted from the latent AND the action chunk (see force_from_actions)
        wm = m.get("force_world_model", {}) or {}
        self.world_model = bool(wm.get("enabled", False)) and self.use_tactile and f["force_output"]
        assert not (self.world_model and m.get("head", "regression") == "flow"), "the force world model is a regression-head option"
        self.wm_pred_prob = float(wm.get("predicted_action_prob", 0.0))
        self.force_residual = self.force_residual or self.world_model
        for k, v in stats.items():
            self.register_buffer(k, v.float())
        # which joint torques the model may read: all / arms / hands (others are set to their mean)
        which = f.get("torque_joints", "all")
        idx = torch.arange(58)
        is_arm = (idx < 7) | ((idx >= 29) & (idx < 36))
        self.torque_keep = {"all": torch.ones(58, dtype=torch.bool), "arms": is_arm, "hands": ~is_arm}[which].float()

        n_cam = len(cfg["vision"]["cameras"])
        self.vis_proj = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, d))
        self.vis_pos = nn.Parameter(torch.zeros(n_cam, vis_tokens, d))
        self.state_in = nn.Linear(n_joints, d)
        self.tactile_in = nn.Linear(60, d)
        self.torque_in = nn.Linear(58, d)
        self.time_emb = nn.Parameter(torch.zeros(K, d))                   # shared by the three histories
        self.type_emb = nn.Parameter(torch.zeros(3, d))                   # state / tactile / torque
        self.null_tactile = nn.Parameter(torch.zeros(d))                  # "force not available" tokens
        self.null_torque = nn.Parameter(torch.zeros(d))
        self.task_emb = nn.Embedding(n_tasks, d)

        layer = dict(d_model=d, nhead=m["n_heads"], dim_feedforward=d * m["ff_mult"], dropout=m["dropout"],
                     batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**layer), m["enc_layers"])
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**layer), m["dec_layers"])
        self.enc_norm, self.dec_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.slots = nn.Parameter(torch.zeros(H + 3, d))                  # H chunk slots + 3 decision slots

        self.n_experts = m["n_experts"]
        # "mixture": softmax-weighted experts. "shared_residual": expert 0 always on, the others add a gated correction.
        self.layout = m.get("expert_layout", "mixture")
        self.gate_source = m.get("gate_source", "learned")               # "contact": gate = predicted contact-soon probability
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_joints))
                                     for _ in range(self.n_experts))
        # FACET-style wrench expert (Facet-0, Sec. 4.1): a kinematic token E_s and wrench-history tokens E_w (two-layer
        # MLP over a causal window, here one token per fingertip sensor) condition a joint action-wrench proposal.
        we = m.get("wrench_expert", {}) or {}
        self.wrench_expert = bool(we.get("enabled", False)) and self.use_tactile
        if self.wrench_expert:
            assert self.layout == "mixture", "the wrench expert joins the softmax mixture"
            self.wr_tokens = bool(we.get("wrench_tokens", True))
            self.kin_in = nn.Linear(n_joints, d)                                          # E_s
            self.wr_in, self.wr_out = nn.Linear(int(we["window"]) * 6, d), nn.Linear(d, d)  # E_w, shared by the ten fingertips
            self.wr_tip = nn.Parameter(torch.zeros(10, d))
            nn.init.normal_(self.wr_tip, std=0.02)
            self.wr_norm = nn.LayerNorm(d)
            self.wr_attn = nn.MultiheadAttention(d, m["n_heads"], batch_first=True)
            # regression: 58 joint targets + 60-D next wrench; flow: velocity of the same joint chunk as the main head
            self.wr_head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                         nn.Linear(d, n_joints + (30 if m.get("head", "regression") == "flow" else 60)))
        # Tactile world-model expert: tokens are TacForeSight latents of the fingertip images (last 8 frames) and the latents it
        # predicts for the next 8 frames (scripts/tf_features.py). The world model is frozen; only this expert reads them.
        te = m.get("tactile_expert", {}) or {}
        self.tactile_expert = bool(te.get("enabled", False)) and self.use_tactile
        if self.tactile_expert:
            assert self.layout == "mixture" and m.get("head", "regression") == "regression", "the tactile expert joins the regression mixture"
            self.te_predicted = bool(te.get("predicted_tokens", True))
            n_tok, dim = int(te["tokens"]), int(te["latent_dim"])
            self.te_in = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d))
            self.te_pos = nn.Parameter(torch.zeros(n_tok, d))
            nn.init.normal_(self.te_pos, std=0.02)
            self.te_norm = nn.LayerNorm(d)
            self.te_attn = nn.MultiheadAttention(d, m["n_heads"], batch_first=True)
            self.te_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_joints))
        self.gate = nn.Linear(3 * d, self.n_experts + int(self.wrench_expert) + int(self.tactile_expert))
        if self.layout == "shared_residual":                              # routed experts start as a zero correction
            for e in self.experts[1:]:
                nn.init.zeros_(e[-1].weight); nn.init.zeros_(e[-1].bias)
        self.force_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 30))
        # flow matching: the chunk slots carry the noisy joint action-force chunk x_tau and predict its velocity
        self.flow = m.get("head", "regression") == "flow"
        self.advantage = False
        self.predict_force = bool(f["force_output"])
        if self.flow:
            fl = m["flow"]
            self.flow_steps, self.flow_beta, self.flow_seed = int(fl["sample_steps"]), tuple(fl["time_beta"]), int(fl["sample_seed"])
            # The flow mixes its targets with unit-size noise, so every target must be about unit size. "auto": the force
            # channels are divided by the size of the force target in the training data instead of train.force_scale_n.
            if fl.get("force_scale") == "auto" and "force_rms" in stats:
                self.force_scale = float(stats["force_change_rms" if self.force_residual else "force_rms"])
            elif fl.get("force_scale") not in (None, "auto"):
                self.force_scale = float(fl["force_scale"])
            self.x_in = nn.Linear(n_joints + 30, d)
            # Facet-0 stage 2: a learned tag ("positive credit") added to the chunk queries of a second flow branch
            self.advantage = bool((m.get("advantage") or {}).get("enabled", False))
            if self.advantage:
                self.adv_emb = nn.Parameter(torch.zeros(d))
                nn.init.normal_(self.adv_emb, std=0.02)
            self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        if self.world_model:
            h = int(wm.get("hidden", 192))
            self.wm_in = nn.Sequential(nn.Linear(d + n_joints, h), nn.GELU())
            self.wm_rnn = nn.GRU(h, h, batch_first=True)
            self.wm_out = nn.Linear(h, 30)
        # Energy-based reasoning (EBR) in the latent space: the slot latents are refined by gradient descent on a learned
        # scalar energy E(z; state cache) before the heads read them. See descend().
        en = m.get("energy", {}) or {}
        self.energy = bool(en.get("enabled", False))
        if self.energy:
            assert not self.flow and not self.world_model, "energy refinement is a regression-head option"
            self.er_steps, self.er_graded, self.er_step_size = int(en["steps"]), int(en["graded_steps"]), float(en["step_size"])
            self.er_noise, self.er_init, self.er_explore, self.er_samples = float(en["noise"]), en["init"], int(en["explore_k"]), int(en["eval_samples"])
            self.er_seed = int(en["eval_seed"])
            self.energy_net = EnergyNet(d, m["n_heads"], int(en["layers"]), H + 3, en["update"] == "direct")
        # Heavy-tailed residual loss (HT-Policy, Zhou et al. 2026): the action residual of a chunk is modelled as a multivariate
        # Student-t whose scale sigma is predicted per chunk by this head. Zero-initialised; sigma starts at the residual size
        # measured before training (engine.calibrate_ht -> ht_scale_bias). Only used in the loss, never at inference.
        ht = cfg["train"].get("ht", {}) or {}
        self.ht = cfg["train"].get("action_loss", "l1") == "ht"
        if self.ht:
            assert not self.flow, "the heavy-tailed loss is a regression-head option"
            self.ht_nu_per_dim, self.ht_min_scale = float(ht["nu_per_dim"]), float(ht["min_scale"])
            self.scale_head = nn.Linear(d, 1)
            nn.init.zeros_(self.scale_head.weight); nn.init.zeros_(self.scale_head.bias)
            self.register_buffer("ht_scale_bias", torch.zeros(()))
        self.contact_head = nn.Linear(d, 1)                               # yes/no
        self.phase_head = nn.Linear(d, n_skills)                          # choice
        n_thr = len(m["decision"]["force_levels_n"])
        self.level_head = nn.Linear(d, 1)                                 # ordinal: one score, ordered thresholds
        self.level_b0 = nn.Parameter(torch.zeros(1))
        self.level_gaps = nn.Parameter(torch.zeros(n_thr - 1))
        for p in (self.vis_pos, self.time_emb, self.type_emb, self.null_tactile, self.null_torque, self.slots):
            nn.init.normal_(p, std=0.02)
        # second embodiment: an expert that imitates the C3 contact MPC. It has its own input projection
        # and output heads; the encoder, the decoder, the query slots and the contact-soon head are shared.
        self.has_mpc = bool(cfg.get("mpc", {}).get("enabled", False))
        if self.has_mpc:
            self.mpc_in = nn.Linear(stats["mpc_obs_mean"].numel(), d)
            self.mpc_emb = nn.Parameter(torch.zeros(2, d))                # history-token type, embodiment token
            nn.init.normal_(self.mpc_emb, std=0.02)
            n_u, n_f = stats["mpc_u_std"].numel(), stats["mpc_force_std"].numel()
            self.mpc_expert = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_u))
            self.mpc_force_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_f))

    def encode(self, b: dict, zero_force: bool | str = False) -> torch.Tensor:
        B = len(b["state"])
        vis = self.vis_proj(b["feats"].float()) + self.vis_pos            # [B, C, T, d]
        toks = [vis.flatten(1, 2)]
        st = (b["state"] - self.state_mean) / self.state_std
        toks.append(self.state_in(st) + self.time_emb + self.type_emb[0])
        tac, tor = b["tactile"], b["torque"]
        if zero_force in (True, "tactile"):                               # measured zero, not "unavailable"
            tac = torch.zeros_like(tac)
        if zero_force in (True, "torque"):
            tor = torch.zeros_like(tor)
        if self.use_tactile:
            toks.append(self.tactile_in(tac / self.tactile_std) + self.time_emb + self.type_emb[1])
        else:
            toks.append((self.null_tactile + self.time_emb + self.type_emb[1]).expand(B, -1, -1))
        if self.use_torque:
            tor = (tor - self.torque_mean) / self.torque_std * self.torque_keep.to(tor.device)
            toks.append(self.torque_in(tor) + self.time_emb + self.type_emb[2])
        else:
            toks.append((self.null_torque + self.time_emb + self.type_emb[2]).expand(B, -1, -1))
        toks.append(self.task_emb(b["task_id"])[:, None])
        return self.enc_norm(self.encoder(torch.cat(toks, 1)))            # the state cache

    def force_from_actions(self, chunk: torch.Tensor, delta_norm: torch.Tensor) -> torch.Tensor:
        """World model: slot latents [B, H, d] + commanded displacement (joint target - joint position now,
        normalised) [B, H, J] -> change of the fingertip force at each step / force_scale, [B, H, 30].
        A recurrent state carries the contact state forward through the chunk."""
        with torch.autocast("cuda", enabled=False):
            x = self.wm_in(torch.cat([chunk.float(), delta_norm.float()], -1))
            return self.wm_out(self.wm_rnn(x)[0])

    def forward(self, b: dict, zero_force: bool | str = False, wm_actions: str | None = None) -> dict:
        """wm_actions (world model only): which actions the force forecast is conditioned on.
        recorded / predicted / zero / shuffled. Default: training -> recorded, mixed with the model's own
        (detached) prediction with probability `predicted_action_prob`; evaluation -> predicted."""
        cache = self.encode(b, zero_force)
        now = b["tactile"][:, -1].view(-1, 10, 6)[..., :3].flatten(1)[:, None]      # [B, 1, 30] N
        base = now if self.force_residual and zero_force not in (True, "tactile") else torch.zeros_like(now)
        if self.flow:
            return self.forward_flow(b, cache, base, zero_force)
        if not self.energy:
            q = self.dec_norm(self.decoder(self.slots.expand(len(cache), -1, -1), cache))
            return self.decode(q, b, base, zero_force, wm_actions)
        return self.forward_energy(b, cache, base, zero_force)

    # ---- energy-based reasoning ---------------------------------------------------------------------------
    def descend(self, z: torch.Tensor, cache: torch.Tensor, noise: torch.Tensor, graded: int = 0):
        """len(noise) descent steps  z <- sphere(sphere(z) + noise - step * dE/dz)  from z [B, S, d].
        The last `graded` steps keep the graph through the gradient (second order), so a loss on their states trains the
        energy; the steps before them are run without a graph (EBR's truncated window). Returns the final state and the
        graded states. `update: direct` replaces - step * dE/dz by the network's own output (no energy)."""
        cache, z, states = cache.float(), z.float().detach(), []
        step = self.er_step_size * (0.8 + 0.2 * torch.rand(len(z), 1, 1, device=z.device) if self.training else 1.0)
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            for t in range(len(noise)):
                g = t >= len(noise) - graded
                v = sphere(z) + noise[t]
                if self.energy_net.direct:
                    z = sphere(v + (self.energy_net(v, cache) if g else self.energy_net(v, cache).detach()))
                else:
                    if not v.requires_grad:
                        v = v.requires_grad_(True)
                    (grad,) = torch.autograd.grad(self.energy_net(v, cache if g else cache.detach()).sum(), v, create_graph=g)
                    z = sphere(v - step * grad)
                if g:
                    states.append(z)
                else:
                    z = z.detach()
        return z, states

    def forward_energy(self, b: dict, cache: torch.Tensor, base: torch.Tensor, zero_force) -> dict:
        """Start state: the decoder's slots (`init: decoder`) or noise (`init: noise`, as in EBR). Training: every graded
        state is decoded and supervised (out["interior"]); with explore_k > 1 several noise draws are descended first
        and only the one closest to the recorded action is trained (EBR's forward SXM). Evaluation: with eval_samples > 1
        the sample with the lowest final energy is kept."""
        B, S, d = len(cache), self.H + 3, cache.shape[-1]
        dev = cache.device
        gen = None if self.training else torch.Generator(device=dev).manual_seed(self.er_seed)
        steps = self.er_steps
        z_dec = sphere(self.dec_norm(self.decoder(self.slots.expand(B, -1, -1), cache)).float()) if self.er_init == "decoder" else None

        def draw():                                                       # start state + per-step noise of one trajectory
            z0 = z_dec if z_dec is not None else torch.randn(B, S, d, device=dev, generator=gen)
            return z0, torch.randn(steps, B, S, d, device=dev, generator=gen) * self.er_noise

        def dec(z):
            return self.decode(z, b, base, zero_force, None)

        if self.training:
            z0, noise = draw()
            if self.er_explore > 1:                                       # forward SXM: train the closest of K trajectories
                tgt = (b["action"] - b["state"][:, -1:, :]) / self.delta_std
                av = b["action_valid"][..., None].float()
                cands, errs = [(z0, noise)], []
                cands += [draw() for _ in range(self.er_explore - 1)]
                with torch.no_grad():
                    for zc, nc in cands:
                        o = dec(self.descend(zc, cache, nc)[0])
                        errs.append(((o["delta_norm"].float() - tgt).abs() * av).sum((1, 2)))
                win, rows = torch.stack(errs).argmin(0), torch.arange(B, device=dev)
                z0 = torch.stack([c[0] for c in cands])[win, rows]
                noise = torch.stack([c[1] for c in cands])[win, :, rows].transpose(0, 1)
            z, states = self.descend(z0, cache, noise, graded=min(self.er_graded, steps))
            out = dec(z)
            out["interior"] = [dec(s) for s in states[:-1]] + ([dec(z_dec)] if z_dec is not None else [])
            return out
        n = max(self.er_samples, 1)
        finals, energies = [], []
        for _ in range(n):
            z0, noise = draw()
            z = self.descend(z0, cache, noise)[0].detach()
            finals.append(z)
            if not self.energy_net.direct:
                energies.append(self.energy_net(z, cache.float()).detach())
        rows = torch.arange(B, device=dev)
        if n > 1 and energies:
            pick = torch.stack(energies).argmin(0)
            z, energy = torch.stack(finals)[pick, rows], torch.stack(energies)[pick, rows]
        else:
            z, energy = finals[0], (energies[0] if energies else torch.zeros(B, device=dev))
        out = dec(z)
        out["energy"] = energy
        if z_dec is not None:                                             # the same model before any descent step
            out["start"] = dec(z_dec)
        return out

    def decode(self, q: torch.Tensor, b: dict, base: torch.Tensor, zero_force, wm_actions) -> dict:
        """Slot latents [B, H + 3, d] -> every output of the regression policy."""
        chunk, dec = q[:, :self.H], q[:, self.H:]                         # [B, H, d], [B, 3, d]
        contact_logit = self.contact_head(dec[:, 0]).squeeze(-1)
        if self.layout == "shared_residual":
            if self.gate_source == "contact":                             # read-out of the decision slot, no gradient from the action
                routed = torch.sigmoid(contact_logit.float()).detach()[:, None].expand(-1, self.n_experts - 1)
            else:
                routed = torch.sigmoid(self.gate(dec.flatten(1)).float())[:, 1:]
            gate = torch.cat([torch.ones_like(routed[:, :1]), routed], 1)
        else:
            gate = F.softmax(self.gate(dec.flatten(1)), -1)               # [B, n_experts]
        delta = sum(gate[:, k, None, None] * self.experts[k](chunk) for k in range(self.n_experts))
        extra = {}
        if self.wrench_expert:
            y = self.wrench_proposal(chunk, self.contact_tokens(b, zero_force))
            extra = {"wrench_action_norm": y[..., :self.J], "wrench_norm": y[..., self.J:]}          # joint proposal
            delta = delta + gate[:, self.n_experts, None, None] * y[..., :self.J]
        if self.tactile_expert:                                           # last expert of the mixture
            toks = self.te_in(b["tac_latent"].float()) + self.te_pos
            if not self.te_predicted:                                     # control: the predicted half is hidden
                toks = toks[:, :toks.shape[1] // 2]
            h = chunk + self.te_attn(self.te_norm(chunk), toks, toks, need_weights=False)[0]
            extra["tactile_action_norm"] = self.te_head(h)
            delta = delta + gate[:, -1, None, None] * extra["tactile_action_norm"]
        if self.world_model:
            mode = wm_actions or ("train" if self.training else "predicted")
            if mode == "predicted":
                a = delta.detach()
            elif mode == "zero":
                a = torch.zeros_like(delta)
            else:
                a = (b["action"] - b["state"][:, -1:, :]) / self.delta_std
                if mode == "shuffled":                                   # actions of other frames in the batch
                    a = a[torch.randperm(len(a), device=a.device)]
                elif mode == "train" and self.wm_pred_prob > 0:
                    own = torch.rand(len(a), 1, 1, device=a.device) < self.wm_pred_prob
                    a = torch.where(own, delta.detach().float(), a)
            force_norm = self.force_from_actions(chunk, a)
        else:
            force_norm = self.force_head(chunk)
        thresholds = self.level_b0 + torch.cat([self.level_gaps.new_zeros(1), F.softplus(self.level_gaps).cumsum(0)])
        if self.ht:                                                       # one residual scale per chunk, from the pooled chunk slots
            extra["sigma"] = F.softplus(self.scale_head(chunk.mean(1)).float().squeeze(-1) + self.ht_scale_bias) + self.ht_min_scale
        return {"delta_norm": delta,                                      # [B, H, J] normalised joint-target delta
                "force_norm": force_norm,                                 # [B, H, 30] (force - base) / force_scale
                "force_base": base,
                "contact_logit": contact_logit,
                "phase_logits": self.phase_head(dec[:, 1]),
                "level_logits": self.level_head(dec[:, 2]) - thresholds,  # [B, n_thr]: logit of P(level > k)
                "gate": gate, **extra}

    def contact_tokens(self, b: dict, zero_force: bool | str = False) -> torch.Tensor:
        """Facet-0's E_s and E_w: one kinematic token and one wrench-history token per fingertip, [B, 1 + 10, d]."""
        win = b["wrench_hist"]                                            # [B, W, 60]
        if zero_force in (True, "tactile"):
            win = torch.zeros_like(win)
        win = (win / self.tactile_std).view(len(win), -1, 10, 6).transpose(1, 2).flatten(2)          # [B, 10 tips, W * 6]
        kin = self.kin_in((b["state"][:, -1] - self.state_mean) / self.state_std)[:, None]
        if not self.wr_tokens:
            return kin
        return torch.cat([kin, self.wr_out(F.silu(self.wr_in(win))) + self.wr_tip], 1)

    def wrench_proposal(self, chunk: torch.Tensor, toks: torch.Tensor) -> torch.Tensor:
        h = chunk + self.wr_attn(self.wr_norm(chunk), toks, toks, need_weights=False)[0]
        return self.wr_head(h)

    # ---- flow matching ------------------------------------------------------------------------------------
    def _time_emb(self, tau: torch.Tensor) -> torch.Tensor:
        half = self.t_mlp[0].in_features // 2
        freq = torch.exp(-torch.arange(half, device=tau.device) * (torch.log(torch.tensor(1000.0)) / (half - 1)))
        a = tau.float()[:, None] * 1000.0 * freq
        return self.t_mlp(torch.cat([a.sin(), a.cos()], -1).to(self.t_mlp[0].weight.dtype))

    def velocity(self, cache, gate, x, tau, toks=None, tagged=False):
        """x [B, H, J + 30] at noise level tau [B] -> velocity of the action part through the gated experts,
        of the force part through the force head; and the wrench expert's own joint velocity (or None)."""
        q = self.x_in(x.to(cache.dtype)) + self.slots[:self.H] + self._time_emb(tau)[:, None].to(cache.dtype)
        if tagged:
            q = q + self.adv_emb.to(q.dtype)
        q = self.dec_norm(self.decoder(q, cache))
        va = sum(gate[:, k, None, None] * self.experts[k](q) for k in range(self.n_experts))
        own = None
        if self.wrench_expert:
            own = self.wrench_proposal(q, toks)
            va = va + gate[:, self.n_experts, None, None] * own[..., :self.J]
        vf = self.force_head(q) if self.predict_force else torch.zeros_like(x[..., self.J:]).to(va.dtype)
        return torch.cat([va, vf], -1), own

    def forward_flow(self, b, cache, base, zero_force):
        """Training: one flow-matching sample (velocity + target). Evaluation: the chunk integrated from noise."""
        B = len(cache)
        dec = self.dec_norm(self.decoder(self.slots[self.H:].expand(B, -1, -1), cache))   # decision slots never see the noise
        contact_logit = self.contact_head(dec[:, 0]).squeeze(-1)
        gate = F.softmax(self.gate(dec.flatten(1)), -1)
        toks = self.contact_tokens(b, zero_force) if self.wrench_expert else None
        thresholds = self.level_b0 + torch.cat([self.level_gaps.new_zeros(1), F.softplus(self.level_gaps).cumsum(0)])
        out = {"force_base": base, "contact_logit": contact_logit, "phase_logits": self.phase_head(dec[:, 1]),
               "level_logits": self.level_head(dec[:, 2]) - thresholds, "gate": gate}
        shape = (B, self.H, self.J + 30)
        if self.training:
            y = torch.cat([(b["action"] - b["state"][:, -1:, :]) / self.delta_std, (b["force"] - base) / self.force_scale], -1)
            if not self.predict_force:                                    # the force channels stay empty
                y[..., self.J:] = 0
            a, c = self.flow_beta

            def draw(tagged):                                             # one noise level and noise sample per branch
                noise = torch.randn(shape, device=cache.device)
                if not self.predict_force:
                    noise[..., self.J:] = 0
                tau = torch.distributions.Beta(a, c).sample((B,)).to(cache.device) * 0.999 + 0.001
                x = tau[:, None, None] * noise + (1 - tau[:, None, None]) * y
                v, own = self.velocity(cache, gate, x, tau, toks, tagged)
                return v, noise - y, own

            out["flow_v"], out["flow_u"], out["flow_own"] = draw(False)
            if self.advantage:
                out["flow_v_tag"], out["flow_u_tag"], out["flow_own_tag"] = draw(True)
            return out
        g = torch.Generator(device=cache.device).manual_seed(self.flow_seed)
        x = torch.randn(shape, device=cache.device, generator=g) * getattr(self, "flow_noise", 1.0)   # 0 = start from the centre of the noise
        if not self.predict_force:
            x[..., self.J:] = 0
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):                                  # tau: 1 -> 0
            tau = torch.full((B,), 1.0 - i * dt, device=cache.device)
            v = self.velocity(cache, gate, x, tau, toks)[0].float()
            if self.advantage and getattr(self, "guidance", 1.0) != 0:   # plain + guidance * (tagged - plain)
                v = v + self.guidance * (self.velocity(cache, gate, x, tau, toks, True)[0].float() - v)
            x = x - dt * v
        out |= {"delta_norm": x[..., :self.J], "force_norm": x[..., self.J:]}
        return out

    def forward_mpc(self, b: dict) -> dict:
        """MPC embodiment: obs history [B, K, 9] -> input plan and planned contact forces (normalised)."""
        obs = (b["obs"] - self.mpc_obs_mean) / self.mpc_obs_std
        toks = torch.cat([self.mpc_in(obs) + self.time_emb + self.mpc_emb[0],
                          self.mpc_emb[1].expand(len(obs), 1, -1)], 1)
        cache = self.enc_norm(self.encoder(toks))
        q = self.dec_norm(self.decoder(self.slots.expand(len(cache), -1, -1), cache))
        chunk = q[:, :self.H]
        return {"u_norm": self.mpc_expert(chunk), "force_norm": self.mpc_force_head(chunk),
                "contact_logit": self.contact_head(q[:, self.H]).squeeze(-1)}

    def targets(self, b: dict, out: dict) -> dict:
        """Loss targets in the model's normalised coordinates."""
        delta = (b["action"] - b["state"][:, -1:, :]) / self.delta_std    # relative to the current joint state
        tgt = {"delta_norm": delta, "force_norm": (b["force"] - out["force_base"]) / self.force_scale}
        if self.wrench_expert and not self.flow:
            tgt["wrench_norm"] = b["wrench_target"] / self.tactile_std    # next-step filtered wrench, [B, H, 60]
        return tgt

    def to_physical(self, out: dict, b: dict) -> dict:
        p = {"action": out["delta_norm"].float() * self.delta_std + b["state"][:, -1:, :],   # rad
             "force": out["force_norm"].float() * self.force_scale + out["force_base"]}                           # N
        if self.wrench_expert and "wrench_norm" in out:                   # fingertip forces of the wrench expert's proposal
            w = (out["wrench_norm"].float() * self.tactile_std).view(*out["wrench_norm"].shape[:2], 10, 6)
            p["wrench_force"] = w[..., :3].flatten(2)
        return p


def ht_loss(pred: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor, valid: torch.Tensor, nu_per_dim: float) -> torch.Tensor:
    """Heteroscedastic multivariate Student-t loss of HT-Policy (github.com/the-labone/regression-policy, ht_loss/functional.py).
    pred, target [B, H, J]; valid [B, H, 1]; sigma [B], one scale per chunk. Per chunk, with d valid elements, S the sum of
    squared residuals and nu = nu_per_dim * d:   L = 0.5 (nu + d) log(1 + S / (nu sigma^2)) + d log(sigma);  returns sum L / sum d."""
    with torch.autocast("cuda", enabled=False):
        w = valid.float().expand_as(pred)
        count = w.flatten(1).sum(1)
        S = ((pred.float() - target.float()).square() * w).flatten(1).sum(1)
        nu = nu_per_dim * count.clamp(min=1)
        per = 0.5 * (nu + count) * torch.log1p(S / (nu * sigma.float().square())) + count * sigma.float().log()
        return per.sum() / count.sum()


def loss_fn(model: ForcePolicy, out: dict, b: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    f, w = cfg["force"], cfg["train"]
    av, fv = b["action_valid"][..., None].float(), b["force_valid"][..., None].float()
    if model.flow:                # joint action-force flow matching: squared velocity error, element means, force weight 0.1
        J, fw = model.J, cfg["model"]["flow"]["force_weight"]

        def joint(v):
            e = (v.float() - out["flow_u"]) ** 2
            return (e[..., :J] * av).sum() / (av.sum() * J), (e[..., J:] * fv).sum() / (fv.sum() * 30)

        if model.advantage:       # Facet-0 refinement: weight * (plain branch + label * tagged branch), per sample
            def per_sample(v, u, own):
                def parts(x):
                    e = (x.float() - u) ** 2
                    return ((e[..., :J] * av).sum((1, 2)) / (av.sum((1, 2)).clamp(min=1) * J),
                            (e[..., J:] * fv).sum((1, 2)) / (fv.sum((1, 2)).clamp(min=1) * 30))
                a, fo = parts(v)
                l = w["w_action"] * a + (fw * fo if f["force_output"] else 0)
                if model.wrench_expert:
                    a2, f2 = parts(own)
                    l = l + cfg["model"]["wrench_expert"]["weight"] * (a2 + (fw * f2 if f["force_output"] else 0))
                return l, a, fo

            l_plain, a, fo = per_sample(out["flow_v"], out["flow_u"], out["flow_own"])
            l_tag, a_t, _ = per_sample(out["flow_v_tag"], out["flow_u_tag"], out["flow_own_tag"])
            label = b["adv_label"].float()
            weight = 1.0 + cfg["model"]["advantage"]["contact_weight"] * b["adv_intensity"]
            loss = (weight * (l_plain + label * l_tag)).mean()
            logs = {"action": a.mean().item(), "force": fo.mean().item(),
                    "action_tagged": ((label * a_t).sum() / label.sum().clamp(min=1)).item()}
            l_dec = F.cross_entropy(out["phase_logits"].float(), b["skill"])
            if f["decision_supervision"]:
                l_dec = l_dec + F.binary_cross_entropy_with_logits(out["contact_logit"].float(), b["contact_soon"])
                above = (b["level"][:, None] > torch.arange(out["level_logits"].shape[1], device=b["level"].device)).float()
                l_dec = l_dec + F.binary_cross_entropy_with_logits(out["level_logits"].float(), above)
            logs["decision"] = l_dec.item()
            return loss + w["w_decision"] * l_dec, logs
        l_act, l_force = joint(out["flow_v"])
        logs = {"action": l_act.item()}
        loss = w["w_action"] * l_act
        if f["force_output"]:
            loss = loss + fw * l_force
            logs["force"] = l_force.item()
        if model.wrench_expert:   # the expert's own proposal gets the same joint objective
            l_a, l_w = joint(out["flow_own"])
            loss = loss + cfg["model"]["wrench_expert"]["weight"] * (l_a + (fw * l_w if f["force_output"] else 0))
            logs["wrench_expert_action"], logs["wrench_expert_wrench"] = l_a.item(), l_w.item()
    else:
        tgt = model.targets(b, out)
        l_act = ((out["delta_norm"].float() - tgt["delta_norm"]).abs() * av).sum() / (av.sum() * model.J)
        logs = {"action": l_act.item()}                                   # always the L1 error, whatever is trained
        if model.ht:              # heavy-tailed residual loss replaces L1 on the action; no extra multiplier (as in HT-Policy)
            l_ht = ht_loss(out["delta_norm"], tgt["delta_norm"], out["sigma"], av, model.ht_nu_per_dim)
            loss = w["w_action"] * l_ht
            logs["ht"], logs["sigma"] = l_ht.item(), out["sigma"].mean().item()
        else:
            loss = w["w_action"] * l_act
        if f["force_output"]:
            l_force = ((out["force_norm"].float() - tgt["force_norm"]).abs() * fv).sum() / (fv.sum() * 30)
            loss = loss + w["w_force"] * l_force
            logs["force"] = l_force.item()
        if out.get("interior"):   # EBR: the states before the last descent step are supervised too
            l_in = 0
            for o in out["interior"]:
                l_in = l_in + w["w_action"] * ((o["delta_norm"].float() - tgt["delta_norm"]).abs() * av).sum() / (av.sum() * model.J)
                if f["force_output"]:
                    l_in = l_in + w["w_force"] * ((o["force_norm"].float() - tgt["force_norm"]).abs() * fv).sum() / (fv.sum() * 30)
            l_in = l_in / len(out["interior"])
            loss = loss + cfg["model"]["energy"]["interior_weight"] * l_in
            logs["interior"] = l_in.item()
    l_dec = F.cross_entropy(out["phase_logits"].float(), b["skill"])     # phase labels are annotation, not force
    if f["decision_supervision"]:
        l_dec = l_dec + F.binary_cross_entropy_with_logits(out["contact_logit"].float(), b["contact_soon"])
        n_thr = out["level_logits"].shape[1]
        above = (b["level"][:, None] > torch.arange(n_thr, device=b["level"].device)).float()
        l_dec = l_dec + F.binary_cross_entropy_with_logits(out["level_logits"].float(), above)
    loss = loss + w["w_decision"] * l_dec
    logs["decision"] = l_dec.item()
    if model.wrench_expert and not model.flow:   # FACET joint action-wrench objective on the expert's own proposal: element means, wrench weight 0.1
        we = cfg["model"]["wrench_expert"]
        l_a = ((out["wrench_action_norm"].float() - tgt["delta_norm"]).abs() * av).sum() / (av.sum() * model.J)
        l_w = ((out["wrench_norm"].float() - tgt["wrench_norm"]).abs() * fv).sum() / (fv.sum() * 60)
        logs["wrench_expert_action"], logs["wrench_expert_wrench"] = l_a.item(), l_w.item()
        if model.ht:              # the expert's own action term uses the same residual model; the chunk's scale is a constant here
            l_a = ht_loss(out["wrench_action_norm"], tgt["delta_norm"], out["sigma"].detach(), av, model.ht_nu_per_dim)
        loss = loss + we["weight"] * (l_a + we["wrench_weight"] * l_w)
    if model.tactile_expert:                     # the tactile expert's own proposal gets the action objective too
        l_t = ((out["tactile_action_norm"].float() - tgt["delta_norm"]).abs() * av).sum() / (av.sum() * model.J)
        logs["tactile_expert_action"] = l_t.item()
        if model.ht:
            l_t = ht_loss(out["tactile_action_norm"], tgt["delta_norm"], out["sigma"].detach(), av, model.ht_nu_per_dim)
        loss = loss + cfg["model"]["tactile_expert"]["weight"] * l_t
    return loss, logs


def mpc_loss_fn(model: ForcePolicy, out: dict, b: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """Imitation of the MPC plan: same three terms and weights as the robot loss."""
    w, n = cfg["train"], b["u"].shape[1]
    l_u = (out["u_norm"][:, :n].float() - b["u"] / model.mpc_u_std).abs().mean()
    l_f = (out["force_norm"][:, :n].float() - b["force"] / model.mpc_force_std).abs().mean()
    l_c = F.binary_cross_entropy_with_logits(out["contact_logit"].float(), b["contact_soon"])
    return w["w_action"] * l_u + w["w_force"] * l_f + w["w_decision"] * l_c, {"mpc_u": l_u.item(), "mpc_force": l_f.item()}
