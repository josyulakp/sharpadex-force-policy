"""Shared pieces of the video world model (Cosmos-Predict2 2B, action-conditioned on joint commands).

Run the wm_train / wm_rollout scripts with the Cosmos environment:  <src>/.venv/bin/python scripts/wm_train.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sharpa.config import load_config  # noqa: E402

CFG = load_config()
W = CFG["video_world_model"]
os.environ.setdefault("COSMOS_PREDICT2_ARGS", f"--checkpoints {W['src']}/checkpoints")
sys.path.insert(0, W["src"])
STEPS = W["chunk_frames"] - 1                 # predicted frames per chunk
ACTION_DIM = 58 * STEPS


def chunk_action(action: np.ndarray, state0: np.ndarray) -> np.ndarray:
    """Joint targets of the predicted frames [STEPS, 58] and the joint state at the conditioning frame [58]
    -> the model's action input [STEPS, 58]."""
    return ((action - state0) / W["action_scale_rad"]).astype(np.float32)


def build_model(weights: str | None = None, lora_rank: int = 0):
    """The action-conditioned model with its action input widened from 12 x 7 to 12 x 58.
    weights: a file written by wm_train.py (trainable parameters only)."""
    from torch import nn

    from cosmos_predict2.configs.action_conditioned.config import get_cosmos_predict2_action_conditioned_pipeline
    from cosmos_predict2.models.video2world_action_model import Predict2Video2WorldActionConditionedModel
    from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig, Predict2Video2WorldModelConfig
    from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
    from imaginaire.constants import get_cosmos_predict2_action_conditioned_checkpoint

    pipe_cfg = get_cosmos_predict2_action_conditioned_pipeline(model_size="2B", resolution="480", fps=4)
    pipe_cfg.guardrail_config.enabled = False
    pipe_cfg.prompt_refiner_config.enabled = False
    dit_path = get_cosmos_predict2_action_conditioned_checkpoint(model_size="2B", resolution="480", fps=4)

    class Model(Predict2Video2WorldActionConditionedModel):
        def __init__(self, config):                                   # as the parent, without text encoder / FSDP
            torch.nn.Module.__init__(self)
            self.config = config
            self.precision = torch.bfloat16
            self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
            self.device = torch.device("cuda")
            self.setup_data_key()
            self.loss_reduce, self.loss_scale = "mean", config.loss_scale
            self.video_noise_multiplier = float(np.sqrt(config.pipe_config.state_t)) if config.adjust_video_noise else 1.0
            self.data_parallel_size = 1
            self.pipe = Video2WorldActionConditionedPipeline.from_config(config.pipe_config, dit_path=dit_path, use_text_encoder=False)

    model = Model(Predict2Video2WorldModelConfig(pipe_config=pipe_cfg, model_manager_config=Predict2ModelManagerConfig(dit_path=dit_path)))
    dit = model.pipe.dit
    dit.requires_grad_(False)
    for emb in (dit.action_embedder_B_D, dit.action_embedder_B_3D):  # new first layer for 12 x 58 inputs; output starts at zero
        emb.fc1 = nn.Linear(ACTION_DIM, emb.fc1.out_features)
        nn.init.zeros_(emb.fc2.weight)
        nn.init.zeros_(emb.fc2.bias)
    if lora_rank:
        model.add_lora_to_model(dit, lora_rank=lora_rank, lora_alpha=lora_rank)
    for n, p in dit.named_parameters():
        if "action_embedder" in n or "lora_" in n:
            p.requires_grad_(True)
    dit.to("cuda")
    for p in dit.parameters():                                        # trained parameters in float32, frozen ones in bfloat16
        p.data = p.data.to(torch.float32 if p.requires_grad else torch.bfloat16)
    if weights:
        sd = torch.load(weights, map_location="cuda")
        missing = [k for k in sd if k not in dict(dit.named_parameters())]
        assert not missing, missing[:5]
        dit.load_state_dict(sd, strict=False)
    return model


def trainable_state(model) -> dict:
    return {n: p.detach().cpu() for n, p in model.pipe.dit.named_parameters() if p.requires_grad}


def batch_dict(video_u8: torch.Tensor, action: torch.Tensor) -> dict:
    """video_u8 [B, 3, T, H, W] uint8 (cuda), action [B, STEPS, 58] -> the data batch Cosmos expects."""
    from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig as TE  # noqa: N817

    B, _, T, H, Wd = video_u8.shape
    return {"video": video_u8, "action": action.cuda().to(torch.bfloat16),
            "t5_text_embeddings": torch.zeros(B, TE.NUM_TOKENS, TE.EMBED_DIM, dtype=torch.bfloat16, device="cuda"),
            "t5_text_mask": torch.ones(B, TE.NUM_TOKENS, dtype=torch.int64, device="cuda"),
            "fps": torch.full((B,), 4, device="cuda"), "image_size": 256 * torch.ones(B, 4, device="cuda"),
            "num_frames": torch.full((B,), T), "padding_mask": torch.zeros(B, 1, 256, 256, device="cuda")}
