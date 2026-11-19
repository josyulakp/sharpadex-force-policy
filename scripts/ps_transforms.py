"""PrimeSim pipeline step for SharpaDex: 75-D state and relative action (see scripts/ps_prepare.py for the layout)."""
import torch
from mmengine import TRANSFORMS
from primesim.data.pipeline.transforms import ToRelativeAction

SHARED = 31                                             # PrimeSim's shared layout; the finger joints follow
VALID = [*range(0, 16), *range(SHARED, 75)]             # hand poses + grippers + finger joints (torso/head/waist/base unused)


@TRANSFORMS.register_module()
class SharpaStateAction:
    """state_vec [T, 75], action_vec [T - 1, 75] (absolute) -> state, action (relative to the first frame), action_mask.
    Hand poses become relative poses exactly as for every other robot in PrimeSim; the mean-finger-angle "gripper"
    stays absolute like PrimeSim's grippers; finger joints are commanded angle - measured angle at the first frame."""

    def __init__(self, state_len, action_len):
        self.state_len, self.action_len = state_len, action_len
        self.rel = ToRelativeAction(key_mapper=[{'("action", "state")': "action"}])

    def __call__(self, data):
        state, action = data["state_vec"].float(), data["action_vec"].float()
        assert state.shape == (self.state_len, 75) and action.shape == (self.action_len, 75), (state.shape, action.shape)
        mask = torch.zeros(self.action_len, 75, dtype=torch.bool)
        mask[:, VALID] = True
        out = action.clone()
        out[:, :SHARED] = self.rel.relative_to_reference(action[:, :SHARED], state[0:1, :SHARED])
        out[:, SHARED:] = action[:, SHARED:] - state[0:1, SHARED:]
        data["state"], data["action"], data["action_mask"] = state, out * mask, mask
        return data
