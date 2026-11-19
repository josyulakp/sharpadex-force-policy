# Force-aware manipulation model on SharpaDex

A small chunked policy for two five-finger hands. It reads cameras, joint state and contact force, and predicts the next
joint targets together with the fingertip force it expects. Built to answer two questions on the public
[SharpaDex](https://huggingface.co/datasets/Sharpa-Robotics/SharpaDex-v1.0) dataset: does force pay off, and how much data
does such a model need.

**Results, the ablations and the path that led to the released model: [RESULTS.md](RESULTS.md).**
In one line: the four-expert model reaches 13.95 mrad joint error on the held-out episodes, the same model without force 14.91.

| Where | What |
|---|---|
| This repository | Code, configs, the fixed split, every run's config, training log and evaluation summary, result tables, figures, world-model rollout videos |
| [Hugging Face: josyula/sharpadex-force-policy](https://huggingface.co/josyula/sharpadex-force-policy) | Released checkpoints (all seeds), tactile and video world-model weights, per-frame evaluation files |
| Branch `gh-pages` | The HTML explainer pages |

Every parameter lives in `configs/default.yaml`; any of them can be overridden on the command line as
`section.key=value` without touching code. Experiment groups (named sets of overrides × seeds) are in `configs/experiments.yaml`.

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt && source .venv/bin/activate
hf download Sharpa-Robotics/SharpaDex-v1.0 --repo-type dataset --local-dir <data.root> \
   --include "clean_plate_with_eraser/*" --include "insert_plug_into_socket/*" --include "remove_block_from_box/*"
hf download josyula/sharpadex-force-policy --local-dir .      # released checkpoints, world-model weights, per-frame evaluation files
```

Then set three paths in `configs/default.yaml` (they default to a local scratch disk):
`data.root` (the dataset), `vision.cache_dir` (cached vision features), `tactile_world_model.data_dir` (tactile latents).

One-time preparation, in this order:

| Step | Command | Output |
|---|---|---|
| Data report and the fixed held-out split | `python scripts/data_report.py` | `reports/data/`, `splits/split_seed0.json` |
| Cache the frozen vision features | `python scripts/cache_features.py` | `<vision.cache_dir>` |
| Tactile latents for the four-expert model | see below | `<tactile_world_model.data_dir>/latents/` |

The four-expert model reads latents of the TacForeSight tactile world model. Its trained weights come with the
`hf download` above (`checkpoints/tacforesight/`):

```bash
git clone https://github.com/TacForeSight/TacForeSight <tactile_world_model.src>   # used unmodified
pip install --target <tactile_world_model.packages> zarr tensorboard    # kept out of the project environment
mkdir -p <tactile_world_model.data_dir>/runs && cp -r checkpoints/tacforesight/sharpa_eraser <tactile_world_model.data_dir>/runs/
python scripts/tf_features.py              # latents of every episode
python scripts/tf_features.py rest=true    # no-contact latents of the held-out episodes, for the zero-force check
```

Models without the tactile expert (every group except `headline/full`, `tactile_expert`, `scaling_four_experts`, `ht_loss`) do not need this step.

## Reproduce an evaluation from a released checkpoint

```bash
python scripts/evaluate.py runs/headline/full/seed0          # the released model  -> eval_heldout.json (13.87 mrad for seed 0)
python scripts/evaluate.py runs/headline/no_force/seed0      # the same model without force
python scripts/evaluate.py runs/headline/two_experts/seed0   # the second approach: two experts, with force
```

One evaluation takes about 10 seconds on a GPU. A checkpoint that is not on disk is fetched from the Hugging Face
repository (`release.hf_repo` in the config). Released checkpoints, three seeds each: `runs/headline/full`,
`runs/headline/no_force`, `runs/headline/two_experts`, `runs/headline/no_force_two_experts`; and the two 8k-step models
used in the first world-model rollouts (`runs/three_tasks/{full,no_force}/seed0`). Every other run keeps its config,
training log and evaluation summary in `runs/`, and is retrained with `run_experiments.py`.

| Evaluation | Command | Output |
|---|---|---|
| Held-out evaluation of one run | `python scripts/evaluate.py <run>` | `<run>/eval_heldout.json` |
| Force inputs set to zero | `python scripts/evaluate.py <run> zero_force=true` (also `=tactile`, `=torque`) | `eval_heldout__zero_force-True.json` |
| Instruction of another task | `python scripts/evaluate.py <run> swap_task=insert_plug_into_socket` | `eval_heldout__swap_task-….json` |
| With and without force, intervals over episodes and seeds | `python scripts/aggregate.py headline` | `reports/results/headline.md` |
| Data scaling, one task against three, input checks | `python scripts/report_steps.py` | `reports/results/steps_4_5.md`, `scaling.png` |
| Each way of using force (two experts) | `python scripts/aggregate.py three_tasks` | `reports/results/three_tasks.md` |
| Figures of RESULTS.md | `python scripts/results_figures.py` | `reports/figures/` |
| Numbers and figures of the report (top two approaches) | `python scripts/report_figures.py` | `reports/report/` |

`aggregate.py`, `report_steps.py` and `report_figures.py` read the per-frame files `eval_*.npz`. The ones behind the report
come with the `hf download` above; for any other group run `evaluate.py` on its runs first.

## Train

| What | Command | Output |
|---|---|---|
| One model | `python scripts/train.py train.name=my_run` | `runs/my_run/seed0/` |
| The released model, the two-expert model and their no-force counterparts, 3 seeds | `python scripts/run_experiments.py headline` | `runs/headline/<variant>/seed<k>/` |
| Any other group | `python scripts/run_experiments.py <group> [--jobs 3] [--only variant,variant]` | `runs/<group>/…` |

Groups in `configs/experiments.yaml`:

| Group | Question |
|---|---|
| `headline` | The four-expert and the two-expert model at 50k steps, each against the same model without force |
| `three_tasks` | Two-expert model at 8k steps: each way of using force alone; the wrench expert and its control |
| `eraser_force` | The same on one task, plus an MPC expert, a contact-gated expert and a force world-model head |
| `scaling`, `scaling_four_experts` | 10, 25 and 50 % of the training episodes, with and without force, for the two-expert and the four-expert model |
| `tactile_expert` | The fourth expert and its control without predicted latents |
| `ht_loss` | L1 against the heavy-tailed residual loss of HT-Policy (`python scripts/ht_report.py`) |
| `energy` | Energy-based refinement of the decoder latents (`python scripts/ebr_report.py`) |
| `flow`, `flow_scaled`, `flow_diagnose` | A flow-matching head in place of regression (`python scripts/report_flow.py`, `flow_diagnose.py`) |
| `refine`, `refine_luna` | Offline refinement of the flow policy with a critic (`rl_reward.py`, `rl_critic.py`, `rl_report.py`) |

## World models

Each step's script documents its exact command in its first lines.

| World model | Scripts, in order | Needs |
|---|---|---|
| Tactile world model (TacForeSight) | `tf_prepare.py`, `tf_train.py`, `tf_readout.py`, `tf_features.py` | A clone of TacForeSight (`tactile_world_model.src`) |
| Video world model (PrimeSim), policy in the loop | `ps_prepare.py`, `ps_train.py`, `ps_rollout.py`, `ps_long.py`, `ps_kinematics.py`, `ps_policy.py`, `ps_report.py` | A clone of PrimeSim with its own environment and base checkpoints (`primesim_world_model.src`) |
| Cosmos-Predict2 (earlier attempt) | `wm_prepare.py`, `wm_train.py`, `wm_rollout.py` | A clone of cosmos-predict2 (`video_world_model.src`) |
| Physics replay (MuJoCo) | `sim/run.py`, `scripts/sim_streams.py` | A MuJoCo environment and the robot model |

The fine-tuned PrimeSim weights and the kinematics networks are in the Hugging Face repository under `checkpoints/primesim/`
(copy them to `<primesim src>/models/<name>/` and next to the PrimeSim data folder). The rollout videos of the released
model, three seeds and three tasks each, are in `reports/world_model_primesim/policy_headline_full_*`;
`python scripts/ps_report.py` tabulates them into `reports/results/world_model_policy.md`.

## Other tools

| What | Command |
|---|---|
| What the model uses for which input, and the architecture page | `python scripts/activations.py <variant dir> && python scripts/build_architecture_page.py` |
| Replay single episodes offline (video and traces) | `python scripts/rollout.py && python scripts/build_rollout_page.py` |
| MPC expert against its C3 teacher | `python scripts/evaluate_mpc.py runs/eraser_force/full_plus_mpc_expert/seed0` |
| Force head edited by hand, without gradients | `python scripts/weight_edit.py` |
| Hosted vision-language judge of task progress | `python scripts/rl_judge_api.py dry_run=true` (key in `OPENAI_API_KEY` or the untracked file `.openai_key`) |

The HTML explainer pages (expert mixture, model anatomy, flow-matching diagnosis) are on the `gh-pages` branch. They were
written before the 50k-step runs and describe the two-expert model as the main one.

## Layout

```
configs/default.yaml       every data / model / training / evaluation parameter
configs/experiments.yaml   experiment groups = named sets of overrides x seeds
sharpa/data/episodes.py    episode index and numeric loading (LeRobot v2.1 export)
sharpa/data/split.py       10 % held-out per task (season-stratified, seeded) + nested training shares
sharpa/data/dataset.py     in-memory frame dataset: history / future windows, labels, normalisation stats
sharpa/model.py            the model and its loss
sharpa/engine.py           training loop, checkpoint I/O, per-frame held-out evaluation
sharpa/metrics.py          bootstrap intervals over episodes and seeds, paired differences
scripts/                   one script per step (see the tables above)
sim/                       MuJoCo replay of recorded command streams
runs/                      per run: config, training log, evaluation summary (checkpoints: Hugging Face)
reports/                   result tables (results/), figures, data report, world-model rollout videos
checkpoints/               (after hf download) tactile and video world-model weights
```

## Licence of the data

SharpaDex v1.0 is released by Sharpa Robotics under CC BY-NC 4.0. The checkpoints and the videos in `reports/` are derived
from it and are shared under the same terms, for non-commercial use.
