# Results

A small policy for two five-finger hands that reads cameras, joint state and contact force, and predicts the next joint
targets together with the fingertip force it expects. Trained and evaluated on three tasks of the public SharpaDex dataset
(`clean_plate_with_eraser`, `insert_plug_into_socket`, `remove_block_from_box`): 1,032 training episodes (4.31 h) and 115
held-out episodes that are never trained on.

**The released model is the four-expert model trained for 50,000 steps. On the held-out episodes its joint error is
13.95 mrad; the same model without force reaches 14.91 mrad.** Every other path tried in this project ended above it.
The second-best approach, the two-expert model at the same budget, reaches 14.34 mrad.

Checkpoints: [huggingface.co/josyula/sharpadex-force-policy](https://huggingface.co/josyula/sharpadex-force-policy).

All numbers below are mean absolute errors on the held-out episodes, mean of 3 training seeds. Intervals are 95 %
bootstrap intervals over held-out episodes and seeds. "In contact" means any fingertip force above 1 N.

Contents: [the result](#1-the-result) · [the model](#2-the-model) · [the paths taken](#3-the-paths-taken) ·
[ablation of the released model](#4-ablation-of-the-released-model) · [data need](#5-data-need) ·
[world models](#6-world-models) · [limits](#7-limits) · [reproduce](#8-reproduce)

## 1. The result

| Model, 50,000 steps | Joint error (mrad) | In contact | Free motion | Force forecast (N) |
|---|---|---|---|---|
| **Four experts, with force** (`headline/full`) | **13.95** [12.94, 14.99] | 13.52 | 14.46 | 0.21 [0.17, 0.24] |
| Same model without force, four plain experts (`headline/no_force`) | 14.91 [13.71, 16.16] | 14.57 | 15.32 | none |
| Two experts, with force (`headline/two_experts`) | 14.34 [13.37, 15.29] | 13.57 | 15.27 | 0.20 [0.17, 0.24] |
| Without force, two plain experts (`headline/no_force_two_experts`) | 14.97 [13.77, 16.18] | 14.65 | 15.35 | none |
| Reference: hold the current joint state / force stays as it is | 46.72 | 35.63 | 59.84 | 0.24 |

- **Force pays off by 0.97 mrad** (paired interval 0.68 to 1.31), about 7 %. In contact the gain is 1.05 mrad (0.66 to
  1.57), in free motion 0.87 (0.56 to 1.18). The arms in contact gain the most, 1.26 mrad.
- **It is not a capacity effect.** Without force, four plain experts are no better than two (0.05 mrad, interval −0.09 to
  +0.20). The extra experts only help when the model has force.
- **Four experts against two, both with force:** 0.40 mrad better (0.21 to 0.57). All of that lead is in free motion
  (0.81, interval 0.56 to 1.04); in contact the two are level (0.05, interval −0.17 to +0.23).
- **Force also pays off for the two-expert model:** 0.63 mrad (0.30 to 1.01), all of it in contact (1.09 in contact,
  0.08 in free motion).
- **By task** (with force against without): eraser 18.77 against 20.23, plug 11.42 against 12.35, block 10.65 against 11.00.
- **Seeds agree:** 13.87, 14.03, 13.94 with force; 14.88, 14.94, 14.92 without; 14.27, 14.33, 14.44 for two experts.

## 2. The model

```mermaid
flowchart LR
  cam["3 cameras<br/>frozen DINOv2 ViT-S/14"] --> enc
  st["joint state<br/>6-frame history"] --> enc
  tac["fingertip force, 10 x 6<br/>6-frame history"] --> enc
  tor["joint torque<br/>6-frame history"] --> enc
  task["task token"] --> enc
  enc["encoder<br/>4 layers, width 256"] --> dec["decoder<br/>16 action slots + 3 decision slots"]
  dec --> e1["plain expert 1"]
  dec --> e2["plain expert 2"]
  dec --> e3["wrench expert<br/>+ fingertip wrench history"]
  dec --> e4["tactile expert<br/>+ tactile world-model latents"]
  dec --> gate["gate<br/>softmax over experts"]
  e1 --> mix(("mix"))
  e2 --> mix
  e3 --> mix
  e4 --> mix
  gate --> mix
  mix --> act["16 joint targets<br/>58 joints, 0.53 s"]
  dec --> force["force forecast<br/>change from the force now"]
  dec --> decis["decisions<br/>contact soon, phase, force level"]
```

- **Size:** 7.57 M trained weights; the vision encoder is frozen and its features are cached.
- **Four experts:** every expert proposes the 16-step action chunk and a learned gate mixes the proposals. Two experts
  are plain MLPs. The wrench expert also reads ten frames of fingertip wrench history and is trained to predict the future
  wrench together with its action. The tactile expert also reads 16 latents of a frozen tactile world model: 8 of the last
  fingertip images and 8 predicted for the next 0.27 s.
- **Force in the output:** the force head predicts the change of the 30 fingertip force components over the chunk.
- **Training:** L1 loss on the actions (weight 1.0), force (0.5), decisions (0.2); AdamW, learning rate 3e-4 with warm-up and
  cosine decay, batch 256. 50,000 steps take about 70 minutes on one L40S GPU when it runs alone.
- **Code:** `sharpa/model.py` (model and loss), `sharpa/engine.py` (training and evaluation), `configs/default.yaml` (every parameter).

## 3. The paths taken

![Held-out joint error of every path, in the order it was tried](reports/figures/paths.png)

The project did not go straight to the four-expert model. The table lists each path in the order it was tried, what it
did to the held-out joint error, and whether it was kept. Unless a row says otherwise the numbers are for three tasks.

| # | Path | Result (mrad) | Effect | Kept |
|---|---|---|---|---|
| 1 | **Does force help on one task?** Two-expert model on the eraser task, 8k steps | 19.08 with force, 20.57 without | Force helps by 1.49; in contact by 2.03 [0.81, 3.36] | Yes |
| 2 | Which force signal matters (eraser) | no torque 20.41, no fingertip force 19.44 | Joint torque carries most of the action gain; fingertip force carries the force forecast (0.48 N against 0.74 N without it) | Both kept |
| 3 | Force head predicts the change from the force now (eraser) | force 0.37 N, was 0.48 N | The first force head was worse than "force stays as it is" (0.40 N); this one beats it | Yes |
| 4 | Imitate a contact MPC as an extra expert; contact-gated expert; small force world model inside the policy (eraser) | 19.14, 19.17, 19.16 against 19.08 | No measurable change in joint error | No |
| 5 | **Three tasks**, two experts, 8k steps | 15.87 with force, 16.23 without | Force helps by 0.36; in contact by 0.66 [0.31, 1.09] | Yes |
| 6 | Flow-matching head in place of regression, 25k steps | 18.28 against 14.56 for regression at 25k | 3.7 worse. Diagnosis: the random starting noise accounts for 2.4 of it (15.86 with zero noise). Twice the width also helps (15.97 at 27 M weights), but regression at that width is at 14.07 | No |
| 7 | Offline refinement of the flow policy with a learned critic (stage 2 of Facet-0), 8k more steps | 17.19 against 17.29 for plain continued training | −0.10 [−0.21, +0.02]. Open vision-language models could not label task progress (agreement below a constant guess); labels from a hosted model gave −0.06 | No |
| 8 | Energy-based refinement of the decoder latents, 8k steps | 15.73 against 15.87 | −0.14, and a control that drops the energy and outputs the update directly does the same (15.68). Starting from noise: 30.91 | No |
| 9 | **Third expert: wrench expert**, 8k steps | 15.48 | −0.39 [−0.48, −0.30] against two experts | Yes |
| 10 | **Fourth expert: tactile world-model expert**, 8k steps | 15.20 | −0.28 [−0.36, −0.19] against three experts; −0.67 [−0.78, −0.53] against two | Yes |
| 11 | Longer training, 25k steps | four experts 14.16, two experts 14.56 | The experts' lead survives: −0.40 [−0.56, −0.21] | Yes |
| 12 | Heavy-tailed residual loss (HT-Policy) on four experts | 15.57 at 25k, 15.07 at 50k | Worse than L1 by 1.41 and 1.13. Root-mean-square error is tied. Its predicted scale is a good uncertainty estimate (rank correlation 0.85 with the actual error) | No |
| 13 | **Four experts, 50k steps** | **13.95** | −0.21 [−0.34, −0.07] against 25k, all of it in free motion | **Released** |
| 14 | Same model without force, 50k steps | 14.91 | +0.97 [0.68, 1.31] | Counterpart |
| 15 | Two experts with force, 50k steps | 14.34 | +0.40 [0.21, 0.57] against four experts, all of it in free motion | Second approach |
| 16 | Data scaling repeated for four experts, 8k steps | 19.50, 17.20, 16.12, 15.20 at 10, 25, 50, 100 % | Four experts lead two at every share (19.84, 17.66, 16.70, 15.87) | Yes |

What the paths show:

- **The gains that held all came from one line:** force inputs, then one expert per force signal, then more training steps.
  Each step on that line was measured with three seeds and survived the next change of budget.
- **The alternatives to that line all lost.** A different head (flow matching), a different loss (heavy-tailed), and two
  refinement schemes (critic-guided, energy-based) each ended between 0.9 and 4.3 mrad above the released model. Their
  write-ups are in `reports/results/` (`flow_steps.md`, `refine.md`, `energy.md`, `ht_loss.md`).
- **The force gain grew with the model and the budget.** It was 0.36 mrad for two experts at 8k steps, is 0.63 for two
  experts at 50k steps and 0.97 for four experts at 50k steps.

![Held-out joint error against training steps](reports/figures/budget.png)

Three things the four-expert result does not show, stated so the claim matches the evidence:

- **The experts' gain is not tied to their special inputs.** At 8k steps the wrench expert without its wrench tokens scores
  15.51 (15.48 with them), and the tactile expert without the predicted latents scores 15.23 (15.20 with them). Both
  differences are inside the noise. The gain comes from an extra expert with its own force-related loss in a model that
  has force, and we cannot attribute it to the wrench history or to the world model's predictions.
- **Width is another lever, not combined here.** A two-expert network of twice the width (26.4 M weights) reaches 14.07 at
  25k steps, level with four experts at 7.57 M weights (14.16). The two were never combined.
- **Longer training is running out.** From 25k to 50k steps the error in contact did not improve (13.46 to 13.52).
- **In contact, four experts are no better than two.** At 50k steps the two models are level in contact (13.52 against
  13.57). The four-expert lead is a free-motion gain.

## 4. Ablation of the released model

### With and without force, 50k steps

See [section 1](#1-the-result); the full table with arms, hands and the gate is `reports/results/headline.md`.

### Does it use its inputs? (no retraining)

| Input changed | Joint error (mrad) | Change | In contact | Free motion |
|---|---|---|---|---|
| Normal | 13.95 | | | |
| All force inputs set to zero | 28.95 | +15.01 [+12.26, +18.76] | +11.87 | +18.72 |
| Fingertip force only set to zero | 15.91 | +1.97 [+1.54, +2.49] | +3.62 | +0.01 |
| Joint torque only set to zero | 25.45 | +11.51 [+9.20, +14.45] | +5.51 | +18.61 |
| Instruction of another task | 14.0 | about +0.1 | | |

- **Force inputs are used heavily,** joint torque most of all. Zeroed fingertip force hurts only in contact, as it should.
- **The instruction is not used.** Swapping it changes the error by about 0.1 mrad on every task: the model tells the three
  tasks apart from the images. With three visually distinct tasks, the instruction input is untested.
- **Force forecast with fingertip force zeroed:** 1.14 N against 0.21 N, so the forecast depends on the fingertip sensors.
- **The two-expert model at 50k steps behaves the same way:** +13.4 mrad with all force inputs zeroed, +9.6 for joint torque
  alone, +1.0 for fingertip force alone, +0.1 for a swapped instruction.
- **A caution on torque.** Zeroed torque costs 18.6 mrad in free motion, where nothing is touched. Torque there reflects
  the controller following the operator, so part of its value may be the operator's next command leaking through. A
  two-expert model retrained without torque loses only 0.73 mrad in contact (last table of this section).

Per task: `reports/results/steps_4_5.md`.

### One expert at a time, 8k steps

| Experts | Joint error (mrad) | In contact | Free motion | Weights |
|---|---|---|---|---|
| Two plain | 15.87 | 14.01 | 18.07 | 6.73 M |
| + wrench expert | 15.48 | 13.81 | 17.46 | 7.19 M |
| + tactile world-model expert | 15.20 | 13.58 | 17.11 | 7.57 M |

With four experts the gate gives the wrench expert 53 % of the weight and the tactile expert 40 %; the two plain experts
share the rest.

### Each way of using force, two experts, 8k steps

| Variant | Joint error (mrad) | Change in contact | Force forecast (N) |
|---|---|---|---|
| Full | 15.87 | | 0.21 |
| No force at all | 16.23 | +0.66 [+0.31, +1.09] | none |
| No joint torque input | 16.39 | +0.73 [+0.43, +1.07] | 0.22 |
| No fingertip force input | 15.97 | +0.10 [−0.04, +0.24] | 0.38 |
| No force output | 15.75 | −0.04 [−0.18, +0.10] | none |
| No force-based decision labels | 15.71 | −0.05 [−0.19, +0.12] | 0.21 |

Joint torque is the input that improves the actions. Fingertip force is what makes the force forecast work. Predicting
force and the decision labels do not improve the actions. This ladder was measured on the two-expert model only
(`reports/results/three_tasks.md`); at 50k steps only the full model and the no-force model exist.

## 5. Data need

![With and without force, data need, input checks](reports/report/results.png)

Nested subsets of the training episodes, 8k steps for every point, both approaches. "Force helps by" is the paired
difference to the same model without force (four plain experts, or two plain experts).

| Share of training episodes | Hours | Four experts (mrad) | Force helps by | Two experts (mrad) | Force helps by |
|---|---|---|---|---|---|
| 10 % | 0.44 | 19.50 | 1.35 [0.85, 1.93] | 19.84 | 1.05 [0.53, 1.76] |
| 25 % | 1.08 | 17.20 | 0.96 [0.63, 1.36] | 17.66 | 0.72 [0.32, 1.25] |
| 50 % | 2.14 | 16.12 | 0.75 [0.47, 1.07] | 16.70 | 0.36 [0.04, 0.75] |
| 100 % | 4.31 | 15.20 | 0.76 [0.55, 1.00] | 15.87 | 0.36 [0.09, 0.68] |

- **Each doubling of the data lowers the error by 7.3 % for four experts and 6.5 % for two,** with no sign of flattening
  at 4.3 hours.
- **Four experts lead two at every data share,** by 0.3 to 0.7 mrad.
- **Force is worth the most when data is scarce:** 1.35 mrad at 26 minutes of data against 0.76 at 4.3 hours for four
  experts, and 1.05 against 0.36 for two.
- **Training budget matters as much as data.** At the full data, 50k steps instead of 8k lowers the four-expert error from
  15.20 to 13.95, as much as 1.1 doublings of the data on this curve.
- **One task against three, on the eraser episodes** (two experts): the one-task model scores 19.22 and the three-task
  model 19.83 at the same 8k steps, so adding tasks at a fixed budget does not help. With the longer budget the
  three-task released model scores 18.77 on the same episodes.

Tables with intervals: `reports/results/scaling_four_experts.md`, `reports/results/steps_4_5.md`.

## 6. World models

Three world models were used. None of them replaces the offline evaluation above; they answer what the offline numbers
cannot: what happens when the policy's own commands decide the next observation.

### 6.1 Video world model (PrimeSim), policy in the loop

PrimeSim (Wan 2.2, 5 B weights, three camera views) was fine-tuned for 6,000 steps on SharpaDex with the joint commands as
input. First check, on held-out eraser windows of 3.2 s: does the video follow the commands?

| Commands given to the video model | Pixel error, head camera (grey levels, 0–255) | Moving pixels only |
|---|---|---|
| Recorded commands | 11.4 | 32.8 |
| Commands of another window | 24.2 | 66.2 |
| Zero commands | 26.5 | 106.3 |
| Reference: repeat the first frame | 26.4 | 106.5 |

The video follows the commands. Over whole episodes the error grows when the model feeds on its own frames (11.2, 15.6,
17.2 grey levels over three 3.2 s blocks) and stays flat when it is re-anchored to a real frame (11.2, 10.0, 9.8).

Then the released four-expert model acts inside the video model: it sees the generated camera frames, its commands drive
the next 1.5 s of video, and so on for up to 16 s. All three training seeds, four held-out episodes per task and seed. The
table gives the distance between the policy's commands and the operator's at the same time, which measures divergence and
not error. Each cell is the mean over episodes and seeds, with the three seeds in brackets.

| Task | | 0–3.2 s | 3.2–6.4 s | 6.4–9.6 s |
|---|---|---|---|---|
| clean_plate_with_eraser | all joints (mrad) | 74 (81, 77, 64) | 106 (124, 98, 95) | 113 (123, 109, 106) |
| | arms (mrad) | 36 (40, 39, 29) | 67 (79, 63, 59) | 84 (96, 78, 78) |
| | force forecast gap (N) | 0.21 | 0.47 | 0.59 |
| insert_plug_into_socket | all joints (mrad) | 34 (35, 34, 32) | 69 (70, 70, 68) | |
| | arms (mrad) | 18 (18, 18, 19) | 36 (36, 34, 36) | |
| | force forecast gap (N) | 0.18 | 0.17 | |
| remove_block_from_box | all joints (mrad) | 27 (27, 27, 27) | 60 (62, 58, 60) | |
| | arms (mrad) | 26 (27, 25, 27) | 33 (38, 29, 33) | |
| | force forecast gap (N) | 0.09 | 0.13 | |

- **The policy drifts from the operator within seconds.** The distance roughly doubles from the first 3.2 s to the next on
  every task. On recorded frames the same model is 14 mrad from the operator; inside the video model it is 27 to 74 mrad
  after 3 s and 60 to 106 after 6 s.
- **Seeds agree on the plug and block tasks** (within 4 mrad) and differ on the eraser task (95 to 124 mrad in the second block).
- **Against the earlier two-expert rollouts** (8k steps, one seed): eraser 88, 133, 133 mrad; plug 36, 66; block 28, 65.
  The released model is closer to the operator on the eraser task and level on the other two. One seed against three,
  so this is an indication only.

![Eraser task: recording above, policy in the video world model below](reports/figures/primesim_clean_plate_with_eraser.png)
![Plug task](reports/figures/primesim_insert_plug_into_socket.png)
![Block task](reports/figures/primesim_remove_block_from_box.png)

Videos (recording on the left, policy in the video model on the right), four episodes per folder:

| Task | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| clean_plate_with_eraser | [episodes 0–3](reports/world_model_primesim/policy_headline_full_seed0_sharpa_three_tasks_clean_plate_with_eraser/) · [first](reports/world_model_primesim/policy_headline_full_seed0_sharpa_three_tasks_clean_plate_with_eraser/episode_0.mp4) | [episodes 0–3](reports/world_model_primesim/policy_headline_full_seed1_sharpa_three_tasks_clean_plate_with_eraser/) · [first](reports/world_model_primesim/policy_headline_full_seed1_sharpa_three_tasks_clean_plate_with_eraser/episode_0.mp4) | [episodes 0–3](reports/world_model_primesim/policy_headline_full_seed2_sharpa_three_tasks_clean_plate_with_eraser/) · [first](reports/world_model_primesim/policy_headline_full_seed2_sharpa_three_tasks_clean_plate_with_eraser/episode_0.mp4) |
| insert_plug_into_socket | [episodes 0–3](reports/world_model_primesim/policy_headline_full_seed0_sharpa_three_tasks_insert_plug_into_socket/) · [first](reports/world_model_primesim/policy_headline_full_seed0_sharpa_three_tasks_insert_plug_into_socket/episode_0.mp4) | [episodes 0–3](reports/world_model_primesim/policy_headline_full_seed1_sharpa_three_tasks_insert_plug_into_socket/) · [first](reports/world_model_primesim/policy_headline_full_seed1_sharpa_three_tasks_insert_plug_into_socket/episode_0.mp4) | [episodes 0–3](reports/world_model_primesim/policy_headline_full_seed2_sharpa_three_tasks_insert_plug_into_socket/) · [first](reports/world_model_primesim/policy_headline_full_seed2_sharpa_three_tasks_insert_plug_into_socket/episode_0.mp4) |
| remove_block_from_box | [episodes 0–3](reports/world_model_primesim/policy_headline_full_seed0_sharpa_three_tasks_remove_block_from_box/) · [first](reports/world_model_primesim/policy_headline_full_seed0_sharpa_three_tasks_remove_block_from_box/episode_0.mp4) | [episodes 0–3](reports/world_model_primesim/policy_headline_full_seed1_sharpa_three_tasks_remove_block_from_box/) · [first](reports/world_model_primesim/policy_headline_full_seed1_sharpa_three_tasks_remove_block_from_box/episode_0.mp4) | [episodes 0–3](reports/world_model_primesim/policy_headline_full_seed2_sharpa_three_tasks_remove_block_from_box/) · [first](reports/world_model_primesim/policy_headline_full_seed2_sharpa_three_tasks_remove_block_from_box/episode_0.mp4) |

Per-seed numbers: `reports/results/world_model_policy.md` (`python scripts/ps_report.py`).

With and without force, an earlier comparison (two-expert models at 8k steps, eraser task, world model trained on that task
only, one seed):

| Policy | 0–3.2 s (all / arms) | 3.2–6.4 s | 6.4–9.6 s |
|---|---|---|---|
| With force | 76 / 36 | 117 / 70 | 120 / 88 |
| Without force | 82 / 45 | 146 / 113 | 130 / 104 |

The policy with force stays closer to the operator, most clearly in the arms. This comparison was not repeated for the
released model.

What this evaluation is not:

- **Forces are not simulated.** Fingertip force, joint torque, the wrench history and the tactile latents are replayed from
  the recording, so they match the imagined scene only while the rollout stays close to the recording. After a few
  seconds the policy acts on force readings of a scene it is no longer in.
- **No success measure.** The table measures distance to the operator, and four episodes per task and seed give no intervals.

### 6.2 Tactile world model (TacForeSight), inside the released model

TacForeSight was trained on the fingertip deformation images of the eraser task to predict the next 8 frames (0.27 s) of
tactile latents from the last 8, the joints and the commands. The tactile expert reads these latents.

| Fingertip force error (N) | 0.27 s ahead | 1.33 s | 2.67 s |
|---|---|---|---|
| Read from the predicted latent | 1.00 | 1.23 | 1.27 |
| Force stays as it is | 0.29 | 1.23 | 1.94 |
| Read from the real latent (read-out limit) | 0.77 | 0.78 | 0.79 |

- **As a force predictor it is weak at short range:** it loses to "force stays as it is" up to 1.3 s and wins beyond.
  The limit is the read-out, which cannot recover force better than 0.77 N even from real latents.
- **It does respond to the commands:** with another window's commands the error rises from 1.00 to 1.74 N.
- **As an input to the policy it is worth 0.28 mrad** (path 10), though the predicted half of its latents adds nothing
  measurable over the measured half.

### 6.3 Earlier attempts

- **Cosmos-Predict2 (2 B weights), head camera only:** after 2,000 fine-tuning steps its replay of one episode was at
  31.7 grey levels, against 11.4 for PrimeSim on its test windows. The two numbers are from different episodes, so they
  are an indication and not a comparison. Dropped for PrimeSim.
- **Physics simulation (MuJoCo) replay of recorded commands:** built for the eraser task (`sim/`), used for replay only.
  No policy was evaluated in it.

## 7. Limits

- **Offline error is not task success.** Every headline number is the distance to the operator's command on recorded
  frames. No policy was run on a robot or scored for success.
- **Three tasks, one robot, 4.3 hours.** The instruction input is unused at this scale (section 4).
- **The ablation of each way of using force is from the two-expert model at 8k steps,** and the scaling curves are at
  8k steps. At 50k steps there are only the with-force and without-force runs of both approaches.
- **Part of the gain from torque may be the operator's command leaking through the controller** (section 4). Only a
  closed-loop test on a robot can settle it.
- **The tactile world model was trained on one task** and its latents are used on all three.
- **The video-model rollouts replay the forces of the recording** and have no success measure (section 6.1).

## 8. Reproduce

Setup, data download and every command are in [README.md](README.md). The numbers of this page:

| What | Command | Output |
|---|---|---|
| Section 1 | `python scripts/aggregate.py headline` | `reports/results/headline.md` |
| Scaling of the four-expert model | `python scripts/aggregate.py scaling_four_experts` | `reports/results/scaling_four_experts.md` |
| One released checkpoint | `python scripts/evaluate.py runs/headline/full/seed0` | `runs/headline/full/seed0/eval_heldout.json` |
| Sections 4 and 5 | `python scripts/report_steps.py`, `python scripts/aggregate.py three_tasks` | `reports/results/steps_4_5.md`, `three_tasks.md`, `scaling.png` |
| Paths 6 to 8 and 12 | `python scripts/report_flow.py`, `python scripts/rl_report.py`, `python scripts/ebr_report.py`, `python scripts/ht_report.py` | `reports/results/` |
| Section 6.1 | `python scripts/ps_report.py` | `reports/results/world_model_policy.md` |
| Figures of this page | `python scripts/results_figures.py` | `reports/figures/` |
| The figure of section 5 and its numbers | `python scripts/report_figures.py` | `reports/report/` |
