# Step 1 — findings and cleaning decisions

Numbers: `data_report.md` / `summary.json` (regenerate with `python scripts/data_report.py`).

## What is in the data (3 tasks, LeRobot v2.1 export)
- 1,147 episodes, 4.78 h, 16 collection seasons; every episode is annotated as successful (score `S`).
- 30 Hz for all streams; timestamps deviate < 0.003 ms from 1/30 s; frame indices contiguous; no NaN/Inf.
- Per frame: 65-D joint state / action / joint torque (2 × 7-DoF arm, 2 × 22-DoF hand, 7 torso),
  60-D fingertip force/torque (10 tips × [Fx Fy Fz Tx Ty Tz], N and N·m, elastomer frame),
  12-D observed and commanded TCP pose, 4 RGB cameras 480×480, 2 tactile videos, subtask label.
- One-handed tasks: plug and block use only the right hand, and only thumb + index (pinch);
  the eraser task is bimanual (left holds plate, right wipes). Ring/little fingertips almost never load.

## Contact
- Fingertip force at rest is ~0.005 N — no sensor offset, no baseline correction needed.
- Contact is nearly binary: the contact fraction changes by only ~4–6 points between 1 N and 10 N,
  so the 1 N default is not a fragile choice (sweep in `contact_vs_threshold.png`).
- The hands are in contact for 35–66 % of frames because a grasped object stays loaded through
  move/place phases; "contact" here means "holding or pushing", not just the insertion moment.
  Free-motion frames are mostly reach/pre-grasp and post-release.
- Peak fingertip |F| in contact: median 17–20 N, p95 25–39 N; the plug insertion shows a
  distinct 15 → 40 N spike at the end of the `plug` phase.

## Action semantics
- `action` is the commanded joint target. It best matches the measured state 3–5 frames
  (100–170 ms) later (`action_state_lag.png`); hands lag more than arms.

## Cleaning / exclusions (applied in the loader, controlled from the config)
| Issue | Decision |
|---|---|
| `observation.state.tcp` force slots (dims 6–11, 18–23) always 0 (README confirms) | dropped; not a force source |
| joint torque torso dims 58–64 always 0 | dropped |
| torso `motor_j5` state stuck at 0 in eraser task while its action moves; torso std ≤ 0.06 rad | torso excluded from state and action (`data.use_torso: false`) → 58-D |
| 10–20 % of frames are outside any subtask segment (labelled with the global-task index) | mapped to subtask `none` |
| one plug season (2026-03-18 19:22) starts with ~12 slow warm-up episodes (28–44 s vs 13 s median) | kept: annotated successful, real operator variation; noted |
| duplicate `lerobot_v3.0` copy of each season | ignored |
| hand joint angles have no per-device zero-offset correction (README) | cannot fix; noted |

## Split
- Held-out: 10 % of episodes per task (34 / 42 / 39), **stratified by season**, fixed by
  `data.split_seed` (default 0). Stratifying keeps every collection session in the held-out set;
  a plain random draw left one eraser season with 0 held-out episodes.
- Caveat: held-out episodes share seasons (operator, day, scene) with training episodes, so the
  held-out error is in-distribution; the dataset README recommends season-level splits for
  generalisation claims. A season-held-out evaluation is a possible extra check.
- Training shares 10 / 25 / 50 / 100 % are nested subsets (seeded by `data.subset_seed`), so the
  data-scaling curve only adds episodes: ≈ 0.44 / 1.08 / 2.14 / 4.31 h over the three tasks.
- `splits/split_seed0.json` lists every held-out and training uid and each share's members.
