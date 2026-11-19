# headline (heldout)

Held-out: 115 episodes, 51258 frames (27778 contact, 23480 free). Seeds per variant: full 3, no_force 3, two_experts 3, no_force_two_experts 3. Values are mean [95 % interval] from a bootstrap over held-out episodes and training seeds; Δ is the paired difference to `full` (negative = better than `full`).

## Action error, all joints (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 13.95 [12.94, 14.99] | 13.52 [11.82, 15.22] | 14.46 [13.57, 15.46] | – | – |
| no_force | 14.91 [13.71, 16.16] | 14.57 [12.60, 16.67] | 15.32 [14.29, 16.41] | +1.05 [+0.66, +1.57] | +0.87 [+0.56, +1.18] |
| two_experts | 14.34 [13.37, 15.29] | 13.57 [11.93, 15.16] | 15.27 [14.33, 16.27] | +0.05 [-0.17, +0.23] | +0.81 [+0.56, +1.04] |
| no_force_two_experts | 14.97 [13.77, 16.19] | 14.65 [12.68, 16.70] | 15.35 [14.35, 16.41] | +1.13 [+0.75, +1.57] | +0.89 [+0.56, +1.20] |
| *baseline: hold current joint state* | 46.72 | 35.63 | 59.84 | – | – |

## Action error, arms (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 11.19 [10.19, 12.40] | 11.89 [10.22, 13.93] | 10.36 [9.91, 10.84] | – | – |
| no_force | 12.06 [10.85, 13.55] | 13.15 [11.11, 15.67] | 10.78 [10.29, 11.33] | +1.26 [+0.73, +1.87] | +0.43 [+0.24, +0.62] |
| two_experts | 11.22 [10.26, 12.39] | 11.75 [10.13, 13.68] | 10.60 [10.15, 11.09] | -0.14 [-0.41, +0.08] | +0.24 [+0.08, +0.40] |
| no_force_two_experts | 12.02 [10.84, 13.45] | 13.07 [11.09, 15.38] | 10.79 [10.29, 11.32] | +1.18 [+0.75, +1.67] | +0.43 [+0.25, +0.63] |

## Action error, hands (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 14.82 [13.76, 15.83] | 14.03 [12.27, 15.72] | 15.76 [14.66, 16.96] | – | – |
| no_force | 15.82 [14.57, 17.03] | 15.02 [13.04, 17.02] | 16.77 [15.54, 18.09] | +0.99 [+0.60, +1.51] | +1.01 [+0.64, +1.39] |
| two_experts | 15.34 [14.32, 16.29] | 14.14 [12.44, 15.76] | 16.75 [15.63, 17.96] | +0.11 [-0.15, +0.32] | +0.99 [+0.69, +1.27] |
| no_force_two_experts | 15.91 [14.63, 17.13] | 15.15 [13.15, 17.12] | 16.80 [15.60, 18.10] | +1.12 [+0.71, +1.57] | +1.04 [+0.63, +1.43] |

## Force error (N)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 0.21 [0.17, 0.24] | 0.36 [0.31, 0.41] | 0.03 [0.02, 0.03] | – | – |
| two_experts | 0.20 [0.17, 0.24] | 0.35 [0.30, 0.40] | 0.03 [0.02, 0.03] | -0.01 [-0.02, -0.00] | -0.00 [-0.00, -0.00] |
| *baseline: force stays as it is now* | 0.24 | 0.41 | 0.04 | – | – |

## Wrench expert: fingertip force error of its joint proposal (N)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 0.29 [0.24, 0.35] | 0.50 [0.42, 0.58] | 0.05 [0.04, 0.05] | – | – |

## Wrench expert: action error of its own proposal, before mixing (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 14.06 [13.04, 15.10] | 13.66 [11.93, 15.38] | 14.54 [13.65, 15.55] | – | – |

## Decision slots and expert gate (mean over seeds)

| variant | contact-soon accuracy | phase accuracy | force-level error (levels) | gate weight of expert 0: contact / free |
|---|---|---|---|---|
| full | 0.973 | 0.953 | 0.106 | 0.08 / 0.02 |
| no_force | 0.560 (untrained) | 0.949 | 1.432 (untrained) | 0.19 / 0.24 |
| two_experts | 0.973 | 0.951 | 0.103 | 0.40 / 0.47 |
| no_force_two_experts | 0.506 (untrained) | 0.951 | 1.424 (untrained) | 0.46 / 0.46 |
