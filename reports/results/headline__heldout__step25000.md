# headline (heldout__step25000)

Held-out: 115 episodes, 51258 frames (27778 contact, 23480 free). Seeds per variant: full 3, no_force 3, no_force_two_experts 3. Values are mean [95 % interval] from a bootstrap over held-out episodes and training seeds; Δ is the paired difference to `full` (negative = better than `full`).

## Action error, all joints (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 14.14 [13.16, 15.12] | 13.49 [11.80, 15.15] | 14.91 [14.05, 15.90] | – | – |
| no_force | 15.24 [14.12, 16.43] | 14.56 [12.63, 16.55] | 16.05 [15.06, 17.13] | +1.06 [+0.66, +1.54] | +1.14 [+0.83, +1.45] |
| no_force_two_experts | 15.23 [14.10, 16.37] | 14.59 [12.71, 16.51] | 15.99 [14.99, 17.06] | +1.09 [+0.75, +1.46] | +1.08 [+0.78, +1.42] |
| *baseline: hold current joint state* | 46.72 | 35.63 | 59.84 | – | – |

## Action error, arms (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 11.18 [10.25, 12.31] | 11.69 [10.11, 13.53] | 10.58 [10.15, 11.06] | – | – |
| no_force | 12.08 [10.93, 13.47] | 12.89 [11.00, 15.18] | 11.12 [10.63, 11.64] | +1.20 [+0.73, +1.77] | +0.54 [+0.35, +0.73] |
| no_force_two_experts | 12.02 [10.91, 13.44] | 12.82 [10.98, 15.16] | 11.07 [10.55, 11.64] | +1.13 [+0.68, +1.78] | +0.48 [+0.24, +0.74] |

## Action error, hands (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 15.08 [14.05, 16.08] | 14.07 [12.27, 15.73] | 16.29 [15.24, 17.50] | – | – |
| no_force | 16.25 [15.08, 17.38] | 15.09 [13.12, 17.03] | 17.62 [16.43, 18.93] | +1.02 [+0.63, +1.50] | +1.34 [+0.95, +1.72] |
| no_force_two_experts | 16.25 [15.08, 17.38] | 15.15 [13.22, 17.03] | 17.56 [16.35, 18.88] | +1.08 [+0.73, +1.46] | +1.28 [+0.90, +1.68] |

## Force error (N)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 0.21 [0.17, 0.24] | 0.36 [0.31, 0.41] | 0.03 [0.03, 0.03] | – | – |
| *baseline: force stays as it is now* | 0.24 | 0.41 | 0.04 | – | – |

## Wrench expert: fingertip force error of its joint proposal (N)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 0.31 [0.26, 0.37] | 0.53 [0.45, 0.61] | 0.05 [0.05, 0.06] | – | – |

## Wrench expert: action error of its own proposal, before mixing (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 14.29 [13.30, 15.29] | 13.66 [11.95, 15.35] | 15.03 [14.17, 16.03] | – | – |

## Decision slots and expert gate (mean over seeds)

| variant | contact-soon accuracy | phase accuracy | force-level error (levels) | gate weight of expert 0: contact / free |
|---|---|---|---|---|
| full | 0.973 | 0.950 | 0.106 | 0.08 / 0.02 |
| no_force | 0.557 (untrained) | 0.949 | 1.428 (untrained) | 0.19 / 0.24 |
| no_force_two_experts | 0.519 (untrained) | 0.950 | 1.424 (untrained) | 0.46 / 0.46 |
