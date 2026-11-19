# flow (heldout__flow_noise_scale-0)

Held-out: 115 episodes, 51258 frames (27778 contact, 23480 free). Seeds per variant: full 3, no_force 3, full_wrench_expert 3, regression_25k 3. Values are mean [95 % interval] from a bootstrap over held-out episodes and training seeds; Δ is the paired difference to `full` (negative = better than `full`).

## Action error, all joints (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 15.86 [14.88, 16.81] | 14.71 [13.03, 16.44] | 17.21 [16.24, 18.30] | – | – |
| no_force | 16.08 [14.93, 17.28] | 15.35 [13.34, 17.47] | 16.95 [16.00, 18.03] | +0.64 [+0.16, +1.20] | -0.26 [-0.59, +0.04] |
| full_wrench_expert | 15.09 [14.10, 16.10] | 14.40 [12.69, 16.10] | 15.92 [15.02, 16.92] | -0.31 [-0.59, -0.05] | -1.30 [-1.60, -1.05] |
| regression_25k | 14.56 [13.66, 15.41] | 13.49 [11.93, 14.95] | 15.82 [14.91, 16.83] | -1.22 [-1.63, -0.87] | -1.39 [-1.73, -1.09] |
| *baseline: hold current joint state* | 46.72 | 35.63 | 59.84 | – | – |

## Action error, arms (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 12.01 [11.03, 13.17] | 12.38 [10.69, 14.29] | 11.59 [11.09, 12.14] | – | – |
| no_force | 12.77 [11.35, 14.58] | 13.80 [11.43, 16.87] | 11.54 [11.00, 12.13] | +1.43 [+0.55, +2.68] | -0.05 [-0.22, +0.12] |
| full_wrench_expert | 11.89 [10.80, 13.21] | 12.42 [10.63, 14.63] | 11.26 [10.77, 11.81] | +0.05 [-0.28, +0.43] | -0.32 [-0.48, -0.16] |
| regression_25k | 11.15 [10.33, 12.11] | 11.39 [9.99, 12.93] | 10.87 [10.41, 11.37] | -0.99 [-1.55, -0.60] | -0.72 [-0.90, -0.55] |

## Action error, hands (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 17.08 [16.07, 18.05] | 15.45 [13.67, 17.21] | 19.00 [17.82, 20.33] | – | – |
| no_force | 17.14 [16.00, 18.23] | 15.84 [13.90, 17.76] | 18.68 [17.52, 19.97] | +0.39 [-0.02, +0.84] | -0.32 [-0.76, +0.06] |
| full_wrench_expert | 16.11 [15.07, 17.06] | 15.03 [13.31, 16.69] | 17.40 [16.31, 18.61] | -0.43 [-0.79, -0.11] | -1.60 [-1.99, -1.28] |
| regression_25k | 15.64 [14.66, 16.52] | 14.16 [12.51, 15.65] | 17.40 [16.28, 18.64] | -1.30 [-1.74, -0.91] | -1.60 [-2.03, -1.23] |

## Force error (N)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 0.29 [0.25, 0.32] | 0.43 [0.38, 0.48] | 0.12 [0.11, 0.12] | – | – |
| full_wrench_expert | 0.29 [0.26, 0.33] | 0.43 [0.38, 0.49] | 0.12 [0.12, 0.13] | +0.00 [-0.00, +0.01] | +0.00 [-0.00, +0.01] |
| regression_25k | 0.20 [0.17, 0.24] | 0.36 [0.31, 0.40] | 0.03 [0.02, 0.03] | -0.07 [-0.08, -0.07] | -0.09 [-0.10, -0.09] |
| *baseline: force stays as it is now* | 0.24 | 0.41 | 0.04 | – | – |

## Decision slots and expert gate (mean over seeds)

| variant | contact-soon accuracy | phase accuracy | force-level error (levels) | gate weight of expert 0: contact / free |
|---|---|---|---|---|
| full | 0.974 | 0.950 | 0.103 | 0.59 / 0.54 |
| no_force | 0.392 (untrained) | 0.949 | 1.481 (untrained) | 0.68 / 0.77 |
| full_wrench_expert | 0.974 | 0.951 | 0.104 | 0.00 / 0.00 |
| regression_25k | 0.973 | 0.950 | 0.102 | 0.41 / 0.47 |
