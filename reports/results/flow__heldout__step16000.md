# flow (heldout__step16000)

Held-out: 115 episodes, 51258 frames (27778 contact, 23480 free). Seeds per variant: full 3, no_force 3, full_wrench_expert 3, regression_25k 3. Values are mean [95 % interval] from a bootstrap over held-out episodes and training seeds; Δ is the paired difference to `full` (negative = better than `full`).

## Action error, all joints (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 19.29 [18.31, 20.28] | 17.50 [15.75, 19.23] | 21.41 [20.42, 22.53] | – | – |
| no_force | 19.82 [18.62, 21.14] | 18.40 [16.26, 20.62] | 21.49 [20.49, 22.58] | +0.90 [+0.27, +1.68] | +0.08 [-0.36, +0.54] |
| full_wrench_expert | 18.35 [17.39, 19.30] | 16.94 [15.23, 18.62] | 20.03 [19.16, 21.03] | -0.56 [-0.93, -0.21] | -1.39 [-1.81, -1.08] |
| regression_25k | 14.73 [13.87, 15.55] | 13.51 [12.01, 14.93] | 16.18 [15.28, 17.14] | -3.99 [-4.54, -3.55] | -5.23 [-5.61, -4.91] |
| *baseline: hold current joint state* | 46.72 | 35.63 | 59.84 | – | – |

## Action error, arms (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 14.93 [13.98, 16.02] | 15.02 [13.43, 16.79] | 14.82 [14.31, 15.37] | – | – |
| no_force | 15.65 [14.26, 17.38] | 16.32 [14.01, 19.27] | 14.85 [14.29, 15.46] | +1.31 [+0.41, +2.56] | +0.03 [-0.29, +0.32] |
| full_wrench_expert | 14.46 [13.41, 15.73] | 14.73 [12.98, 16.84] | 14.13 [13.62, 14.67] | -0.29 [-0.63, +0.20] | -0.68 [-0.94, -0.45] |
| regression_25k | 11.24 [10.46, 12.13] | 11.30 [9.97, 12.76] | 11.17 [10.70, 11.68] | -3.72 [-4.28, -3.36] | -3.65 [-3.89, -3.44] |

## Action error, hands (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 20.68 [19.64, 21.71] | 18.29 [16.43, 20.04] | 23.51 [22.32, 24.91] | – | – |
| no_force | 21.14 [19.91, 22.37] | 19.06 [16.95, 21.15] | 23.61 [22.42, 24.92] | +0.77 [+0.18, +1.45] | +0.09 [-0.45, +0.65] |
| full_wrench_expert | 19.59 [18.60, 20.54] | 17.64 [15.88, 19.28] | 21.90 [20.86, 23.11] | -0.65 [-1.14, -0.23] | -1.61 [-2.16, -1.21] |
| regression_25k | 15.84 [14.91, 16.70] | 14.21 [12.61, 15.68] | 17.78 [16.70, 18.94] | -4.08 [-4.69, -3.58] | -5.73 [-6.19, -5.33] |

## Force error (N)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 0.54 [0.50, 0.59] | 0.65 [0.58, 0.71] | 0.42 [0.40, 0.43] | – | – |
| full_wrench_expert | 0.51 [0.47, 0.55] | 0.63 [0.56, 0.69] | 0.37 [0.36, 0.38] | -0.02 [-0.03, -0.01] | -0.05 [-0.06, -0.04] |
| regression_25k | 0.21 [0.17, 0.24] | 0.36 [0.31, 0.40] | 0.03 [0.02, 0.03] | -0.29 [-0.32, -0.27] | -0.39 [-0.41, -0.38] |
| *baseline: force stays as it is now* | 0.24 | 0.41 | 0.04 | – | – |

## Decision slots and expert gate (mean over seeds)

| variant | contact-soon accuracy | phase accuracy | force-level error (levels) | gate weight of expert 0: contact / free |
|---|---|---|---|---|
| full | 0.974 | 0.950 | 0.102 | 0.59 / 0.54 |
| no_force | 0.407 (untrained) | 0.949 | 1.470 (untrained) | 0.70 / 0.78 |
| full_wrench_expert | 0.975 | 0.952 | 0.103 | 0.00 / 0.00 |
| regression_25k | 0.974 | 0.949 | 0.103 | 0.41 / 0.47 |
