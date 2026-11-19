# flow (heldout__step8000)

Held-out: 115 episodes, 51258 frames (27778 contact, 23480 free). Seeds per variant: full 3, no_force 3, full_wrench_expert 3, regression_25k 3. Values are mean [95 % interval] from a bootstrap over held-out episodes and training seeds; Δ is the paired difference to `full` (negative = better than `full`).

## Action error, all joints (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 22.31 [21.46, 23.16] | 19.78 [18.14, 21.37] | 25.30 [24.44, 26.24] | – | – |
| no_force | 22.38 [21.26, 23.58] | 20.28 [18.19, 22.40] | 24.87 [23.81, 25.90] | +0.50 [-0.21, +1.37] | -0.43 [-1.07, +0.21] |
| full_wrench_expert | 21.32 [20.38, 22.27] | 19.07 [17.41, 20.72] | 23.99 [22.98, 25.06] | -0.71 [-1.25, -0.23] | -1.31 [-1.89, -0.71] |
| regression_25k | 15.58 [14.81, 16.31] | 13.89 [12.41, 15.25] | 17.58 [16.81, 18.39] | -5.89 [-6.51, -5.43] | -7.72 [-8.18, -7.32] |
| *baseline: hold current joint state* | 46.72 | 35.63 | 59.84 | – | – |

## Action error, arms (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 17.01 [16.14, 18.02] | 16.66 [15.20, 18.23] | 17.42 [16.90, 17.99] | – | – |
| no_force | 17.56 [16.23, 19.21] | 17.78 [15.57, 20.50] | 17.31 [16.59, 18.01] | +1.12 [-0.10, +2.78] | -0.11 [-0.58, +0.27] |
| full_wrench_expert | 16.41 [15.42, 17.65] | 16.18 [14.58, 18.14] | 16.67 [16.01, 17.38] | -0.48 [-1.32, +0.43] | -0.75 [-1.12, -0.33] |
| regression_25k | 11.81 [11.06, 12.60] | 11.43 [10.20, 12.74] | 12.25 [11.75, 12.80] | -5.23 [-5.96, -4.59] | -5.17 [-5.40, -4.94] |

## Action error, hands (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 24.00 [23.12, 24.87] | 20.78 [19.05, 22.43] | 27.81 [26.80, 28.92] | – | – |
| no_force | 23.91 [22.79, 25.03] | 21.08 [18.96, 23.04] | 27.27 [26.02, 28.46] | +0.30 [-0.31, +1.02] | -0.54 [-1.27, +0.26] |
| full_wrench_expert | 22.89 [21.92, 23.85] | 19.99 [18.23, 21.59] | 26.31 [25.13, 27.58] | -0.79 [-1.25, -0.39] | -1.49 [-2.20, -0.78] |
| regression_25k | 16.78 [15.97, 17.54] | 14.67 [13.07, 16.09] | 19.27 [18.36, 20.23] | -6.10 [-6.71, -5.63] | -8.53 [-9.13, -8.03] |

## Force error (N)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 0.67 [0.63, 0.71] | 0.75 [0.69, 0.80] | 0.58 [0.57, 0.60] | – | – |
| full_wrench_expert | 0.62 [0.58, 0.66] | 0.71 [0.65, 0.77] | 0.52 [0.50, 0.54] | -0.04 [-0.05, -0.02] | -0.07 [-0.08, -0.05] |
| regression_25k | 0.21 [0.18, 0.24] | 0.36 [0.32, 0.40] | 0.03 [0.03, 0.03] | -0.39 [-0.40, -0.37] | -0.55 [-0.57, -0.54] |
| *baseline: force stays as it is now* | 0.24 | 0.41 | 0.04 | – | – |

## Decision slots and expert gate (mean over seeds)

| variant | contact-soon accuracy | phase accuracy | force-level error (levels) | gate weight of expert 0: contact / free |
|---|---|---|---|---|
| full | 0.976 | 0.949 | 0.105 | 0.61 / 0.55 |
| no_force | 0.405 (untrained) | 0.948 | 1.502 (untrained) | 0.71 / 0.77 |
| full_wrench_expert | 0.976 | 0.948 | 0.108 | 0.00 / 0.00 |
| regression_25k | 0.975 | 0.948 | 0.105 | 0.42 / 0.48 |
