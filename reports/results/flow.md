# flow (heldout)

Held-out: 115 episodes, 51258 frames (27778 contact, 23480 free). Seeds per variant: full 3, no_force 3, full_wrench_expert 3, regression_25k 3. Values are mean [95 % interval] from a bootstrap over held-out episodes and training seeds; Δ is the paired difference to `full` (negative = better than `full`).

## Action error, all joints (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 18.28 [17.27, 19.28] | 16.86 [15.10, 18.61] | 19.96 [19.00, 21.03] | – | – |
| no_force | 18.70 [17.47, 19.95] | 17.66 [15.54, 19.87] | 19.92 [18.95, 20.94] | +0.80 [+0.32, +1.37] | -0.04 [-0.31, +0.20] |
| full_wrench_expert | 17.53 [16.52, 18.57] | 16.40 [14.66, 18.15] | 18.85 [17.90, 19.89] | -0.45 [-0.67, -0.25] | -1.10 [-1.34, -0.89] |
| regression_25k | 14.56 [13.66, 15.41] | 13.49 [11.93, 14.95] | 15.82 [14.91, 16.83] | -3.37 [-3.77, -3.03] | -4.13 [-4.47, -3.83] |
| *baseline: hold current joint state* | 46.72 | 35.63 | 59.84 | – | – |

## Action error, arms (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 14.24 [13.26, 15.37] | 14.51 [12.87, 16.41] | 13.91 [13.42, 14.43] | – | – |
| no_force | 15.05 [13.63, 16.83] | 15.97 [13.61, 18.98] | 13.96 [13.45, 14.54] | +1.46 [+0.63, +2.68] | +0.05 [-0.09, +0.18] |
| full_wrench_expert | 13.82 [12.74, 15.13] | 14.28 [12.49, 16.47] | 13.27 [12.79, 13.78] | -0.23 [-0.55, +0.17] | -0.64 [-0.75, -0.54] |
| regression_25k | 11.15 [10.33, 12.11] | 11.39 [9.99, 12.93] | 10.87 [10.41, 11.37] | -3.13 [-3.63, -2.77] | -3.04 [-3.20, -2.89] |

## Action error, hands (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 19.56 [18.49, 20.59] | 17.61 [15.77, 19.38] | 21.88 [20.69, 23.21] | – | – |
| no_force | 19.86 [18.64, 21.02] | 18.20 [16.15, 20.25] | 21.82 [20.66, 23.06] | +0.59 [+0.20, +1.05] | -0.06 [-0.44, +0.25] |
| full_wrench_expert | 18.71 [17.64, 19.70] | 17.08 [15.28, 18.79] | 20.63 [19.49, 21.88] | -0.52 [-0.83, -0.26] | -1.25 [-1.56, -0.98] |
| regression_25k | 15.64 [14.66, 16.52] | 14.16 [12.51, 15.65] | 17.40 [16.28, 18.64] | -3.45 [-3.90, -3.08] | -4.48 [-4.91, -4.10] |

## Force error (N)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 0.51 [0.47, 0.56] | 0.62 [0.56, 0.69] | 0.38 [0.37, 0.39] | – | – |
| full_wrench_expert | 0.48 [0.44, 0.52] | 0.60 [0.54, 0.67] | 0.33 [0.32, 0.34] | -0.02 [-0.03, -0.01] | -0.05 [-0.06, -0.04] |
| regression_25k | 0.20 [0.17, 0.24] | 0.36 [0.31, 0.40] | 0.03 [0.02, 0.03] | -0.27 [-0.29, -0.24] | -0.36 [-0.37, -0.34] |
| *baseline: force stays as it is now* | 0.24 | 0.41 | 0.04 | – | – |

## Decision slots and expert gate (mean over seeds)

| variant | contact-soon accuracy | phase accuracy | force-level error (levels) | gate weight of expert 0: contact / free |
|---|---|---|---|---|
| full | 0.974 | 0.950 | 0.103 | 0.59 / 0.54 |
| no_force | 0.392 (untrained) | 0.949 | 1.481 (untrained) | 0.68 / 0.77 |
| full_wrench_expert | 0.974 | 0.951 | 0.104 | 0.00 / 0.00 |
| regression_25k | 0.973 | 0.950 | 0.102 | 0.41 / 0.47 |
