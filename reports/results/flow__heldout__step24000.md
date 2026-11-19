# flow (heldout__step24000)

Held-out: 115 episodes, 51258 frames (27778 contact, 23480 free). Seeds per variant: full 3, no_force 3, full_wrench_expert 3, regression_25k 3. Values are mean [95 % interval] from a bootstrap over held-out episodes and training seeds; Δ is the paired difference to `full` (negative = better than `full`).

## Action error, all joints (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 18.28 [17.27, 19.28] | 16.86 [15.12, 18.60] | 19.96 [18.97, 21.05] | – | – |
| no_force | 18.71 [17.48, 19.96] | 17.66 [15.54, 19.87] | 19.95 [18.97, 21.02] | +0.81 [+0.34, +1.39] | -0.00 [-0.30, +0.25] |
| full_wrench_expert | 17.52 [16.51, 18.56] | 16.40 [14.66, 18.15] | 18.84 [17.90, 19.89] | -0.45 [-0.65, -0.27] | -1.11 [-1.37, -0.87] |
| regression_25k | 14.56 [13.67, 15.42] | 13.49 [11.94, 14.96] | 15.83 [14.91, 16.83] | -3.37 [-3.75, -3.04] | -4.13 [-4.48, -3.83] |
| *baseline: hold current joint state* | 46.72 | 35.63 | 59.84 | – | – |

## Action error, arms (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 14.23 [13.25, 15.36] | 14.50 [12.87, 16.40] | 13.91 [13.42, 14.43] | – | – |
| no_force | 15.05 [13.63, 16.81] | 15.98 [13.61, 18.99] | 13.96 [13.44, 14.54] | +1.47 [+0.64, +2.71] | +0.05 [-0.07, +0.17] |
| full_wrench_expert | 13.81 [12.73, 15.13] | 14.27 [12.50, 16.47] | 13.26 [12.79, 13.78] | -0.23 [-0.55, +0.18] | -0.64 [-0.75, -0.53] |
| regression_25k | 11.15 [10.33, 12.11] | 11.39 [10.00, 12.93] | 10.87 [10.41, 11.38] | -3.11 [-3.60, -2.76] | -3.03 [-3.19, -2.88] |

## Action error, hands (mrad)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 19.56 [18.49, 20.58] | 17.61 [15.78, 19.37] | 21.88 [20.67, 23.21] | – | – |
| no_force | 19.88 [18.64, 21.05] | 18.20 [16.15, 20.25] | 21.86 [20.69, 23.12] | +0.59 [+0.21, +1.04] | -0.02 [-0.43, +0.32] |
| full_wrench_expert | 18.70 [17.64, 19.70] | 17.08 [15.29, 18.80] | 20.62 [19.49, 21.88] | -0.53 [-0.81, -0.28] | -1.26 [-1.58, -0.96] |
| regression_25k | 15.65 [14.67, 16.52] | 14.16 [12.52, 15.66] | 17.40 [16.29, 18.63] | -3.45 [-3.88, -3.09] | -4.48 [-4.93, -4.09] |

## Force error (N)

| variant | all | contact | free | Δ contact | Δ free |
|---|---|---|---|---|---|
| full | 0.51 [0.47, 0.56] | 0.62 [0.56, 0.69] | 0.38 [0.37, 0.39] | – | – |
| full_wrench_expert | 0.48 [0.44, 0.52] | 0.60 [0.54, 0.67] | 0.33 [0.32, 0.34] | -0.02 [-0.03, -0.01] | -0.05 [-0.06, -0.04] |
| regression_25k | 0.20 [0.17, 0.24] | 0.36 [0.31, 0.40] | 0.03 [0.02, 0.03] | -0.27 [-0.29, -0.24] | -0.35 [-0.37, -0.34] |
| *baseline: force stays as it is now* | 0.24 | 0.41 | 0.04 | – | – |

## Decision slots and expert gate (mean over seeds)

| variant | contact-soon accuracy | phase accuracy | force-level error (levels) | gate weight of expert 0: contact / free |
|---|---|---|---|---|
| full | 0.974 | 0.950 | 0.103 | 0.59 / 0.54 |
| no_force | 0.391 (untrained) | 0.949 | 1.481 (untrained) | 0.68 / 0.77 |
| full_wrench_expert | 0.974 | 0.951 | 0.104 | 0.00 / 0.00 |
| regression_25k | 0.973 | 0.950 | 0.102 | 0.41 / 0.47 |
