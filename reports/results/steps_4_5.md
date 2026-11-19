# Steps 4 and 5: data scaling, one task against three, input checks

## Data scaling (three tasks, all held-out episodes)

Two-expert model (`three_tasks/full` and `no_force`). Nested subsets of the training episodes, the same held-out episodes and 8000 steps for every point. Δ is no-force minus with-force at the same share (positive = force helps).

| share | episodes | hours | with force, all (mrad) | without force, all | Δ all | with force, contact | without force, contact | Δ contact | force error (N) |
|---|---|---|---|---|---|---|---|---|---|
| 10 % | 104 | 0.44 | 19.84 [18.60, 21.13] | 20.89 [19.35, 22.55] | +1.05 [+0.53, +1.76] | 18.37 [16.16, 20.55] | 19.69 [16.97, 22.65] | +1.32 [+0.66, +2.42] | 0.23 [0.20, 0.27] |
| 25 % | 260 | 1.08 | 17.66 [16.62, 18.68] | 18.37 [17.16, 19.62] | +0.72 [+0.32, +1.25] | 16.28 [14.45, 18.06] | 17.15 [14.93, 19.36] | +0.87 [+0.35, +1.63] | 0.22 [0.19, 0.26] |
| 50 % | 517 | 2.14 | 16.70 [15.83, 17.61] | 17.06 [15.99, 18.14] | +0.36 [+0.04, +0.75] | 15.06 [13.43, 16.63] | 15.84 [13.91, 17.79] | +0.78 [+0.34, +1.40] | 0.22 [0.19, 0.25] |
| 100 % | 1032 | 4.31 | 15.87 [15.11, 16.60] | 16.23 [15.32, 17.13] | +0.36 [+0.09, +0.68] | 14.01 [12.61, 15.32] | 14.67 [13.06, 16.27] | +0.66 [+0.31, +1.09] | 0.21 [0.18, 0.25] |

Log-log slope: with force -0.097 (-6.5 % error per doubling of data), without force -0.112 (-7.4 %).

## One task against three, on the 34 eraser held-out episodes

One-task models saw 1.59 h, three-task models 4.31 h; two experts and 8000 steps each unless noted. Δ is against the one-task model with force (negative = better).

| model | all (mrad) | contact | free | Δ contact | Δ free | force error (N) |
|---|---|---|---|---|---|---|
| one_task | 19.22 [18.32, 20.14] | 20.22 [19.18, 21.23] | 17.23 [16.04, 18.52] | – | – | 0.37 [0.33, 0.42] |
| three_tasks | 19.83 [19.15, 20.49] | 20.13 [19.35, 20.84] | 19.22 [18.01, 20.60] | -0.09 [-0.68, +0.47] | +2.00 [+1.53, +2.44] | 0.38 [0.34, 0.42] |
| one_task_no_force | 20.57 [18.99, 22.25] | 22.10 [20.16, 24.14] | 17.52 [16.18, 18.93] | +1.88 [+0.73, +3.13] | +0.29 [-0.11, +0.76] | – |
| three_tasks_no_force | 20.51 [19.54, 21.53] | 21.27 [20.10, 22.46] | 18.99 [17.72, 20.45] | +1.06 [+0.54, +1.61] | +1.76 [+1.37, +2.23] | – |
| three_tasks_3x_steps | 18.92 [18.13, 19.74] | 19.90 [18.92, 20.89] | 16.97 [15.85, 18.21] | -0.32 [-0.77, +0.12] | -0.25 [-0.58, +0.16] | 0.37 [0.33, 0.41] |
| three_tasks_four_experts_50k | 18.77 [17.68, 19.96] | 20.33 [19.01, 21.69] | 15.66 [14.62, 16.76] | +0.11 [-0.41, +0.75] | -1.56 [-2.00, -1.10] | 0.38 [0.33, 0.43] |
| three_tasks_four_experts_50k_no_force | 20.23 [18.68, 21.98] | 21.84 [19.96, 23.99] | 17.00 [15.71, 18.36] | +1.62 [+0.54, +3.03] | -0.23 [-0.73, +0.28] | – |

## Input checks (released model `headline/full`: four experts, 50k steps; no retraining)

zero_force: fingertip force, joint torque and wrench history set to zero, and the tactile expert given the latents of no-contact fingertip images. zero_tactile_only keeps the torque, zero_torque_only keeps everything from the fingertips.

Rows are held-out episodes of one task; Δ is the paired change against the normal evaluation of the same model.

| held-out task | input | action, all (mrad) | Δ all | Δ contact | Δ free | force error (N) | Δ force |
|---|---|---|---|---|---|---|---|
| clean_plate_with_eraser (34) | normal | 18.77 [17.67, 19.97] | – | – | – | 0.38 [0.33, 0.43] | – |
| clean_plate_with_eraser (34) | zero_force | 31.64 [30.02, 33.72] | +12.87 [+11.52, +14.36] | +14.68 [+13.47, +15.90] | +9.23 [+6.35, +13.21] | 1.90 [1.67, 2.17] | +1.52 [+1.33, +1.74] |
| clean_plate_with_eraser (34) | zero_tactile_only | 21.52 [20.41, 22.66] | +2.75 [+2.16, +3.32] | +4.12 [+3.19, +5.06] | -0.00 [-0.25, +0.19] | 1.93 [1.70, 2.20] | +1.56 [+1.37, +1.78] |
| clean_plate_with_eraser (34) | zero_torque_only | 27.38 [25.68, 29.24] | +8.60 [+7.54, +9.72] | +8.49 [+6.96, +10.42] | +8.82 [+6.07, +12.35] | 0.44 [0.39, 0.52] | +0.07 [+0.04, +0.11] |
| clean_plate_with_eraser (34) | instruction=insert_plug_into_socket | 18.86 [17.74, 20.04] | +0.09 [-0.11, +0.28] | +0.05 [-0.21, +0.30] | +0.18 [-0.01, +0.35] | 0.38 [0.33, 0.42] | -0.00 [-0.01, +0.00] |
| clean_plate_with_eraser (34) | instruction=remove_block_from_box | 18.89 [17.81, 20.07] | +0.12 [-0.08, +0.30] | +0.06 [-0.17, +0.27] | +0.24 [+0.03, +0.43] | 0.38 [0.33, 0.42] | -0.00 [-0.01, +0.00] |
| insert_plug_into_socket (42) | normal | 11.42 [10.51, 12.51] | – | – | – | 0.14 [0.13, 0.15] | – |
| insert_plug_into_socket (42) | zero_force | 31.75 [26.98, 38.03] | +20.33 [+15.34, +26.75] | +9.12 [+7.03, +12.10] | +34.41 [+25.81, +45.19] | 0.80 [0.76, 0.83] | +0.66 [+0.62, +0.69] |
| insert_plug_into_socket (42) | zero_tactile_only | 13.47 [12.24, 14.85] | +2.04 [+1.32, +3.04] | +3.75 [+2.45, +5.59] | -0.10 [-0.44, +0.26] | 0.79 [0.75, 0.83] | +0.65 [+0.61, +0.68] |
| insert_plug_into_socket (42) | zero_torque_only | 27.81 [23.39, 32.89] | +16.39 [+11.71, +21.58] | +2.19 [+1.69, +2.68] | +34.22 [+24.40, +45.11] | 0.16 [0.15, 0.17] | +0.02 [+0.01, +0.03] |
| insert_plug_into_socket (42) | instruction=clean_plate_with_eraser | 11.51 [10.60, 12.58] | +0.08 [-0.09, +0.23] | +0.04 [-0.13, +0.18] | +0.14 [-0.18, +0.42] | 0.14 [0.13, 0.15] | +0.00 [-0.00, +0.00] |
| insert_plug_into_socket (42) | instruction=remove_block_from_box | 11.53 [10.62, 12.64] | +0.11 [-0.04, +0.26] | +0.03 [-0.12, +0.18] | +0.21 [-0.06, +0.48] | 0.14 [0.13, 0.15] | +0.00 [-0.00, +0.00] |
| remove_block_from_box (39) | normal | 10.65 [10.00, 11.32] | – | – | – | 0.07 [0.06, 0.07] | – |
| remove_block_from_box (39) | zero_force | 21.68 [18.46, 25.94] | +11.03 [+7.93, +15.18] | +10.24 [+8.56, +12.44] | +11.46 [+7.02, +17.71] | 0.51 [0.49, 0.54] | +0.45 [+0.42, +0.47] |
| remove_block_from_box (39) | zero_tactile_only | 11.46 [10.82, 12.15] | +0.81 [+0.47, +1.16] | +2.08 [+1.22, +3.12] | +0.11 [-0.16, +0.38] | 0.53 [0.50, 0.56] | +0.46 [+0.44, +0.49] |
| remove_block_from_box (39) | zero_torque_only | 19.79 [16.99, 23.84] | +9.14 [+6.44, +13.14] | +4.60 [+3.31, +5.69] | +11.62 [+7.06, +18.15] | 0.12 [0.11, 0.13] | +0.05 [+0.04, +0.06] |
| remove_block_from_box (39) | instruction=clean_plate_with_eraser | 10.78 [10.14, 11.44] | +0.13 [-0.11, +0.34] | +0.09 [-0.09, +0.27] | +0.16 [-0.14, +0.43] | 0.07 [0.06, 0.07] | +0.00 [-0.00, +0.00] |
| remove_block_from_box (39) | instruction=insert_plug_into_socket | 10.77 [10.14, 11.45] | +0.12 [-0.09, +0.34] | +0.07 [-0.11, +0.27] | +0.15 [-0.13, +0.43] | 0.07 [0.06, 0.07] | +0.00 [-0.00, +0.00] |
| all (115) | normal | 13.95 [12.94, 14.99] | – | – | – | 0.21 [0.17, 0.24] | – |
| all (115) | zero_force | 28.95 [26.00, 32.85] | +15.01 [+12.26, +18.76] | +11.87 [+10.60, +13.27] | +18.72 [+13.33, +26.03] | 1.13 [0.97, 1.31] | +0.92 [+0.79, +1.06] |
| all (115) | zero_tactile_only | 15.91 [14.73, 17.04] | +1.97 [+1.54, +2.49] | +3.62 [+2.87, +4.59] | +0.01 [-0.17, +0.21] | 1.14 [0.98, 1.32] | +0.94 [+0.80, +1.08] |
| all (115) | zero_torque_only | 25.45 [23.00, 28.39] | +11.51 [+9.20, +14.45] | +5.51 [+4.39, +6.66] | +18.61 [+13.10, +25.85] | 0.25 [0.21, 0.30] | +0.05 [+0.03, +0.07] |

Reference: holding the current joint state gives 46.7 mrad, keeping the current force gives 0.24 N.
