# Policies inside the video world model (PrimeSim `sharpa_three_tasks`), every seed

Distance between the policy's commands and the operator's at the same time in the recording (mrad). It measures divergence, not error: after a few seconds the two need not be doing the same thing. A policy with force inputs gets them replayed from the recording. Each cell: mean over episodes and seeds (per-seed means in brackets).

## `headline/full`, clean_plate_with_eraser: 3 seeds, 4 held-out episodes each

| | 0.0-3.2 s | 3.2-6.4 s | 6.4-9.6 s |
|---|---|---|---|
| all joints (mrad) | 74 (81, 77, 64) | 106 (124, 98, 95) | 113 (123, 109, 106) |
| arms (mrad) | 36 (40, 39, 29) | 67 (79, 63, 59) | 84 (96, 78, 78) |
| force forecast gap (N) | 0.21 (0.21, 0.21, 0.21) | 0.47 (0.45, 0.48, 0.46) | 0.59 (0.60, 0.61, 0.57) |

## `headline/full`, insert_plug_into_socket: 3 seeds, 4 held-out episodes each

| | 0.0-3.2 s | 3.2-6.4 s |
|---|---|---|
| all joints (mrad) | 34 (35, 34, 32) | 69 (70, 70, 68) |
| arms (mrad) | 18 (18, 18, 19) | 36 (36, 34, 36) |
| force forecast gap (N) | 0.18 (0.18, 0.18, 0.17) | 0.17 (0.18, 0.17, 0.16) |

## `headline/full`, remove_block_from_box: 3 seeds, 4 held-out episodes each

| | 0.0-3.2 s | 3.2-6.4 s |
|---|---|---|
| all joints (mrad) | 27 (27, 27, 27) | 60 (62, 58, 60) |
| arms (mrad) | 26 (27, 25, 27) | 33 (38, 29, 33) |
| force forecast gap (N) | 0.09 (0.10, 0.07, 0.09) | 0.13 (0.13, 0.14, 0.13) |
