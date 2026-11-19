# SharpaDex data report (step 1)

Contact = any fingertip |F| > 1.0 N. Held-out = 10% of episodes per task, split_seed=0. Split file: `splits/split_seed0.json`.

| | clean_plate_with_eraser | insert_plug_into_socket | remove_block_from_box | total |
|---|---|---|---|---|
| seasons | 5 | 6 | 5 | 16 |
| episodes | 343 | 416 | 388 | 1147 |
| hours | 1.76 | 1.70 | 1.32 | 4.78 |
| episode length mean / min / max (s) | 18.5 / 12.5 / 55.5 | 14.7 / 10.1 / 44.1 | 12.2 / 10.1 / 16.9 |  |
| rate | 30 Hz (max dev 0.003 ms) | 30 Hz (max dev 0.003 ms) | 30 Hz (max dev 0.001 ms) |  |
| non-finite values / bad frame index | 0 / 0 | 0 / 0 | 0 / 0 |  |
| fingertip |F| at rest, median (N) | 0.000 | 0.005 | 0.007 |  |
| frames in contact (> 1.0 N) | 66.0% | 56.3% | 34.8% |  |
|   left hand / right hand | 60.6% / 45.5% | 0.0% / 56.3% | 0.0% / 34.8% |  |
| contact onsets per episode | 3.8 | 1.1 | 1.6 |  |
| peak fingertip |F| in contact, median / p95 (N) | 17.5 / 39.4 | 19.8 / 30.5 | 19.1 / 25.1 |  |
| action leads state by (frames) arm / hand | 3 / 3 | 4 / 5 | 3 / 5 |  |
| train / held-out episodes | 309 / 34 | 374 / 42 | 349 / 39 | 1032 / 115 |
| train / held-out hours | 1.59 / 0.18 | 1.53 / 0.17 | 1.19 / 0.13 |  |
| held-out frames contact / free | 12724 / 6355 | 10102 / 8045 | 4952 / 9080 |  |
| train share 10%: episodes (h) | 31 (0.16) | 38 (0.16) | 35 (0.12) |  |
| train share 25%: episodes (h) | 78 (0.40) | 94 (0.38) | 88 (0.30) |  |
| train share 50%: episodes (h) | 155 (0.79) | 187 (0.76) | 175 (0.59) |  |
| train share 100%: episodes (h) | 309 (1.59) | 374 (1.53) | 349 (1.19) |  |

## Contact by subtask phase

- **clean_plate_with_eraser**: none 14% (11 min), pick 52% (52 min), place 86% (12 min), wipe 100% (31 min)
- **insert_plug_into_socket**: align 100% (27 min), move 100% (14 min), none 5% (16 min), pick 7% (31 min), plug 98% (14 min)
- **remove_block_from_box**: move 100% (12 min), none 18% (16 min), pick 11% (43 min), place 99% (8 min)

## Contact fraction vs threshold

| threshold (N) | clean_plate_with_eraser | insert_plug_into_socket | remove_block_from_box |
|---|---|---|---|
| 0.25 | 75.4% | 70.2% | 52.7% |
| 0.5 | 70.7% | 62.3% | 46.5% |
| 1.0 | 66.0% | 56.3% | 34.8% |
| 2.0 | 62.3% | 54.9% | 31.2% |
| 3.0 | 61.1% | 54.5% | 30.7% |
| 5.0 | 60.3% | 54.2% | 30.2% |
| 10.0 | 57.2% | 52.4% | 29.2% |

## Held-out episodes by season

- **clean_plate_with_eraser**: _2026_04_17_11_01_59: 2, _2026_04_18_10_54_31: 4, _2026_04_18_14_25_36: 9, _2026_04_18_16_43_25: 5, _2026_04_18_19_27_11: 14
- **insert_plug_into_socket**: _2026_03_18_13_58_40: 11, _2026_03_18_14_57_46: 12, _2026_03_18_16_19_30: 4, _2026_03_18_17_16_26: 5, _2026_03_18_19_22_03: 4, _2026_03_18_19_36_01: 6
- **remove_block_from_box**: _2026_05_01_10_36_40: 7, _2026_05_01_14_06_37: 8, _2026_05_01_15_34_56: 6, _2026_05_01_16_22_53: 6, _2026_05_01_19_17_40: 12

## Channels with no signal

- **clean_plate_with_eraser**: dead torque dims [58, 59, 60, 61, 62, 63, 64], dead state dims [63], torso state std (rad) [0.001, 0.0, 0.019, 0.012, 0.016, 0.0, 0.059]
- **insert_plug_into_socket**: dead torque dims [58, 59, 60, 61, 62, 63, 64], dead state dims [], torso state std (rad) [0.0, 0.0, 0.003, 0.001, 0.004, 0.017, 0.064]
- **remove_block_from_box**: dead torque dims [58, 59, 60, 61, 62, 63, 64], dead state dims [], torso state std (rad) [0.0, 0.0, 0.012, 0.002, 0.008, 0.041, 0.044]
