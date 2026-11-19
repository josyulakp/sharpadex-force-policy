# Heavy-tailed residual loss (HT-Policy) on the four-expert model, held-out episodes, mean of seeds

Joint error in mrad: MAE = mean absolute error, RMSE = root-mean-square error. Force error in N.

| Model | MAE | in contact | free | RMSE | in contact | free | force | per seed (MAE) |
|---|---|---|---|---|---|---|---|---|
| two experts, L1, 25k | 14.56 | 13.49 | 15.82 |  |  |  | 0.2045 | 14.58, 14.54, 14.55 |
| four experts, L1, 8k (earlier run) | 15.20 | 13.58 | 17.11 |  |  |  | 0.2162 | 15.18, 15.17, 15.25 |
| four experts, L1, 8k | 15.09 | 13.57 | 16.88 | 47.72 | 37.15 | 57.78 | 0.2130 | 15.00, 15.21, 15.04 |
| four experts, heavy-tailed, 8k | 16.38 | 14.68 | 18.39 | 47.25 | 37.44 | 56.71 | 0.2162 | 16.53, 16.16, 16.46 |
| four experts, L1, 25k | 14.16 | 13.46 | 14.99 | 46.74 | 37.39 | 55.81 | 0.2092 | 14.13, 14.19, 14.16 |
| four experts, heavy-tailed, 25k | 15.57 | 14.54 | 16.78 | 47.18 | 38.13 | 56.03 | 0.2138 | 15.55, 15.53, 15.62 |
| four experts, L1, 50k | 13.95 | 13.52 | 14.46 | 47.02 | 37.99 | 55.84 | 0.2079 | 13.87, 14.03, 13.94 |
| four experts, heavy-tailed, 50k | 15.07 | 14.26 | 16.04 | 46.99 | 37.94 | 55.83 | 0.2126 | 15.04, 15.07, 15.11 |
| four experts, L1, 50k run at step 25k | 14.14 | 13.49 | 14.91 | 46.79 | 37.74 | 55.63 | 0.2080 | 14.13, 14.20, 14.10 |
| four experts, heavy-tailed, 50k run at step 25k | 15.36 | 14.34 | 16.56 | 46.68 | 37.75 | 55.42 | 0.2126 | 15.33, 15.32, 15.43 |

## Heavy-tailed minus L1 (paired over held-out episodes, 95 % interval)

| Steps | MAE (mrad) | RMSE (mrad) | force (N) |
|---|---|---|---|
| 8k | +1.297 [+1.073, +1.532] | -0.468 [-1.253, +0.323] | +0.0032 [+0.0009, +0.0053] |
| 25k | +1.407 [+1.262, +1.571] | +0.440 [-0.100, +0.989] | +0.0046 [-0.0005, +0.0081] |
| 50k | +1.125 [+0.991, +1.269] | -0.030 [-0.571, +0.506] | +0.0047 [-0.0009, +0.0086] |

## Predicted scale of the heavy-tailed model (25k)

Rank correlation between the predicted scale of a chunk and its joint error: +0.860. Mean scale 0.136 (normalised action units; 0.108 in contact, 0.170 free).

| Frames by predicted scale | lowest fifth | 2nd | 3rd | 4th | highest fifth |
|---|---|---|---|---|---|
| joint MAE (mrad) | 4.38 | 8.11 | 12.50 | 16.51 | 36.33 |
| share in contact | 65% | 60% | 67% | 42% | 37% |

## Predicted scale of the heavy-tailed model (50k)

Rank correlation between the predicted scale of a chunk and its joint error: +0.850. Mean scale 0.110 (normalised action units; 0.087 in contact, 0.137 free).

| Frames by predicted scale | lowest fifth | 2nd | 3rd | 4th | highest fifth |
|---|---|---|---|---|---|
| joint MAE (mrad) | 4.26 | 7.76 | 12.30 | 16.23 | 34.81 |
| share in contact | 65% | 59% | 67% | 43% | 37% |
