# Energy-based reasoning on the slot latents (three tasks, held-out episodes, mean of seeds)

Joint error in mrad, force error in N. "Before descent" is the same model decoded from its start state.

| Model | joint | in contact | free | force | joint before descent | force before descent | per seed (joint) |
|---|---|---|---|---|---|---|---|
| two experts, no energy (`three_tasks/full`) | 15.87 | 14.01 | 18.07 | 0.2134 | | | 15.89, 15.79, 15.93 |
| `descent` | 15.73 | 13.97 | 17.81 | 0.2125 | 15.91 | 0.2158 | 15.58, 15.64, 15.96 |
| `direct_update` | 15.68 | 13.90 | 17.79 | 0.2124 | 15.80 | 0.2191 | 15.60, 15.64, 15.82 |
| `from_noise_sxm` | 30.91 | 19.92 | 43.92 | 0.2890 |  |  | 30.75, 31.01, 30.97 |

## `descent`: test-time checks

| | joint (mrad) | force (N) |
|---|---|---|
| 1 steps | 17.09 | 0.2196 |
| 3 steps | 15.84 | 0.2135 |
| 6 steps (as trained) | 15.73 | 0.2125 |
| 12 steps | 16.85 | 0.2220 |
| 24 steps | 19.44 | 0.2429 |
| as trained, no noise | 15.68 | 0.2121 |
| best of 4 by lowest energy | 15.72 | 0.2124 |
| best of 8 by lowest energy | 15.72 | 0.2124 |

Verification: rank correlation between final energy and joint error -0.336; joint error of the fifth of frames with the lowest energy 21.59 mrad, with the highest 9.68 mrad.

## `direct_update`: test-time checks

| | joint (mrad) | force (N) |
|---|---|---|
| 1 steps | 16.16 | 0.2169 |
| 3 steps | 15.76 | 0.2130 |
| 6 steps (as trained) | 15.68 | 0.2124 |
| 12 steps | 15.88 | 0.2144 |
| 24 steps | 16.00 | 0.2154 |
| as trained, no noise | 15.63 | 0.2118 |

## `from_noise_sxm`: test-time checks

| | joint (mrad) | force (N) |
|---|---|---|
| 1 steps | 50.97 | 1.6562 |
| 3 steps | 36.48 | 0.6910 |
| 6 steps (as trained) | 30.91 | 0.2890 |
| 12 steps | 31.58 | 0.3452 |
| 24 steps | 32.13 | 0.3828 |
| as trained, no noise | 30.90 | 0.2876 |
| best of 4 by lowest energy | 30.91 | 0.2886 |
| best of 8 by lowest energy | 30.95 | 0.2907 |

Verification: rank correlation between final energy and joint error -0.249; joint error of the fifth of frames with the lowest energy 46.24 mrad, with the highest 20.10 mrad.
