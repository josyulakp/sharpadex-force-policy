# Force head edited by hand (`flow/regression_25k`, 3 seeds)

Each of the 30 rows of the force head's last layer is multiplied by one number; 7,710 of 6,726,142 weights change, no gradients. Rows worse than "force stays as it is" on the training data are zeroed; the other gains are chosen on the practice half of the held-out episodes (26,397 frames). Force error (N) on the test half (24,861 frames), mean of seeds.

| | all | in contact | free |
|---|---|---|---|
| original | 0.2020 | 0.3577 | 0.0236 |
| zeroed rows only (training data) | 0.2015 | 0.3568 | 0.0234 |
| gains chosen on training data only | 0.2015 | 0.3569 | 0.0235 |
| one gain for all rows (practice half) | 0.1993 | 0.3525 | 0.0236 |
| **edited: one gain per row** | 0.1989 | 0.3523 | 0.0231 |
| force stays as it is | 0.2415 | 0.4202 | 0.0366 |

Per seed (test half, all frames): seed0 0.2016 -> 0.1983, seed1 0.2004 -> 0.1981, seed2 0.2041 -> 0.2003.
Practice half: 0.2068 -> 0.2015. Training frames: 0.1660 -> 0.1704. Joint error on the test half: 14.62 -> 14.62 mrad.

Gain per fingertip (mean of x, y, z and seeds):

| left thumb | left index | left middle | left ring | left little | right thumb | right index | right middle | right ring | right little |
|---|---|---|---|---|---|---|---|---|---|
| 0.66 | 0.79 | 0.76 | 0.00 | 0.00 | 0.83 | 0.86 | 0.73 | 0.00 | 0.00 |
