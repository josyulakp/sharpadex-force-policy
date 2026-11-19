# TacForeSight tactile world model on SharpaDex (clean_plate_with_eraser)

Held-out episodes, 1999 windows. Prediction is 8 frames (0.27 s) at once. Force error = mean absolute error of the 30 fingertip force components (N).

## One step ahead (0.27 s)
| | all | in contact | free |
|---|---|---|---|
| force read from the real latent (read-out limit) | 0.766 | 1.026 | 0.019 |
| force read from the predicted latent | 1.004 | 1.298 | 0.162 |
| the same with another window's commands | 1.744 | 2.156 | 0.565 |
| force stays as it is | 0.288 | 0.385 | 0.010 |

Joint torque (mean absolute error, raw units): predicted 0.560, torque stays 0.342.
Latent error (mean squared): predicted 0.0634, latent stays 0.0608.

## Chained without seeing a real frame again
| Seconds ahead | 0.27 | 0.53 | 0.80 | 1.07 | 1.33 | 1.60 | 1.87 | 2.13 | 2.40 | 2.67 |
|---|---|---|---|---|---|---|---|---|---|---|
| predicted | 1.004 | 1.113 | 1.177 | 1.211 | 1.231 | 1.246 | 1.255 | 1.260 | 1.264 | 1.267 |
| force stays as it is | 0.288 | 0.611 | 0.851 | 1.052 | 1.230 | 1.393 | 1.541 | 1.678 | 1.810 | 1.937 |
| read-out limit | 0.766 | 0.774 | 0.779 | 0.781 | 0.783 | 0.784 | 0.785 | 0.786 | 0.786 | 0.787 |
