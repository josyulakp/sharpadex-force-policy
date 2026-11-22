# Stage 3 in the simulator: the frozen four-expert model with and without learned corrections

## Stage 3 (critic gradient and imitation of successes): 29 held-out scenes

**Scenes as reconstructed** (one run per scene; 95 % interval over scenes)

| | Operator's commands | Frozen model | With corrections | Corrections minus frozen (paired) |
|---|---|---|---|---|
| Task success | 37.9% [20.7, 55.2] | 31.0% [13.8, 48.3] | 72.4% [55.2, 86.2] | +41.4 pts [+20.7, +62.1] |
| Plate picked up | 89.7% [79.3, 100.0] | 89.7% [79.3, 100.0] | 93.1% [82.8, 100.0] | +3.4 pts [-6.9, +13.8] |
| Plate back on the table at the end | 75.9% [58.6, 89.7] | 72.4% [55.2, 86.2] | 89.7% [75.9, 100.0] | +17.2 pts [+3.4, +34.5] |
| Plate dropped off the table | 17.2% [3.4, 31.0] | 17.2% [3.4, 31.0] | 6.9% [0.0, 17.2] | -10.3 pts [-27.6, +6.9] |
| Wiped path, cm | 8.7 [5.9, 11.4] | 10.2 [6.7, 14.0] | 16.6 [12.9, 20.3] | +6.3 [+2.7, +10.2] |

**Scenes perturbed** (20 perturbations per scene; 95 % interval over scenes)

| | Operator's commands | Frozen model | With corrections | Corrections minus frozen (paired) |
|---|---|---|---|---|
| Task success | 35.9% [26.9, 45.5] | 29.3% [21.4, 38.1] | 62.9% [54.3, 71.2] | +33.6 pts [+25.0, +42.4] |
| Plate picked up | 87.8% [82.4, 92.2] | 85.5% [78.4, 91.9] | 94.7% [90.0, 97.6] | +9.1 pts [+4.3, +14.7] |
| Plate back on the table at the end | 79.7% [74.5, 84.7] | 76.4% [71.7, 81.2] | 85.7% [80.3, 90.3] | +9.3 pts [+2.9, +15.5] |
| Plate dropped off the table | 15.3% [11.0, 19.7] | 15.0% [11.4, 18.8] | 8.4% [5.0, 12.8] | -6.6 pts [-11.2, -1.6] |
| Wiped path, cm | 9.1 [7.5, 10.7] | 8.6 [6.8, 10.6] | 15.4 [13.3, 17.6] | +6.8 [+5.0, +8.6] |

## Stage 3, second training seed: 29 held-out scenes

**Scenes as reconstructed** (one run per scene; 95 % interval over scenes)

| | Operator's commands | Frozen model | With corrections | Corrections minus frozen (paired) |
|---|---|---|---|---|
| Task success | 37.9% [20.7, 55.2] | 31.0% [13.8, 48.3] | 55.2% [37.9, 72.4] | +24.1 pts [+3.4, +44.8] |
| Plate picked up | 89.7% [79.3, 100.0] | 89.7% [79.3, 100.0] | 89.7% [75.9, 100.0] | +0.0 pts [-13.8, +13.8] |
| Plate back on the table at the end | 75.9% [58.6, 89.7] | 72.4% [55.2, 86.2] | 79.3% [65.5, 93.1] | +6.9 pts [-17.2, +31.0] |
| Plate dropped off the table | 17.2% [3.4, 31.0] | 17.2% [3.4, 31.0] | 10.3% [0.0, 24.1] | -6.9 pts [-27.6, +13.8] |
| Wiped path, cm | 8.7 [5.9, 11.4] | 10.2 [6.7, 14.0] | 15.7 [11.5, 20.0] | +5.5 [+0.1, +10.5] |

**Scenes perturbed** (20 perturbations per scene; 95 % interval over scenes)

| | Operator's commands | Frozen model | With corrections | Corrections minus frozen (paired) |
|---|---|---|---|---|
| Task success | 35.9% [26.9, 45.5] | 29.3% [21.4, 38.1] | 57.9% [49.5, 66.0] | +28.6 pts [+19.7, +37.2] |
| Plate picked up | 87.8% [82.4, 92.2] | 85.5% [78.4, 91.9] | 94.5% [90.3, 97.9] | +9.0 pts [+3.3, +15.3] |
| Plate back on the table at the end | 79.7% [74.5, 84.7] | 76.4% [71.7, 81.2] | 77.4% [71.9, 82.2] | +1.0 pts [-6.4, +7.8] |
| Plate dropped off the table | 15.3% [11.0, 19.7] | 15.0% [11.4, 18.8] | 14.3% [10.2, 19.8] | -0.7 pts [-6.0, +5.5] |
| Wiped path, cm | 9.1 [7.5, 10.7] | 8.6 [6.8, 10.6] | 15.0 [12.8, 17.3] | +6.4 [+4.7, +8.1] |

## Matched control (imitation of successes only, no critic gradient): 29 held-out scenes

**Scenes as reconstructed** (one run per scene; 95 % interval over scenes)

| | Operator's commands | Frozen model | With corrections | Corrections minus frozen (paired) |
|---|---|---|---|---|
| Task success | 37.9% [20.7, 55.2] | 31.0% [13.8, 48.3] | 20.7% [6.9, 37.9] | -10.3 pts [-31.0, +10.3] |
| Plate picked up | 89.7% [79.3, 100.0] | 89.7% [79.3, 100.0] | 82.8% [69.0, 96.6] | -6.9 pts [-24.2, +10.3] |
| Plate back on the table at the end | 75.9% [58.6, 89.7] | 72.4% [55.2, 86.2] | 72.4% [55.2, 86.2] | +0.0 pts [-24.1, +24.1] |
| Plate dropped off the table | 17.2% [3.4, 31.0] | 17.2% [3.4, 31.0] | 20.7% [6.9, 34.5] | +3.4 pts [-13.8, +20.7] |
| Wiped path, cm | 8.7 [5.9, 11.4] | 10.2 [6.7, 14.0] | 8.8 [5.2, 12.4] | -1.5 [-6.9, +3.9] |

**Scenes perturbed** (20 perturbations per scene; 95 % interval over scenes)

| | Operator's commands | Frozen model | With corrections | Corrections minus frozen (paired) |
|---|---|---|---|---|
| Task success | 35.9% [26.9, 45.5] | 29.3% [21.4, 38.1] | 13.6% [9.5, 18.1] | -15.7 pts [-25.0, -7.1] |
| Plate picked up | 87.8% [82.4, 92.2] | 85.5% [78.4, 91.9] | 78.4% [71.2, 85.0] | -7.1 pts [-17.1, +1.7] |
| Plate back on the table at the end | 79.7% [74.5, 84.7] | 76.4% [71.7, 81.2] | 74.3% [69.3, 79.1] | -2.1 pts [-8.4, +4.0] |
| Plate dropped off the table | 15.3% [11.0, 19.7] | 15.0% [11.4, 18.8] | 19.1% [14.7, 23.4] | +4.1 pts [-0.9, +9.1] |
| Wiped path, cm | 9.1 [7.5, 10.7] | 8.6 [6.8, 10.6] | 5.1 [4.0, 6.4] | -3.5 [-5.4, -1.6] |
