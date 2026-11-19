# PrimeSim on whole held-out eraser episodes, checkpoint step 6000

3 held-out episodes (15-27 s), operator's commands, 30 denoising steps, started 3.2 s into the recording. Mean absolute pixel error in grey levels (0-255) per 3.2 s of episode time, for the first 3 blocks (present in every episode). first_frame = the first frame of the episode repeated. chained: 3.2 s windows, each started from the last generated frame. reanchored: every window started from the real frame. receding: 0.5 s kept per window, later commands held.

## head
| Source | 0.0-3.2 s | 3.2-6.4 s | 6.4-9.6 s |
|---|---|---|---|
| first_frame | 24.1 | 36.7 | 38.5 |
| receding_known | 15.1 | 21.2 | 25.8 |
| receding16 | 14.4 | 20.3 | 23.5 |

## wrist_left
| Source | 0.0-3.2 s | 3.2-6.4 s | 6.4-9.6 s |
|---|---|---|---|
| first_frame | 18.8 | 23.4 | 23.4 |
| receding_known | 12.8 | 18.8 | 22.2 |
| receding16 | 10.0 | 14.2 | 16.1 |

## wrist_right
| Source | 0.0-3.2 s | 3.2-6.4 s | 6.4-9.6 s |
|---|---|---|---|
| first_frame | 17.3 | 33.3 | 34.6 |
| receding_known | 10.0 | 18.0 | 22.4 |
| receding16 | 8.3 | 15.2 | 18.7 |

