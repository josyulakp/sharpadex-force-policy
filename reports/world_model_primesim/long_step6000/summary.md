# PrimeSim on whole held-out eraser episodes, checkpoint step 6000

4 held-out episodes (15-27 s), operator's commands, 30 denoising steps, started 3.2 s into the recording. Mean absolute pixel error in grey levels (0-255) per 3.2 s of episode time, for the first 3 blocks (present in every episode). first_frame = the first frame of the episode repeated. chained: 3.2 s windows, each started from the last generated frame. reanchored: every window started from the real frame. receding: 0.5 s kept per window, later commands held.

## head
| Source | 0.0-3.2 s | 3.2-6.4 s | 6.4-9.6 s |
|---|---|---|---|
| first_frame | 23.0 | 35.1 | 36.8 |
| chained | 11.2 | 15.6 | 17.2 |
| reanchored | 11.2 | 10.0 | 9.8 |
| receding | 16.7 | 25.9 | 36.2 |

## wrist_left
| Source | 0.0-3.2 s | 3.2-6.4 s | 6.4-9.6 s |
|---|---|---|---|
| first_frame | 19.8 | 23.9 | 23.8 |
| chained | 9.3 | 13.3 | 15.3 |
| reanchored | 9.3 | 7.8 | 7.6 |
| receding | 14.4 | 20.9 | 26.4 |

## wrist_right
| Source | 0.0-3.2 s | 3.2-6.4 s | 6.4-9.6 s |
|---|---|---|---|
| first_frame | 18.6 | 33.0 | 33.9 |
| chained | 8.0 | 14.0 | 16.2 |
| reanchored | 8.0 | 10.7 | 9.4 |
| receding | 10.6 | 19.0 | 25.0 |

