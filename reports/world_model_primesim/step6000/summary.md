# PrimeSim on held-out eraser windows, checkpoint step 6000

16 windows of 3.2 s from 8 held-out episodes, 30 denoising steps. Mean absolute pixel error in grey levels (0-255) over the 32 generated frames; "moving" = pixels where the recording differs from the first frame by more than 25.

## All pixels, 3.2 s
| Source | head | wrist_left | wrist_right |
|---|---|---|---|
| first_frame | 26.4 | 15.5 | 23.3 |
| recorded | 11.4 | 9.0 | 11.2 |
| zero | 26.5 | 15.6 | 23.3 |
| swapped | 24.2 | 15.2 | 20.7 |

## All pixels, first 1.2 s
| Source | head | wrist_left | wrist_right |
|---|---|---|---|
| first_frame | 18.4 | 12.1 | 17.1 |
| recorded | 9.7 | 7.7 | 9.5 |
| zero | 18.5 | 12.2 | 17.1 |
| swapped | 21.0 | 12.6 | 19.0 |

## Moving pixels, 3.2 s
| Source | head | wrist_left | wrist_right |
|---|---|---|---|
| first_frame | 106.5 | 64.4 | 61.6 |
| recorded | 32.8 | 24.4 | 20.5 |
| zero | 106.3 | 63.9 | 61.1 |
| swapped | 66.2 | 43.0 | 39.3 |

## Difference from the video generated with the recorded commands (all views)
- zero: 21.2
- swapped: 17.6

Share of moving pixels: head 20%, wrist_left 17%, wrist_right 30%
