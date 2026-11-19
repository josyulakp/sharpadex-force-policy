# PrimeSim on held-out eraser windows, checkpoint step 3000

16 windows of 3.2 s from 8 held-out episodes, 30 denoising steps. Mean absolute pixel error in grey levels (0-255) over the 32 generated frames; "moving" = pixels where the recording differs from the first frame by more than 25.

## All pixels, 3.2 s
| Source | head | wrist_left | wrist_right |
|---|---|---|---|
| first_frame | 26.4 | 15.5 | 23.3 |
| recorded | 13.8 | 9.5 | 11.7 |
| zero | 26.5 | 15.5 | 23.3 |
| swapped | 25.0 | 15.1 | 20.6 |

## All pixels, first 1.2 s
| Source | head | wrist_left | wrist_right |
|---|---|---|---|
| first_frame | 18.4 | 12.1 | 17.1 |
| recorded | 11.5 | 8.1 | 9.8 |
| zero | 18.5 | 12.2 | 17.1 |
| swapped | 21.8 | 12.5 | 19.0 |

## Moving pixels, 3.2 s
| Source | head | wrist_left | wrist_right |
|---|---|---|---|
| first_frame | 106.5 | 64.4 | 61.6 |
| recorded | 35.9 | 25.5 | 21.6 |
| zero | 106.2 | 63.8 | 61.1 |
| swapped | 65.1 | 42.4 | 39.2 |

## Difference from the video generated with the recorded commands (all views)
- zero: 21.7
- swapped: 17.3

Share of moving pixels: head 20%, wrist_left 17%, wrist_right 30%
