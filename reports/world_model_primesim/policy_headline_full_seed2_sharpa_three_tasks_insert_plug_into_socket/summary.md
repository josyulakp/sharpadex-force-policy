# Policy `headline/full/seed2` inside PrimeSim (step 6000)

4 held-out `insert_plug_into_socket` episodes, world model `sharpa_three_tasks`, started 3.2 s into the recording, 30 denoising steps.

Distance between the policy's commands and the operator's at the same time in the recording (mrad, all joints / arms). It is a measure of divergence, not of error: after a few seconds the two need not be doing the same thing.

| | 0.0-3.2 s | 3.2-6.4 s |
|---|---|---|
| all joints | 32 | 68 |
| arms | 19 | 36 |
| predicted vs recorded fingertip force (N) | 0.17 | 0.16 |

Force inputs: recorded. The force row compares the policy's force forecast (up to 0.5 s ahead) with the force in the recording.
