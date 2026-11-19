# Stage-2 judge check: can a vision-language model label subgoal progress?

12 episodes spread over the three tasks, head camera, one judgement per second (3 frames: 2 s ago, 1 s ago, now).
Agreement = share of judged frames where the judge's number of finished steps equals the dataset's own subtask annotation.
Constant guess = always answering the episode's most common value (needs the answer in advance, so it is a generous baseline).

| judge | question | agreement | constant guess |
|---|---|---|---|
| Qwen2.5-VL-3B | how many steps are finished | 0.317 | 0.418 |
| Qwen2.5-VL-7B (4-bit) | how many steps are finished | 0.412 | 0.418 |
| Qwen2.5-VL-7B (4-bit) | which step is the robot doing now | 0.145 | 0.418 |

    python scripts/rl_judge.py limit=12 batch_size=4 quant=4bit question=count     (also question=current, model=Qwen/Qwen2.5-VL-3B-Instruct)

No judge beats the constant guess, so its labels are not used as a reward: the critic, the credit assignment and the
advantage-conditioned policy update of Facet-0's stage 2 are not built on them.
