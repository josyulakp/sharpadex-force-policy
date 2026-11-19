"""Stage-2 judge: a vision-language model reads the head camera and says how many subgoals are finished.

    <cosmos src>/.venv/bin/python scripts/rl_judge.py [limit=N] [model=...]      (any environment with transformers + torch)
        -> <rl.out_dir>/judge/<model>/<task>/<season>/<episode>.npz : t (judged frames), probs [n, S+1], done (monotone count),
           truth (count from the dataset's own subtask annotation, used only to check the judge)

The model sees the task, the numbered subgoals and a few frames (older -> current) and must answer with one digit:
the number of subgoals already completed. The digit probabilities are read from the next-token logits (no free text).
Per-frame judge columns in Facet-0's format (subgoal_index, subgoal_complete, phase) are derived in rl_reward.py.
"""
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sharpa.config import load_config  # noqa: E402
from sharpa.data.episodes import load_numeric  # noqa: E402
from sharpa.data.split import build_splits  # noqa: E402

cfg = load_config()
J = cfg["rl"]["judge"] | {k: yaml.safe_load(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])}


def subgoals(ref) -> tuple[str, list[str], dict]:
    meta = Path(ref.path).parents[2] / "meta"
    rows = [json.loads(x) for x in (meta / "subtasks.jsonl").read_text().splitlines()]
    steps = [(r["subtask_index"], r["subtask"].split("| Skill:")[0].strip()) for r in rows if not r["subtask"].startswith("Task:")]
    task = json.loads((meta / "tasks.jsonl").read_text().splitlines()[0])["task"]
    instruction = task.split("Instruction:")[1].split("|")[0].strip()
    return instruction, [s for _, s in steps], {i: k for k, (i, _) in enumerate(steps)}


class Episodes(torch.utils.data.Dataset):
    def __init__(self, refs):
        self.refs = refs

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        ref = self.refs[i]
        p = Path(ref.path)
        video = p.parents[2] / "videos" / p.parent.name / f"observation.images.{J['camera']}" / (p.stem + ".mp4")
        ts = list(range(0, ref.num_frames, J["stride"]))
        need = sorted({max(t - b, 0) for t in ts for b in [*J["lookback"], 0]})
        frames, s = {}, J["image_size"]
        with av.open(str(video)) as c:
            for k, f in enumerate(c.decode(c.streams.video[0])):
                if k in need:
                    frames[k] = f.to_image().resize((s, s), Image.BICUBIC)
        instruction, steps, order = subgoals(ref)
        sub = load_numeric(ref)["subtask_index"]
        # ground truth for checking only: subgoals finished before frame t (the annotation runs through them in order)
        pos = np.array([order.get(int(x), -1) for x in sub])
        seen = np.maximum.accumulate(pos)
        truth = np.where(pos >= 0, pos, seen + 1)                         # outside a segment: everything seen so far is done
        return {"ref": ref, "ts": ts, "frames": frames, "instruction": instruction, "steps": steps, "truth": truth[ts]}


def main():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    splits = build_splits(cfg)
    refs = [r for t in cfg["data"]["tasks"] for part in ("heldout", "train") for r in splits[t][part]]
    if J.get("limit"):
        refs = refs[::max(1, len(refs) // J["limit"])][:J["limit"]]
    name = J["model"].split("/")[-1] + ("" if J["question"] == "count" else "__" + J["question"]) + ("__4bit" if J["quant"] == "4bit" else "")
    out_root = Path(cfg["rl"]["out_dir"]) / "judge" / name
    todo = [r for r in refs if not (out_root / r.task / r.season / f"{r.episode_index:06d}.npz").exists()]
    print(f"{len(todo)} of {len(refs)} episodes to judge with {J['model']}", flush=True)
    proc = AutoProcessor.from_pretrained(J["model"], min_pixels=J["image_size"] ** 2 // 2, max_pixels=J["image_size"] ** 2)
    proc.tokenizer.padding_side = "left"
    quant = {}
    if J["quant"] == "4bit":                                              # fits next to another job on the GPU
        from transformers import BitsAndBytesConfig
        quant = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(J["model"], torch_dtype=torch.bfloat16, device_map="cuda", **quant).eval()
    digit = [proc.tokenizer.encode(str(k), add_special_tokens=False)[0] for k in range(10)]
    loader = torch.utils.data.DataLoader(Episodes(todo), batch_size=None, num_workers=12)
    agree, guess = [], []
    for n, ep in enumerate(loader):
        S = len(ep["steps"])
        listing = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(ep["steps"]))
        question = (f"A two-handed robot is doing this task: {ep['instruction']}\nThe task has {S} steps, done in order:\n{listing}\n"
                    f"The images are from the robot's head camera, oldest first; the last image is now. "
                    f"How many of the steps are already fully completed now? Answer with one number from 0 to {S}.")
        first = 0
        if J["question"] == "current":                                    # which step is in progress; answer k means k - 1 are done
            question = (question.rsplit("How many", 1)[0] + f"Which step is the robot doing now? Answer with one number from 1 to {S}, "
                        f"or {S + 1} if every step is finished.")
            first = 1
        texts, images = [], []
        for t in ep["ts"]:
            ims = [ep["frames"][max(t - b, 0)] for b in [*J["lookback"], 0]]
            msg = [{"role": "user", "content": [*({"type": "image"} for _ in ims), {"type": "text", "text": question}]}]
            texts.append(proc.apply_chat_template(msg, add_generation_prompt=True))
            images.append(ims)
        probs = []
        for s in range(0, len(texts), J["batch_size"]):
            batch = proc(text=texts[s:s + J["batch_size"]], images=[i for ims in images[s:s + J["batch_size"]] for i in ims],
                         return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                logits = model(**batch).logits[:, -1].float()
            probs.append(torch.softmax(logits[:, digit[first:first + S + 1]], -1).cpu().numpy())
        probs = np.concatenate(probs)
        done = np.maximum.accumulate(probs.argmax(1))                     # completed steps never decrease
        ref = ep["ref"]
        out = out_root / ref.task / ref.season / f"{ref.episode_index:06d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, t=np.array(ep["ts"]), probs=probs, done=done, raw=probs.argmax(1), truth=np.asarray(ep["truth"]), steps=np.array(ep["steps"]), uid=ref.uid)
        truth = np.asarray(ep["truth"])
        agree.append([float((done == truth).mean()), float((probs.argmax(1) == truth).mean())])
        guess.append(float(np.bincount(truth).max() / len(truth)))        # always answering the episode's most common count
        if n % 20 == 0 or n == len(todo) - 1:
            a = np.mean(agree, 0)
            print(f"{n + 1}/{len(todo)} agreement with the dataset annotation: monotone {a[0]:.3f}, raw {a[1]:.3f}; "
                  f"constant guess {np.mean(guess):.3f}", flush=True)


if __name__ == "__main__":
    main()
