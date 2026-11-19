"""Stage-2 judge through the OpenAI API: a hosted vision-language model says how many subgoals are finished.

    1. put the key in OPENAI_API_KEY (preferred) or in the untracked file .openai_key at the repository root
    2. python scripts/rl_judge_api.py dry_run=true            # builds one request, prints it without images, sends nothing
    3. python scripts/rl_judge_api.py limit=12                # 12 episodes (about 200 requests), prints agreement
    4. python scripts/rl_judge_api.py                         # every episode of the three tasks (about 17,000 requests)

    options: shard=k/n model=<id> effort=low|medium|high question=count|current workers=8 limit=N stride=30

Output has the same format as scripts/rl_judge.py:
    <rl.out_dir>/judge/<model>__api[__current]/<task>/<season>/<episode>.npz   (t, probs, done, raw, truth, steps, uid)
Finished episodes are skipped on the next run, so an interrupted run can simply be started again.
This sends head-camera frames of the public SharpaDex episodes and the task text to api.openai.com; nothing else.
"""
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_judge import Episodes, J, build_splits, cfg  # noqa: E402  (same frames, subgoal lists and ground truth as the local judge)

KEY_FILE = Path(__file__).resolve().parents[1] / ".openai_key"     # untracked (.gitignore); alternative to the OPENAI_API_KEY variable
API_KEY = KEY_FILE.read_text().strip() if KEY_FILE.exists() else ""
URL = "https://api.openai.com/v1/responses"
A = {"model": "PUT-THE-MODEL-ID-HERE", "effort": "medium", "workers": 8, "retries": 5, "dry_run": False} | (cfg["rl"].get("judge_api") or {}) | \
    {k: v for k, v in J.items() if k in ("model", "effort", "workers", "retries", "dry_run") and f"{k}=" in " ".join(sys.argv[1:])}


def jpeg(im) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def question(ep) -> tuple[str, int]:
    S = len(ep["steps"])
    listing = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(ep["steps"]))
    head = (f"A two-handed robot is doing this task: {ep['instruction']}\nThe task has {S} steps, done in order:\n{listing}\n"
            f"The images are from the robot's head camera, oldest first; the last image is now. ")
    if J["question"] == "current":                                        # answer k means k - 1 steps are done
        return head + f"Which step is the robot doing now? Answer with one number from 1 to {S}, or {S + 1} if every step is finished.", 1
    return head + f"How many of the steps are already fully completed now? Answer with one number from 0 to {S}.", 0


def ask(key: str, text: str, images: list, S: int, first: int) -> int:
    """One request -> number of finished steps (0..S), or -1 if no usable answer came back."""
    body = {"model": A["model"], "reasoning": {"effort": A["effort"]},
            "input": [{"role": "user", "content": [*({"type": "input_image", "image_url": jpeg(im)} for im in images),
                                                    {"type": "input_text", "text": text + " Reply with the number only."}]}]}
    if A["dry_run"]:
        body["input"][0]["content"] = [c if c["type"] == "input_text" else {"type": "input_image", "image_url": f"<jpeg, {len(c['image_url'])} chars>"}
                                       for c in body["input"][0]["content"]]
        print(json.dumps(body, indent=1))
        sys.exit(0)
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(A["retries"]):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            reply = "".join(c.get("text", "") for item in out.get("output", []) if item.get("type") == "message" for c in item.get("content", []))
            digits = [int(ch) for ch in reply if ch.isdigit()]
            return int(np.clip(digits[0] - first, 0, S)) if digits else -1
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code not in (408, 409, 429, 500, 502, 503, 504):          # wrong key, wrong model id, bad request: stop at once
                sys.exit(f"API error {e.code}: {msg}")
            wait = 2 ** attempt * 2
        except (urllib.error.URLError, TimeoutError):
            wait = 2 ** attempt * 2
        time.sleep(wait)
    return -1


def main():
    key = API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not key and not A["dry_run"]:
        sys.exit("no API key: set OPENAI_API_KEY or put the key in the untracked file .openai_key")
    splits = build_splits(cfg)
    refs = [r for t in cfg["data"]["tasks"] for part in ("heldout", "train") for r in splits[t][part]]
    if J.get("limit"):
        refs = refs[::max(1, len(refs) // J["limit"])][:J["limit"]]
    if J.get("shard"):                                                    # shard=k/n: every n-th episode, for parallel runs
        k, n = map(int, str(J["shard"]).split("/"))
        refs = refs[k::n]
    name = A["model"].split("/")[-1] + "__api" + ("" if J["question"] == "count" else "__" + J["question"])
    out_root = Path(cfg["rl"]["out_dir"]) / "judge" / name
    todo = [r for r in refs if not (out_root / r.task / r.season / f"{r.episode_index:06d}.npz").exists()]
    print(f"{len(todo)} of {len(refs)} episodes to judge with {A['model']} (effort {A['effort']})", flush=True)
    data = Episodes(todo)
    agree, guess, failed = [], [], 0
    with ThreadPoolExecutor(A["workers"]) as pool:
        for n in range(len(data)):
            ep = data[n]
            S = len(ep["steps"])
            text, first = question(ep)
            frames = [[ep["frames"][max(t - b, 0)] for b in [*J["lookback"], 0]] for t in ep["ts"]]
            if A["dry_run"]:
                ask(key, text, frames[0], S, first)                       # prints the request and exits
            raw = np.array(list(pool.map(lambda ims: ask(key, text, ims, S, first), frames)))
            failed += int((raw < 0).sum())
            filled = np.maximum.accumulate(np.where(raw < 0, 0, raw))     # an unanswered frame keeps the count reached so far
            raw = np.where(raw < 0, filled, raw)
            probs = np.eye(S + 1, dtype=np.float32)[raw]
            done, truth, ref = np.maximum.accumulate(raw), np.asarray(ep["truth"]), ep["ref"]
            out = out_root / ref.task / ref.season / f"{ref.episode_index:06d}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, t=np.array(ep["ts"]), probs=probs, done=done, raw=raw, truth=truth, steps=np.array(ep["steps"]), uid=ref.uid)
            agree.append([float((done == truth).mean()), float((raw == truth).mean())])
            guess.append(float(np.bincount(truth).max() / len(truth)))
            if n % 5 == 0 or n == len(data) - 1:
                a = np.mean(agree, 0)
                print(f"{n + 1}/{len(data)} agreement with the dataset annotation: monotone {a[0]:.3f}, raw {a[1]:.3f}; "
                      f"constant guess {np.mean(guess):.3f}; unanswered frames {failed}", flush=True)


if __name__ == "__main__":
    main()
