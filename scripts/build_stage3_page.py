"""Stage 3: the web page with the result, the side-by-side videos of every held-out scene and the curves of all three stages.

    python scripts/build_stage3_page.py [out=reports/stage3/page] [run=rl] [second=rl_seed1] [control=control]
        -> <out>/index.html, <out>/videos/*.mp4, <out>/posters/*.jpg      (self-contained: copy the folder to the web host)

The text of the page is reports/stage3/page_template.html; this script fills in the numbers, charts and videos.

Needs the outputs of scripts/s3_eval.py (all three runs), scripts/s3_fixed.py, scripts/s3_video.py pick=29 and
runs/refine/critic_log.json (scripts/stage_curves.py explains where it comes from). Charts are drawn as inline SVG from
the training logs and evaluation files, so the page carries the numbers it shows.
"""
import html
import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RUNS, VIDEOS = REPO / "runs", REPO / "reports" / "stage3" / "videos"
GH = "https://github.com/josyulakp/sharpadex-force-policy"
SEEDS = (0, 1, 2)


# ---- data ---------------------------------------------------------------------------------------------------
def rolling(name: str, kind: str, window: int = 300, points: int = 240):
    rows = [json.loads(l) for l in (RUNS / "stage3" / name / "log.jsonl").open() if '"error"' not in l]
    sel = [r for r in rows if r["kind"] == kind]
    x, y = np.array([r["index"] for r in sel]), np.array([float(r["success"]) for r in sel])
    c = np.cumsum(np.r_[0.0, y])
    k = np.arange(window, len(y) + 1)
    xs, ys = x[k - 1] / 1000, 100 * (c[k] - c[k - window]) / window
    keep = np.unique(np.r_[np.arange(0, len(xs), max(len(xs) // points, 1)), len(xs) - 1])
    base = [float(r["success"]) for r in sel if r["mean_abs_correction"] == 0]         # before the actor's first update
    return xs[keep].round(2).tolist(), ys[keep].round(1).tolist(), 100 * float(np.mean(base))


def train_log(run: str, key: str, skip: int = 0):
    rows = [json.loads((RUNS / run / f"seed{s}" / "train_log.json").read_text()) for s in SEEDS]
    x = [r["step"] / 1000 for r in rows[0]][skip:]
    return x, np.mean([[r[key] for r in seed] for seed in rows], 0)[skip:].tolist()


def mrad(run: str, tag: str = "") -> float:
    return float(np.mean([1000 * json.loads((RUNS / run / f"seed{s}" / f"eval_heldout{tag}.json").read_text())["act"]["all"] for s in SEEDS]))


# ---- charts (inline SVG; colours come from the page's CSS variables) ----------------------------------------
def line_chart(cid: str, series: list[dict], xlabel: str, ylabel: str, unit: str = "", log: bool = False, yticks=None, refs=(), ymin=None,
               ymax=None, w: int = 860, h: int = 330, decimals: int = 0) -> str:
    """series: {name, x, y, cls, width}. refs: (value, cls, label) horizontal dashed lines. Hover values come from the embedded JSON."""
    L, R, T, B = 52, 64, 12, 40
    xs = [v for s in series for v in s["x"]]
    ys = [v for s in series for v in s["y"]] + [r[0] for r in refs]
    x0, x1 = 0.0, float(np.ceil(max(xs) - 0.01))                                 # logs end one step short of the round number
    f = (lambda v: np.log10(v)) if log else (lambda v: v)
    y0 = f(ymin if ymin is not None else min(ys))
    y1 = f(ymax if ymax is not None else max(ys))
    if ymin is None and not log:
        y0 -= 0.06 * (y1 - y0)
    if ymax is None:
        y1 += 0.06 * (y1 - y0)
    px = lambda v: L + (v - x0) / (x1 - x0) * (w - L - R)  # noqa: E731
    py = lambda v: T + (1 - (f(v) - y0) / (y1 - y0)) * (h - T - B)  # noqa: E731
    if yticks is None:
        step = next(s for s in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if (y1 - y0) / s <= 7)
        yticks = [round(v, 4) for v in np.arange(np.ceil(y0 / step) * step, y1 + 1e-9, step)]
    xstep = next(s for s in (1, 2, 5, 10, 20) if x1 / s <= 9)
    o = [f'<svg id="{cid}" viewBox="0 0 {w} {h}" style="min-width:{min(620, int(0.72 * w))}px" role="img" aria-label="{html.escape(ylabel)} against {html.escape(xlabel)}">']
    for t in yticks:
        o.append(f'<line class="grid" x1="{L}" x2="{w - R}" y1="{py(t):.1f}" y2="{py(t):.1f}"/><text class="tick" x="{L - 8}" y="{py(t) + 4:.1f}" text-anchor="end">{t:g}{unit}</text>')
    for t in np.arange(0, x1 + 1e-9, xstep):
        o.append(f'<text class="tick" x="{px(t):.1f}" y="{h - B + 17}" text-anchor="middle">{t:g}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}"/>')
    o.append(f'<text class="lab" x="{(L + w - R) / 2:.0f}" y="{h - 5}" text-anchor="middle">{html.escape(xlabel)}</text>')
    for v, cls, _ in refs:
        o.append(f'<line class="ref {cls}" x1="{L}" x2="{w - R}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>')
    for s in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(a):.1f},{py(b):.1f}" for i, (a, b) in enumerate(zip(s["x"], s["y"])))
        o.append(f'<path class="ln {s["cls"]}" style="stroke-width:{s.get("width", 2)}" d="{d}"/>')
    ends = sorted(((py(s["y"][-1]), s) for s in series), key=lambda e: e[0])               # end labels, pushed apart where they would collide
    at = [e[0] for e in ends]
    for i in range(1, len(at)):
        at[i] = max(at[i], at[i - 1] + 14)
    for yy, (_, s) in zip(at, ends):
        o.append(f'<text class="end" x="{px(s["x"][-1]) + 7:.1f}" y="{yy + 4:.1f}">{s["y"][-1]:.{decimals}f}{unit}</text>')
    o.append(f'<line class="cross" x1="0" x2="0" y1="{T}" y2="{h - B}" hidden/><rect class="hit" x="{L}" y="{T}" width="{w - L - R}" height="{h - T - B}"/></svg>')
    data = {"x0": x0, "x1": x1, "L": L, "R": R, "w": w, "unit": unit, "decimals": decimals, "xlabel": xlabel,
            "series": [{"name": s["name"], "cls": s["cls"], "x": s["x"], "y": s["y"]} for s in series]}
    key = "".join(f'<span><i class="{s["cls"]}"></i>{html.escape(s["name"])}</span>' for s in series)
    key += "".join(f'<span><i class="dash {cls}"></i>{html.escape(label)}</span>' for _, cls, label in refs)
    return (f'<figure class="chart"><div class="ylab">{html.escape(ylabel)}</div><div class="scroll">{"".join(o)}</div><div class="key">{key}</div>'
            f'<script type="application/json" data-for="{cid}">{json.dumps(data)}</script></figure>')


def bar_chart(bars: list[tuple], ylabel: str, w: int = 860, h: int = 300) -> str:
    """bars: (label, [mean, low, high] in %, cls)."""
    L, R, T, B = 52, 16, 22, 46
    top = 10 * np.ceil(max(b[1][2] for b in bars) / 10 + 0.5)
    py = lambda v: T + (1 - v / top) * (h - T - B)  # noqa: E731
    slot = (w - L - R) / len(bars)
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(ylabel)}">']
    for t in np.arange(0, top + 1, 10 if top <= 80 else 20):
        o.append(f'<line class="grid" x1="{L}" x2="{w - R}" y1="{py(t):.1f}" y2="{py(t):.1f}"/><text class="tick" x="{L - 8}" y="{py(t) + 4:.1f}" text-anchor="end">{t:g}%</text>')
    for k, (label, v, cls) in enumerate(bars):
        cx, bw = L + slot * (k + 0.5), min(slot * 0.42, 84)
        tip = f"{label.replace(chr(10), ' ')}: {v[0]:.1f}% [{v[1]:.1f}, {v[2]:.1f}]"
        o.append(f'<g><title>{html.escape(tip)}</title><path class="bar {cls}" d="M{cx - bw / 2:.1f},{py(0):.1f} V{py(v[0]) + 4:.1f} q0,-4 4,-4 H{cx + bw / 2 - 4:.1f} q4,0 4,4 V{py(0):.1f} Z"/>'
                 f'<line class="err" x1="{cx:.1f}" x2="{cx:.1f}" y1="{py(v[1]):.1f}" y2="{py(v[2]):.1f}"/>'
                 f'<text class="val" x="{cx:.1f}" y="{py(v[2]) - 7:.1f}" text-anchor="middle">{v[0]:.0f}%</text></g>')
        for j, part in enumerate(label.split("\n")):
            o.append(f'<text class="cat" x="{cx:.1f}" y="{h - B + 17 + 15 * j}" text-anchor="middle">{html.escape(part)}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{w - R}" y1="{py(0):.1f}" y2="{py(0):.1f}"/></svg>')
    return f'<figure class="chart"><div class="ylab">{html.escape(ylabel)}</div><div class="scroll">{"".join(o)}</div></figure>'


# ---- page ---------------------------------------------------------------------------------------------------
def pct(v, signed=False):
    g = "+" if signed else ""
    return (f"{100 * v[0]:{g}.1f}{' points' if signed else ' %'} <span class=\"ps\">[{100 * v[1]:{g}.1f}, {100 * v[2]:{g}.1f}]</span>").replace("-", "−")


def videos(out: Path, ev: dict) -> tuple[str, dict]:
    """Copy the videos, cut a poster from each, and return the cards and the count per kind of outcome."""
    index = json.loads((VIDEOS / "index.json").read_text())["videos"]
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "posters").mkdir(parents=True, exist_ok=True)
    per = {}
    for e in ev["episodes"]:
        if e["repeat"] > 0:
            per.setdefault((e["uid"], e["policy"]), []).append(e["success"])
    kinds = {"fixed": "Corrections fix a failure", "both": "Both succeed", "neither": "Both fail", "broken": "Corrections break a success"}
    cards, count = [], dict.fromkeys(kinds, 0)
    order = {k: i for i, k in enumerate(kinds)}
    rows = []
    for v in index:
        a, b = v["outcome"]["frozen four-expert policy"], v["outcome"]["with stage-3 corrections"]
        kind = {(False, True): "fixed", (True, True): "both", (False, False): "neither", (True, False): "broken"}[(a["success"], b["success"])]
        rows.append((order[kind], -sum(per[(v["uid"], "stage3")]), v, a, b, kind))
    for n, (_, _, v, a, b, kind) in enumerate(sorted(rows, key=lambda r: r[:2])):
        count[kind] += 1
        name = f"scene{n + 1:02d}"
        shutil.copyfile(VIDEOS / v["video"], out / "videos" / f"{name}.mp4")
        with av.open(str(VIDEOS / v["video"])) as c:
            frames = [f for f in c.decode(video=0)]
        Image.fromarray(frames[int(0.62 * len(frames))].to_ndarray(format="rgb24")).save(out / "posters" / f"{name}.jpg", quality=80)
        season, episode = v["uid"].split("/")[1:]
        f20, s20 = sum(per[(v["uid"], "frozen")]), sum(per[(v["uid"], "stage3")])
        word = lambda o: "success" if o["success"] else "fail"  # noqa: E731
        cards.append(
            f'<figure class="vid" data-kind="{kind}"><div class="cols"><span>frozen model</span><span>with corrections</span></div>'
            f'<video controls muted loop playsinline preload="none" poster="posters/{name}.jpg" src="videos/{name}.mp4" '
            f'aria-label="Scene {n + 1}: frozen model on the left, with corrections on the right"></video>'
            f'<figcaption><span><b>{kinds[kind]}</b> · {word(a)} → {word(b)} · wiped {a["wipe_path_smooth_cm"]:.0f} → {b["wipe_path_smooth_cm"]:.0f} cm</span>'
            f'<span class="p20">under 20 perturbations: {f20} → {s20} successes</span>'
            f'<span class="uid">{html.escape(season.replace("season_", "").replace("_train", ""))} / {episode}</span></figcaption></figure>')
    buttons = f'<button type="button" data-pick="" aria-pressed="true">All {len(index)}</button>' + "".join(
        f'<button type="button" data-pick="{k}" aria-pressed="false">{label} ({count[k]})</button>' for k, label in kinds.items() if count[k])
    return f'<div class="bar"><span class="lab" id="kindlab">Show</span><div class="seg" role="group" aria-labelledby="kindlab">{buttons}</div></div><div class="vids">{"".join(cards)}</div>', count


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    out = REPO / args.get("out", "reports/stage3/page")
    run, second, control = args.get("run", "rl"), args.get("second", "rl_seed1"), args.get("control", "control")
    out.mkdir(parents=True, exist_ok=True)
    ev = {n: json.loads((RUNS / "stage3" / n / "eval_heldout.json").read_text()) for n in (run, second, control)}
    fx = json.loads((RUNS / "stage3" / run / "fixed_heldout.json").read_text())["summary"]["success"]
    S = {n: e["summary"]["perturbed"] for n, e in ev.items()}
    s, s2, sc = S[run]["success"], S[second]["success"], S[control]["success"]
    raw = ev[run]["summary"]["as reconstructed"]["success"]

    # the result
    bars = bar_chart([("operator's\ncommands", [100 * v for v in s["operator"]], "c-op"), ("frozen\nmodel", [100 * v for v in s["frozen"]], "c-frozen"),
                      ("matched control\n(no critic gradient)", [100 * v for v in sc["stage3"]], "c-ctl"), ("with stage-3\ncorrections", [100 * v for v in s["stage3"]], "c-s3"),
                      ("stage 3,\nsecond seed", [100 * v for v in s2["stage3"]], "c-s3b")], "task success on the held-out scenes, % (line = 95 % interval over scenes)")
    rows = [("Task success", "success", pct), ("Plate picked up", "plate_lifted", pct), ("Plate dropped off the table", "plate_dropped", pct)]
    table = "".join(f'<tr{" class=hl" if k == "success" else ""}><td>{label}</td><td>{f(S[run][k]["operator"])}</td><td>{f(S[run][k]["frozen"])}</td><td>{f(S[run][k]["stage3"])}</td>'
                    f'<td>{f(S[run][k]["stage3 - frozen"], True)}</td></tr>' for label, k, f in rows)
    w = S[run]["wipe_path_smooth_cm"]
    cm = lambda v, g="": f"{v[0]:{g}.1f} cm <span class=\"ps\">[{v[1]:{g}.1f}, {v[2]:{g}.1f}]</span>".replace("-", "−")  # noqa: E731
    table += f'<tr><td>Eraser travel on the held plate</td><td>{cm(w["operator"])}</td><td>{cm(w["frozen"])}</td><td>{cm(w["stage3"])}</td><td>{cm(w["stage3 - frozen"], "+")}</td></tr>'

    # learning curve
    mx, my, mbase = rolling(run, "monitor")
    vx, vy, vbase = rolling(run, "validation")
    v2x, v2y, _ = rolling(second, "validation")
    cx, cy, _ = rolling(control, "validation")
    learn = line_chart("learn", [{"name": "training scenes", "x": mx, "y": my, "cls": "c-frozen", "width": 2},
                                 {"name": "validation scenes (never trained in)", "x": vx, "y": vy, "cls": "c-s3", "width": 2.4},
                                 {"name": "second seed, validation scenes", "x": v2x, "y": v2y, "cls": "c-s3b", "width": 1.6},
                                 {"name": "matched control, validation scenes", "x": cx, "y": cy, "cls": "c-ctl", "width": 1.6}],
                       "training episodes, thousands", "success in episodes without exploration noise, last 300 such episodes", unit="%", ymin=0, ymax=75,
                       yticks=[0, 10, 20, 30, 40, 50, 60, 70], refs=[(vbase, "c-s3", "frozen model in the validation scenes"), (mbase, "c-frozen", "frozen model in the training scenes")])

    # where the gain comes from
    def modes(pol):
        e = [x for x in ev[run]["episodes"] if x["policy"] == pol and x["repeat"] > 0]
        m = lambda x: ("ok" if x["success"] else "lift" if not x["plate_lifted"] else "drop" if x["plate_dropped"]  # noqa: E731
                       else "short" if x["wipe_path_smooth_cm"] < 10 else "place")
        return {k: 100 * sum(m(x) == k for x in e) / len(e) for k in ("ok", "short", "lift", "drop", "place")}, len(e)
    (mf, n_runs), (ms, _) = modes("frozen"), modes("stage3")
    mode_rows = "".join(f"<tr{' class=hl' if k == 'ok' else ''}><td>{label}</td><td>{mf[k]:.1f} %</td><td>{ms[k]:.1f} %</td></tr>" for k, label in
                        (("ok", "Success"), ("short", "Eraser travel under 10 cm"), ("lift", "Plate never picked up"), ("drop", "Plate dropped off the table"),
                         ("place", "Wiped, plate not set down properly")))
    fixed_rows = (f'<tr><td>Frozen model</td><td>{100 * fx["frozen"][0]:.1f} %</td><td></td></tr>'
                  f'<tr><td>The actor\'s mean correction, held constant</td><td>{100 * fx["constant"][0]:.1f} %</td><td>{pct(fx["constant - frozen"], True)}</td></tr>'
                  f'<tr><td>Its mean while the recording wipes, and its mean otherwise</td><td>{100 * fx["two_phase"][0]:.1f} %</td><td>{pct(fx["two_phase - frozen"], True)}</td></tr>'
                  f'<tr class=hl><td>The actor</td><td>{100 * fx["stage3"][0]:.1f} %</td><td>{pct(fx["stage3 - frozen"], True)}</td></tr>')

    # stages 1 and 2
    x, total = train_log("headline/full", "loss", 1)
    s1 = line_chart("s1", [{"name": "total loss (all terms, weighted)", "x": x, "y": total, "cls": "c-frozen"},
                           {"name": "joint targets (L1, normalised units)", "x": x, "y": train_log("headline/full", "action", 1)[1], "cls": "c-s3"},
                           {"name": "fingertip-force forecast", "x": x, "y": train_log("headline/full", "force", 1)[1], "cls": "c-ctl"}],
                    "training steps, thousands", "training loss, mean of 3 seeds, one batch every 200 steps (log scale)", log=True, yticks=[0.02, 0.05, 0.1, 0.2, 0.5, 1],
                    ymin=0.012, ymax=1.3, decimals=3)
    critic = json.loads((RUNS / "refine" / "critic_log.json").read_text())
    kx = [r["step"] / 1000 for r in critic]
    s2a = line_chart("s2a", [{"name": "critic: |value − return| on held-out episodes", "x": kx, "y": [r["heldout_abs_error"] for r in critic], "cls": "c-s3"}],
                     "critic training steps, thousands", "|value − return| on held-out episodes", ymin=0, decimals=2,
                     refs=[(critic[0]["heldout_abs_error_constant"], "c-frozen", "constant guess (mean return of the training episodes)")], w=470, h=270)
    s2c = line_chart("s2c", [{"name": "critic: cross-entropy to its projected target", "x": kx[1:], "y": [r["loss"] for r in critic[1:]], "cls": "c-frozen"}],
                     "critic training steps, thousands", "critic training loss, mean of the last 2,000 steps", decimals=2, w=470, h=270)
    rx, cont = train_log("refine/continued", "action")
    s2b = line_chart("s2b", [{"name": "continued training (control)", "x": rx, "y": cont, "cls": "c-ctl"},
                             {"name": "refined: plain branch", "x": rx, "y": train_log("refine/refined", "action")[1], "cls": "c-frozen"},
                             {"name": "refined: branch tagged as positive credit", "x": rx, "y": train_log("refine/refined", "action_tagged")[1], "cls": "c-s3"}],
                     "training steps, thousands (8k more steps from the trained flow policy)", "flow-matching loss on joint targets, mean of 3 seeds, one batch every 200 steps",
                     decimals=3, h=260)
    cards, count = videos(out, ev[run])
    better = sum(1 for i in range(ev[run]["scenes"]) if sum(e["success"] for e in ev[run]["episodes"] if e["scene"] == i and e["policy"] == "stage3" and e["repeat"] > 0)
                 > sum(e["success"] for e in ev[run]["episodes"] if e["scene"] == i and e["policy"] == "frozen" and e["repeat"] > 0))

    page = TEMPLATE
    for k, v in {
        "BARS": bars, "TABLE": table, "LEARN": learn, "MODES": mode_rows, "FIXED": fixed_rows, "S1": s1, "S2A": s2a, "S2B": s2b, "S2C": s2c, "VIDEOS": cards,
        "FROZEN": f"{100 * s['frozen'][0]:.0f}", "S3": f"{100 * s['stage3'][0]:.0f}", "S3B": f"{100 * s2['stage3'][0]:.0f}", "CTL": f"{100 * sc['stage3'][0]:.0f}",
        "OP": f"{100 * s['operator'][0]:.0f}", "DIFF": pct(s["stage3 - frozen"], True), "DIFF2": pct(s2["stage3 - frozen"], True), "DIFFC": pct(sc["stage3 - frozen"], True),
        "RAW_F": f"{100 * raw['frozen'][0]:.0f}", "RAW_S": f"{100 * raw['stage3'][0]:.0f}", "BETTER": str(better), "SCENES": str(ev[run]["scenes"]), "RUNS": str(n_runs),
        "N_FIXED": str(count["fixed"]), "N_BOTH": str(count["both"]), "N_NEITHER": str(count["neither"]), "N_BROKEN": str(count["broken"]),
        "VS_TWO": pct(json.loads((RUNS / "stage3" / run / "fixed_heldout.json").read_text())["summary"]["success"]["stage3 - two_phase"], True),
        "M25": f"{mrad('headline/full', '__step25000'):.2f}", "M50": f"{mrad('headline/full'):.2f}", "N25": f"{mrad('headline/no_force', '__step25000'):.2f}",
        "N50": f"{mrad('headline/no_force'):.2f}", "CONT": f"{mrad('refine/continued'):.2f}", "REF": f"{mrad('refine/refined'):.2f}",
        "CRIT": f"{critic[-1]['heldout_abs_error']:.2f}", "CRIT0": f"{critic[0]['heldout_abs_error_constant']:.2f}", "CORR": f"{critic[-1]['heldout_correlation']:.2f}", "GH": GH,
    }.items():
        page = page.replace("{{" + k + "}}", v)
    assert "{{" not in page, page[page.index("{{"):][:40]
    (out / "index.html").write_text(page)
    print("wrote", out / "index.html", f"({len(page) // 1024} kB), {sum(count.values())} videos")


TEMPLATE = (REPO / "reports" / "stage3" / "page_template.html").read_text()

if __name__ == "__main__":
    main()
