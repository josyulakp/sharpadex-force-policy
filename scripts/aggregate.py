"""Results table for an experiment group, with intervals over episodes and seeds.

    python scripts/aggregate.py <group> [tag]      -> reports/results/<group>[__tag].md / .json
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, load_config  # noqa: E402
from sharpa.metrics import GROUPS, bootstrap, load_variant  # noqa: E402

METRICS = [("act", "Action error, all joints (mrad)", 1000), ("act_arm", "Action error, arms (mrad)", 1000),
           ("act_hand", "Action error, hands (mrad)", 1000), ("force", "Force error (N)", 1),
           ("force_wrench_expert", "Wrench expert: fingertip force error of its joint proposal (N)", 1),
           ("act_wrench_expert", "Wrench expert: action error of its own proposal, before mixing (mrad)", 1000),
           # world-model variants only: the same force forecast conditioned on other actions
           ("force_recorded", "World model: force error given the recorded actions (N)", 1),
           ("force_zero", "World model: force error given zero actions (N)", 1),
           ("force_shuffled", "World model: force error given the actions of other frames (N)", 1)]


def main():
    group = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "heldout"
    spec = yaml.safe_load((REPO_ROOT / "configs" / "experiments.yaml").read_text())[group]
    ev = load_config()["eval"]
    variants = {v: load_variant(REPO_ROOT / "runs" / group / v, tag) for v in spec["variants"]}
    variants = {v: s for v, s in variants.items() if s}
    ref = spec.get("reference")
    first = next(iter(variants.values()))[0]
    c = first["contact"] > 0.5
    L = [f"# {group} ({tag})", "",
         f"Held-out: {len(first['ep_uid'])} episodes, {len(c)} frames ({c.sum()} contact, {(~c).sum()} free). "
         f"Seeds per variant: {', '.join(f'{v} {len(s)}' for v, s in variants.items())}. "
         f"Values are mean [95 % interval] from a bootstrap over held-out episodes and training seeds; "
         f"Δ is the paired difference to `{ref}` (negative = better than `{ref}`)."]
    results = {}
    for key, title, scale in METRICS:
        use = {v: s for v, s in variants.items() if key in s[0] and (key != "force" or _predicts_force(group, v))}
        if not use:
            continue
        res = bootstrap(use, key, ref if ref in use else None, ev["bootstrap_samples"], ev["ci"])
        results[key] = res
        L += ["", f"## {title}", "", "| variant | " + " | ".join(GROUPS) + " | Δ contact | Δ free |", "|---|---|---|---|---|---|"]
        for v, r in res.items():
            cells = [f"{r[g]['mean'] * scale:.2f} [{r[g]['lo'] * scale:.2f}, {r[g]['hi'] * scale:.2f}]" for g in GROUPS]
            diffs = [f"{r[g]['diff'] * scale:+.2f} [{r[g]['diff_lo'] * scale:+.2f}, {r[g]['diff_hi'] * scale:+.2f}]"
                     if "diff" in r[g] else "–" for g in ("contact", "free")]
            L.append(f"| {v} | " + " | ".join(cells + diffs) + " |")
        base = {"act": "act_hold", "force": "force_persist"}.get(key)
        if base:
            b = first[base]
            name = "hold current joint state" if key == "act" else "force stays as it is now"
            L.append(f"| *baseline: {name}* | {b.mean() * scale:.2f} | {b[c].mean() * scale:.2f} | {b[~c].mean() * scale:.2f} | – | – |")
    L += ["", "## Decision slots and expert gate (mean over seeds)", "",
          "| variant | contact-soon accuracy | phase accuracy | force-level error (levels) | gate weight of expert 0: contact / free |",
          "|---|---|---|---|---|"]
    for v in variants:
        js = [json.loads(p.read_text()) for p in sorted((REPO_ROOT / "runs" / group / v).glob(f"seed*/eval_{tag}.json"))]
        g = np.mean([j["gate0_contact_vs_free"] for j in js], 0)
        sup = js[0]["decision_supervision"]
        L.append(f"| {v} | {np.mean([j['contact_soon_acc'] for j in js]):.3f}{'' if sup else ' (untrained)'} | "
                 f"{np.mean([j['phase_acc'] for j in js]):.3f} | {np.mean([j['level_mae'] for j in js]):.3f}"
                 f"{'' if sup else ' (untrained)'} | {g[0]:.2f} / {g[1]:.2f} |")
    out = REPO_ROOT / "reports" / "results"
    out.mkdir(parents=True, exist_ok=True)
    name = group if tag == "heldout" else f"{group}__{tag}"
    (out / f"{name}.md").write_text("\n".join(L) + "\n")
    (out / f"{name}.json").write_text(json.dumps(results, indent=1))
    print("\n".join(L))


def _predicts_force(group: str, variant: str) -> bool:
    j = next((REPO_ROOT / "runs" / group / variant).glob("seed*/eval_heldout.json"))
    return json.loads(j.read_text())["force_output"]


if __name__ == "__main__":
    main()
