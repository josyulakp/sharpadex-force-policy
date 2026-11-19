"""Fill reports/rollouts/page_template.html with the rollout traces and result tables -> reports/rollouts/index.html

    python scripts/rollout.py && python scripts/aggregate.py eraser_force && python scripts/build_rollout_page.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT, load_config  # noqa: E402

r = load_config()["rollout"]
out, res = REPO_ROOT / r["out_dir"], REPO_ROOT / "reports" / "results"
page = (out / "page_template.html").read_text()
for key, path in (("ROLLOUTS", out / "rollouts.json"), ("RESULTS", res / f"{r['group']}.json"),
                  ("ZERO", res / f"{r['group']}__heldout__zero_force-True.json")):
    page = page.replace(f"/*{key}*/null", path.read_text())
(out / "index.html").write_text(page)
print(out / "index.html", f"{len(page) / 1e6:.2f} MB")
