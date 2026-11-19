"""Fill reports/architecture/page_template.html with measurements -> reports/architecture/index.html

    python scripts/activations.py runs/eraser_force/full && python scripts/activations.py runs/eraser_force/no_force
    python scripts/build_architecture_page.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT  # noqa: E402

out, res = REPO_ROOT / "reports" / "architecture", REPO_ROOT / "reports" / "results"
page = (out / "page_template.html").read_text()
for key, path in (("ACT_FULL", out / "activations_full.json"), ("ACT_NOFORCE", out / "activations_no_force.json"),
                  ("RESULTS", res / "eraser_force.json"), ("ZERO", res / "eraser_force__heldout__zero_force-True.json")):
    page = page.replace(f"/*{key}*/null", path.read_text())
(out / "index.html").write_text(page)
print(out / "index.html", f"{len(page) / 1e3:.0f} kB")
