"""Stressed-capstone figure: registered envelope vs realized excess for
the twelve degraded deployments. Usage:
    python experiments/wp2_prospective_stress/plot.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
res = json.loads((ROOT / "artifacts/wp2p5s_8e357241960a63b7/results.json").read_text())
cells = res["cells"]

labels = [f"{c['kind'].replace('_', ' ')}  {c['feature'].split('_')[0]} b={c['beta']:g}"
          for c in cells]
y = np.arange(len(cells))[::-1]
bounds = [c["bound_median_registered"] for c in cells]
exc = [c["excess_est_mean"] for c in cells]
ese = [c["excess_est_se"] for c in cells]

fig, ax = plt.subplots(figsize=(9, 5.6))
ax.barh(y, bounds, height=0.62, color="#d9d9d9",
        label="registered certificate envelope")
ax.errorbar(exc, y, xerr=[2 * s for s in ese], fmt="o", color="#d62728",
            capsize=3, label="realized excess (300 trials, 2 SE)")
ax.axvline(0.0, color="gray", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("excess marginal risk over alpha = 0.10")
ax.set_title("Coverage under stress: twelve degraded deployments on the holdout\n"
             "(envelope registered before the sweep; wp2envs/wp2p5s "
             "8e357241960a63b7; rung 2 exact tilts)", fontsize=9)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
out = ROOT / "docs" / "wp2_prospective_stress.png"
fig.savefig(out, dpi=150)
print(f"[plot] {out}")
