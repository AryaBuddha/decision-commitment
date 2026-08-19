"""Phase 0 figure: m(alpha) per environment (fresh seeds) and the
m(n_cal) plateau. Usage:
    python experiments/wp2_phase0_battery/plot.py
Reads the two evidence-tier artifacts by their registered hashes.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
bat = json.loads((ROOT / "artifacts/wp2p0_ec15383b39b52206/results.json").read_text())
bud = json.loads((ROOT / "artifacts/wp2p0b_16996ec167284e46/results.json").read_text())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

COLORS = {"claims": "#1f77b4", "claims_logit": "#ff7f0e",
          "spike": "#2ca02c", "tickets": "#d62728"}
for env, col in COLORS.items():
    pts = [(a["alpha"], a["m"], a["m_se"]) for a in bat["analyses"]
           if a["env"] == env and a["m"] is not None]
    pts.sort()
    ax1.errorbar([p[0] for p in pts], [p[1] for p in pts],
                 yerr=[p[2] for p in pts], marker="o", capsize=3,
                 label=env, color=col)
ax1.axhline(1.0, color="gray", lw=0.8, ls=":")
ax1.set_xlabel("certified level alpha")
ax1.set_ylabel("amplification m (fresh seeds, n_cal = 1000)")
ax1.set_title("m(alpha) across four environments")
ax1.legend(fontsize=8)
ax1.invert_xaxis()

for alpha, col in [(0.05, "#9467bd"), (0.10, "#8c564b")]:
    pts = [(r["n_cal"], r["m"], r["m_se"]) for r in bud["m_grid"]
           if r["alpha"] == alpha and r["m"] is not None]
    pts.sort()
    ax2.errorbar([p[0] for p in pts], [p[1] for p in pts],
                 yerr=[p[2] for p in pts], marker="s", capsize=3,
                 label=f"alpha = {alpha}", color=col)
ax2.axhline(1.0, color="gray", lw=0.8, ls=":")
ax2.set_xscale("log")
ax2.set_xlabel("calibration budget n_cal (claims)")
ax2.set_ylabel("amplification m")
ax2.set_title("m(n_cal): steep fall, then a plateau above 1")
ax2.legend(fontsize=8)

fig.suptitle("WP2 Phase 0 (evidence tier; rung 2 exact rejection tilts; "
             "configs ec15383b39b52206, 16996ec167284e46)", fontsize=9)
fig.tight_layout()
out = ROOT / "docs" / "wp2_phase0_m.png"
fig.savefig(out, dpi=150)
print(f"[plot] {out}")
