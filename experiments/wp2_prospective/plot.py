"""The capstone figure: the registered envelope against realized risk on
the holdout environment. Usage:
    python experiments/wp2_prospective/plot.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
reg = json.loads((ROOT / "registrations/wp2_prospective.json").read_text())
res = json.loads((ROOT / "artifacts/wp2p5_7644a21fb970d7bd/results.json").read_text())
env = {(r["feature"], r["beta"]): r for r in reg["envelope"]["rows"]}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, feat, col in [(axes[0], "serial_rate", "#d62728"),
                      (axes[1], "desc_vagueness", "#1f77b4")]:
    cells = sorted([c for c in res["cells"] if c["feature"] == feat],
                   key=lambda c: c["beta"])
    betas = [c["beta"] for c in cells]
    ax.fill_between(betas,
                    [env[(feat, b)]["bound_p10"] for b in betas],
                    [env[(feat, b)]["bound_p90"] for b in betas],
                    alpha=0.15, color="gray",
                    label="envelope draw spread (p10-p90)")
    ax.plot(betas, [env[(feat, b)]["bound_median"] for b in betas],
            "k--", lw=1.5, label="registered certificate envelope")
    ax.errorbar(betas, [c["excess_est_mean"] for c in cells],
                yerr=[c["excess_est_se"] for c in cells], marker="o",
                color=col, capsize=3, label="realized excess (deployed)")
    ax.plot(betas, [c["excess_oracle_mean"] for c in cells], ":",
            color="gray", lw=1.2, label="oracle arm")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel(f"tilt strength beta ({feat})")
    ax.set_title(feat)
    ax.legend(fontsize=7)
axes[0].set_ylabel("excess marginal risk over alpha = 0.10")
fig.suptitle("Holdout environment (returns): certificate envelope registered before the "
             "sweep vs realized risk (rung 2 exact tilts; wp2env/wp2p5 7644a21fb970d7bd)",
             fontsize=9)
fig.tight_layout()
out = ROOT / "docs" / "wp2_prospective_envelope.png"
fig.savefig(out, dpi=150)
print(f"[plot] {out}")
