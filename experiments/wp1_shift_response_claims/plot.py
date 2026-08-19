"""Shift-response figure for the claims environment (evidence tier).

Usage: python experiments/wp1_shift_response_claims/plot.py artifacts/wp1c_<hash>
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ["unweighted", "oracle", "estimated", "glob_oracle"]
STYLE = {"unweighted": ("tab:red", "o", "CRC, no weights"),
         "oracle": ("tab:blue", "s", "Prop. 2, oracle ratio"),
         "estimated": ("tab:green", "^", "Prop. 2, estimated ratio"),
         "glob_oracle": ("tab:gray", "v", "global shortcut, oracle ratio")}


def main(run_dir: str) -> None:
    d = pathlib.Path(run_dir)
    rows = json.loads((d / "results.json").read_text())
    cfg = json.loads((d / "config.json").read_text())
    alpha = cfg["alpha"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        x = [v["chi2_mc"] for v in r]
        c, m, lab = STYLE[arm]
        ax[0].plot(x, [v["marginal_risk_mean"] for v in r], marker=m, color=c, label=lab)
        ax[0].fill_between(x, [v["marginal_risk_p05"] for v in r],
                           [v["marginal_risk_p95"] for v in r], color=c, alpha=0.10)
        ax[1].plot(x, [v["deferral_rate_mean"] for v in r], marker=m, color=c, label=lab)
        ax[2].plot(x, [v["ess_mean"] for v in r], marker=m, color=c, label=lab)

    ax[0].axhline(alpha, ls="--", c="k", lw=1)
    ax[0].text(0.02, alpha * 1.03, f"certified $\\alpha$ = {alpha}", fontsize=9,
               transform=ax[0].get_yaxis_transform())
    ax[0].set_ylabel("realized marginal commit-error risk")
    ax[0].set_title("Realized vs certified risk (claims env)")
    ax[1].set_ylabel("deferral rate")
    ax[1].set_title("Cost of the guarantee")
    ax[2].set_ylabel("effective sample size (of 1000)")
    ax[2].set_title("ESS under the bounded tilt")

    for a in ax:
        a.set_xlabel(r"shift magnitude, $\chi^2$ (MC estimate, rung 2)")
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    fig.suptitle("EVIDENCE TIER: gated claims environment, 500 trials/level, "
                 "exact rejection tilt on the evidence-blind feature",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    out = d / "wp1_claims_shift_response.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main(sys.argv[1])
