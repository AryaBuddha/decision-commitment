"""The paper's central figure, five environments: excess marginal risk vs
signed aligned error, 290 evidence-tier cells (58 claims + 4 x 58 family).

Usage: python experiments/wp1_misspec_family/plot_all.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = {
    "claims": "artifacts/wp1mc_56704982681d6960",
    "tickets": "artifacts/wp1mf_tickets_85e921864acbedcc",
    "fraud": "artifacts/wp1mf_fraud_047610e36449b1c7",
    "moderation": "artifacts/wp1mf_moderation_037ea765a0016581",
    "compliance": "artifacts/wp1mf_compliance_c3609a222408ec0f",
}
COLOR = {"claims": "black", "tickets": "tab:red", "fraud": "tab:blue",
         "moderation": "tab:purple", "compliance": "tab:green"}
SLOPE = {"claims": 0.862, "tickets": 0.963, "fraud": 0.741,
         "moderation": 0.939, "compliance": 0.798}


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    n = 0
    for env, path in SOURCES.items():
        cells = json.loads((ROOT / path / "cells.json").read_text())
        n += len(cells)
        x = [c["aligned_mean"] for c in cells]
        y = [c["excess_mean"] for c in cells]
        ax.plot(x, y, "o", color=COLOR[env], markersize=4, alpha=0.75,
                label=f"{env}  (slope {SLOPE[env]:.2f})")
    xs = np.linspace(-0.012, 0.05, 2)
    ax.plot(xs, xs, ls="--", c="gray", lw=1.2, label="y = a  (kappa = 1)")
    ax.axhline(0, ls=":", c="gray", lw=0.8)
    ax.axvline(0, ls=":", c="gray", lw=0.8)
    ax.set_xlabel("signed aligned weight error,  "
                  r"$a = \mathrm{E}[(w - \hat{w})\,L(\lambda^{*})]$")
    ax.set_ylabel(r"excess marginal risk,  realized $-\ \alpha$")
    ax.set_title("One coordinate, five environments\n"
                 f"{n} evidence-tier cells; within-env collapse 5/5 "
                 "(Spearman 0.90 to 0.96, residual pass >= 98%);\n"
                 "through-origin slopes shown are superseded by the affine "
                 "form (Block A): see F17")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.text(0.99, 0.02,
            "rung 2 throughout; aligned error at the oracle threshold,\n"
            "per-sample normalisation of the exact ratio",
            transform=ax.transAxes, ha="right", fontsize=7, color="dimgray")
    fig.tight_layout()
    out = ROOT / "docs" / "wp1_all_env_collapse.png"
    fig.savefig(out, dpi=160)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
