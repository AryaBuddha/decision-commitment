"""WP1 aligned-error deliverable: excess marginal risk vs signed aligned
weight error, all cells pooled (misspec recompute + synthetic families),
with the pooled isotonic fit and the collapse verdict.

Usage: python experiments/wp1_aligned_error/plot.py artifacts/wp1ae_<hash>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AXIS_COLOR = {"deprivation": "tab:red", "inflation": "tab:blue",
              "starvation": "tab:green", "mismatch": "tab:purple",
              "temper": "black", "directional": "tab:orange"}
BETA_MARKER = {0.75: "o", 1.25: "s"}


def main(run_dir: str) -> None:
    d = pathlib.Path(run_dir)
    cells = json.loads((d / "cells.json").read_text())
    ana = json.loads((d / "collapse.json").read_text())
    cfg = json.loads((d / "config.json").read_text())
    z = cfg["z_one_sided"]

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    seen = set()
    for c in cells:
        group = c.get("axis", c.get("family"))
        col = AXIS_COLOR[group]
        m = BETA_MARKER[c["beta_scale"]]
        lab = None
        if group not in seen:
            seen.add(group)
            lab = group + (" (synthetic)" if c["source"] == "synthetic" else "")
        ax.errorbar(c["aligned_mean"], c["excess_mean"],
                    xerr=z * c["aligned_se"], yerr=z * c["risk_se"],
                    fmt=m, color=col, markersize=5.5, capsize=2, lw=0.9,
                    label=lab, alpha=0.9)

    fit = sorted(ana["isotonic_fit"])
    ax.plot([p[0] for p in fit], [p[1] for p in fit],
            c="dimgray", lw=1.6, zorder=1,
            label="pooled isotonic fit")
    ax.axhline(0.0, ls=":", c="gray", lw=1)
    ax.axvline(0.0, ls=":", c="gray", lw=1)

    v = "COLLAPSE" if ana["collapse"] else "NO COLLAPSE"
    ax.set_xlabel("signed aligned weight error,  "
                  r"$a = \mathrm{E}[(w - \hat{w})\,L(\lambda^{*})]$")
    ax.set_ylabel(r"excess marginal risk,  realized $-\ \alpha$")
    ax.set_title("One coordinate for estimated-shift risk control?\n"
                 f"placeholder pilot; {ana['n_cells']} cells; "
                 f"Spearman {ana['spearman']:.3f}, "
                 f"residual pass {ana['residual_pass_fraction']:.0%}: {v}")
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="gray", marker="o", ls="", label=r"$\beta$ = 0.75"),
                Line2D([], [], color="gray", marker="s", ls="", label=r"$\beta$ = 1.25")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.text(0.99, 0.02,
            "misspec cells: rung 1 (linear axes) and rung 2 (mismatch);\n"
            "synthetic cells: rung 1; aligned error at the oracle threshold",
            transform=ax.transAxes, ha="right", fontsize=7, color="dimgray")

    fig.tight_layout()
    out = d / "wp1_aligned_error_collapse.png"
    fig.savefig(out, dpi=160)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main(sys.argv[1])
