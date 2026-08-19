"""Evidence-tier collapse figure: excess marginal risk vs signed aligned
error on the claims environment, all cells pooled.

Usage: python experiments/wp1_misspec_claims/plot.py artifacts/wp1mc_<hash>
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOR = {"deprivation": "tab:red", "starvation": "tab:green",
         "inflation": "tab:blue", "mismatch": "tab:purple",
         "temper": "black", "directional": "tab:orange"}
BETA_MARKER = {3.0: "o", 5.0: "s"}


def main(run_dir: str) -> None:
    d = pathlib.Path(run_dir)
    cells = json.loads((d / "cells.json").read_text())
    ana = json.loads((d / "collapse.json").read_text())
    cfg = json.loads((d / "config.json").read_text())
    z = cfg["z_one_sided"]

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    seen = set()
    for c in cells:
        col = COLOR[c["name"]]
        m = BETA_MARKER[c["beta"]]
        lab = None
        if c["name"] not in seen:
            seen.add(c["name"])
            lab = c["name"] + (" (synthetic)" if c["group"] == "family" else "")
        ax.errorbar(c["aligned_mean"], c["excess_mean"],
                    xerr=z * c["aligned_se"], yerr=z * c["risk_se"],
                    fmt=m, color=col, markersize=5.5, capsize=2, lw=0.9,
                    label=lab, alpha=0.9)

    fit = sorted(ana["isotonic_fit"])
    ax.plot([p[0] for p in fit], [p[1] for p in fit], c="dimgray", lw=1.6,
            zorder=1, label="pooled isotonic fit")
    ax.axhline(0.0, ls=":", c="gray", lw=1)
    ax.axvline(0.0, ls=":", c="gray", lw=1)

    v = "COLLAPSE" if ana["collapse"] else "NO COLLAPSE"
    ax.set_xlabel("signed aligned weight error,  "
                  r"$a = \mathrm{E}[(w - \hat{w})\,L(\lambda^{*})]$")
    ax.set_ylabel(r"excess marginal risk,  realized $-\ \alpha$")
    ax.set_title("EVIDENCE TIER: the collapse on a real environment\n"
                 f"claims env; {ana['n_cells']} cells; "
                 f"Spearman {ana['spearman']:.3f}, residual pass "
                 f"{ana['residual_pass_fraction']:.0%}, origin slope "
                 f"{ana['origin_slope']:.2f}: {v}")
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="gray", marker="o", ls="", label=r"$\beta$ = 3"),
                Line2D([], [], color="gray", marker="s", ls="", label=r"$\beta$ = 5")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.text(0.99, 0.02,
            "rung 2 throughout: exact rejection tilt on the blind feature;\n"
            "chi2 labels MC-estimated; aligned error uses per-sample "
            "normalisation of the exact ratio",
            transform=ax.transAxes, ha="right", fontsize=7, color="dimgray")

    fig.tight_layout()
    out = d / "wp1_claims_collapse.png"
    fig.savefig(out, dpi=160)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main(sys.argv[1])
