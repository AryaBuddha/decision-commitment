"""WP1 misspecification deliverable: excess marginal risk vs L1(P0) weight
error, pooled across degradation axes. The WP2 target figure.

Usage: python experiments/wp1_misspecification/plot.py artifacts/wp1m_<hash>
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AXIS_COLOR = {"deprivation": "tab:red", "inflation": "tab:blue",
              "starvation": "tab:green", "mismatch": "tab:purple"}
BETA_MARKER = {0.75: "o", 1.25: "s"}


def main(run_dir: str) -> None:
    d = pathlib.Path(run_dir)
    rows = json.loads((d / "results.json").read_text())
    cfg = json.loads((d / "config.json").read_text())
    est = [r for r in rows if r["arm"] == "estimated"]
    z = cfg["z_one_sided"]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    seen = set()
    for r in est:
        c = AXIS_COLOR[r["axis"]]
        m = BETA_MARKER[r["beta_scale"]]
        lab = None
        if r["axis"] not in seen:
            seen.add(r["axis"])
            lab = r["axis"]
        ax.errorbar(r["ratio_w_l1_mean"], r["excess_marginal_risk_mean"],
                    yerr=z * r["marginal_risk_se"], fmt=m, color=c,
                    markersize=6, capsize=2, lw=1, label=lab)

    xs = [0, max(r["ratio_w_l1_mean"] for r in est) * 1.05]
    ax.plot(xs, xs, ls="--", c="k", lw=1.2, label="y = B·L1  (B = 1)")
    ax.axhline(0.0, ls=":", c="gray", lw=1)

    ax.set_xlabel("normalized-weight L1(P₀) estimator error,  E|ŵ − w|")
    ax.set_ylabel("excess marginal risk,  realized − α")
    ax.set_title("Estimated-ratio Prop. 2 under controlled estimator degradation\n"
                 "placeholder pilot; rung 1 (linear axes), rung 2 (mismatch)"
                 )
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], color="gray", marker="o", ls="", label="β = 0.75"),
                Line2D([], [], color="gray", marker="s", ls="", label="β = 1.25")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.text(0.99, 0.02,
            "mismatch cells face smaller shift (chi2 = 0.21/0.52) than the\n"
            "linear axes (0.76/3.77); cross-axis comparisons are not matched-shift",
            transform=ax.transAxes, ha="right", fontsize=7, color="dimgray")

    fig.tight_layout()
    out = d / "wp1_misspec_l1_vs_excess.png"
    fig.savefig(out, dpi=160)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main(sys.argv[1])
