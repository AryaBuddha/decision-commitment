"""The practitioner figure: APE forecast vs realized unweighted risk,
28 confirmatory level-cells across four environments built AFTER the
forecast recipe was registered, plus the claims top level (post-hoc,
descriptive, open marker).

Usage: python experiments/wp1_shift_response_family/plot_ape.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
APE = {
    "tickets": [-0.0032, 0.0085, 0.0216, 0.0370, 0.0534, 0.0704, 0.0887],
    "fraud": [-0.0009, 0.0020, 0.0037, 0.0046, 0.0035, -0.0003, -0.0051],
    "moderation": [0.0000, 0.0088, 0.0153, 0.0200, 0.0226, 0.0211, 0.0161],
    "compliance": [-0.0073, -0.0012, 0.0059, 0.0136, 0.0210, 0.0292, 0.0366],
}
HASHES = {"tickets": "d8b48cff3ab2e336", "fraud": "2f83073cb88bea1e",
          "moderation": "bc612e3b59311f78", "compliance": "99e91888f277db19"}
COLOR = {"tickets": "tab:red", "fraud": "tab:blue",
         "moderation": "tab:purple", "compliance": "tab:green"}
ALPHA = 0.10


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for env, h in HASHES.items():
        rows = json.load(open(ROOT / f"artifacts/wp1f_{env}_{h}/results.json"))
        unw = [r for r in rows if r["arm"] == "unweighted"]
        x = [ALPHA + a for a in APE[env]]
        y = [r["marginal_risk_mean"] for r in unw]
        ye = [1.645 * r["marginal_risk_se"] for r in unw]
        ax.errorbar(x, y, yerr=ye, fmt="o", color=COLOR[env], markersize=5,
                    capsize=2, lw=1, label=env)
    # claims top level: post-hoc, descriptive
    ax.plot([ALPHA + 0.0249], [ALPHA + 0.0231], marker="s", mfc="none",
            mec="black", ls="", markersize=7,
            label="claims (APE post-hoc, descriptive)")

    lo, hi = 0.085, 0.20
    ax.plot([lo, hi], [lo, hi], ls="--", c="k", lw=1.2, label="forecast = realized")
    ax.fill_between([lo, hi], [lo - 0.0075, hi - 0.0075],
                    [lo + 0.0075, hi + 0.0075], color="gray", alpha=0.15,
                    label="registered tolerance ±0.0075")
    ax.axhline(ALPHA, ls=":", c="gray", lw=1)
    ax.axvline(ALPHA, ls=":", c="gray", lw=1)
    ax.set_xlabel("forecast risk before any labelled target data:  α + APE(level)")
    ax.set_ylabel("realized unweighted CRC risk (500 trials/level)")
    ax.set_title("A number you can compute before trusting selective commitment\n"
                 "28 preregistered level-cells, 4 environments built after the "
                 "forecast recipe;\nmax |realized − forecast| = 0.0063; "
                 "rank ordering exact")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    fig.tight_layout()
    out = ROOT / "docs" / "wp1_ape_forecast.png"
    fig.savefig(out, dpi=160)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
