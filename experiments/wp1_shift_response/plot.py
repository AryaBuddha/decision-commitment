"""Render the WP1 shift-response figure from a saved results.json."""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ["unweighted", "oracle", "estimated"]
STYLE = {"unweighted": ("tab:red", "o", "CRC, no weights"),
         "oracle": ("tab:blue", "s", "weighted CRC, oracle ratio"),
         "estimated": ("tab:green", "^", "weighted CRC, estimated ratio")}


def main(run_dir: str) -> None:
    d = pathlib.Path(run_dir)
    rows = json.loads((d / "results.json").read_text())
    cfg = json.loads((d / "config.json").read_text())
    alpha = cfg["alpha"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        x = [v["chi2_divergence"] for v in r]
        c, m, lab = STYLE[arm]

        ax[0].plot(x, [v["marginal_risk_mean"] for v in r], marker=m, color=c, label=lab)
        ax[0].fill_between(x, [v["marginal_risk_p05"] for v in r],
                           [v["marginal_risk_p95"] for v in r], color=c, alpha=0.12)
        ax[1].plot(x, [v["deferral_rate_mean"] for v in r], marker=m, color=c, label=lab)
        ax[2].plot(x, [v.get("marginal_risk_region1_mean") for v in r],
                   marker=m, color=c, label=lab)

    ax[0].axhline(alpha, ls="--", c="k", lw=1)
    ax[0].text(0.02, alpha * 1.06, f"certified $\\alpha$ = {alpha}", fontsize=9,
               transform=ax[0].get_yaxis_transform())
    ax[0].set_ylabel("realized marginal commit-error risk")
    ax[0].set_title("Realized vs certified risk")

    ax[1].set_ylabel("deferral rate")
    ax[1].set_title("Cost of the guarantee")

    ax[2].axhline(alpha, ls="--", c="k", lw=1)
    ax[2].set_ylabel("region 1 marginal risk")
    ax[2].set_title("Minority region (aggregate control does not transfer)")

    for a in ax:
        a.set_xlabel(r"shift magnitude,  $\chi^2(\tilde{P}_X \| P_X)$")
        a.grid(alpha=0.25)
        a.legend(fontsize=8)

    fig.tight_layout()
    out = d / "wp1_shift_response.png"
    fig.savefig(out, dpi=160)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main(sys.argv[1])
