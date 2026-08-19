"""Block B2: score-granularity sweep on claims. EVIDENCE TIER.

Claims rules re-induced at (depth 4, min_leaf 80) and (depth 10,
min_leaf 20) beside the archived (7, 40); temper-family collapse per
variant in the Block A paired-difference form. The coarse variant FAILS
gate G5 (13 distinct scores, floor 20; audit disclosed in the
registration) and runs as an explicitly labelled OUT-OF-GATE boundary
probe whose cells may not enter any within-gate claim.

Run:   python experiments/wp1_granularity_claims/run.py
Smoke: add --smoke 10
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import crc, tests                   # noqa: E402
from cus.envs.claims import ClaimsEnv        # noqa: E402


CONFIG = {
    "experiment": "wp1_granularity_claims",
    "alpha": 0.10,
    "variants": {"d4l80": [4, 80], "d10l20": [10, 20]},
    "baseline_archived": "d7l40: slope 1.249 (wp1mc via blockA), kappa_pred 1.058, oracle excess -0.0039",
    "n_cal": 1000, "n_eval": 1000, "n_trials": 300, "n_lambda": 400,
    "tilt_feature": "inconsistency",
    "temper_betas": [3.0, 5.0],
    "temper_gammas": [0.0, 0.25, 0.5, 0.75, 0.8, 1.0, 1.25, 1.5],
    "z_one_sided": 1.645,
    "seed": 20260819,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp1_granularity_claims.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration()
    n = smoke if smoke is not None else CONFIG["n_trials"]
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    alpha = CONFIG["alpha"]
    feat = CONFIG["tilt_feature"]
    all_rows = []
    for vi, (vname, (depth, leaf)) in enumerate(CONFIG["variants"].items()):
        env = ClaimsEnv.induce(max_depth=depth, min_samples_leaf=leaf)
        cellcount = len(np.unique(np.searchsorted(
            lambdas, sorted(set(env.leaf_score.values())))))
        print(f"[wp1g] {vname}: rules={len(env.leaf_score)} score_cells={cellcount}"
              f"{' OUT-OF-GATE (G5 fail)' if cellcount < 20 else ''}")
        cells = []
        for bi, beta in enumerate(CONFIG["temper_betas"]):
            for gi, gamma in enumerate(CONFIG["temper_gammas"]):
                rng = np.random.default_rng([CONFIG["seed"], 300 + vi, gi, bi])
                aligned, paired, ro_ = [], [], []
                for _ in range(n):
                    cal = env.case_table(rng, CONFIG["n_cal"])
                    ev = env.case_table(rng, CONFIG["n_eval"], beta=beta,
                                        feature=feat)
                    w = np.exp(env.tilt_logweight(cal.X, beta, feat))
                    wev = np.exp(env.tilt_logweight(ev.X, beta, feat))
                    what, wevhat = w ** gamma, wev ** gamma
                    losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
                    lam_o = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
                    ls = float(np.mean(lam_o))
                    ro = float(((ev.s >= np.asarray(lam_o)) & ev.wrong).mean())
                    lam_e = crc.lhat_prop2(losses, lambdas, alpha, what, wevhat)
                    re = float(((ev.s >= np.asarray(lam_e)) & ev.wrong).mean())
                    wn = w / w.mean(); hn = what / what.mean()
                    L = ((cal.s >= ls) & cal.wrong).astype(float)
                    aligned.append(float(np.mean((wn - hn) * L)))
                    paired.append(re - ro)
                    ro_.append(ro)
                cells.append({"variant": vname, "gamma": gamma, "beta": beta,
                              "aligned_mean": float(np.mean(aligned)),
                              "paired_diff_mean": float(np.mean(paired)),
                              "paired_diff_se": float(np.std(paired, ddof=1)
                                                      / np.sqrt(len(paired))),
                              "oracle_excess": float(np.mean(ro_) - alpha)})
        all_rows.extend(cells)
        x = np.array([c["aligned_mean"] for c in cells])
        y = np.array([c["paired_diff_mean"] for c in cells])
        m = np.abs(x) > 1e-12
        slope = float(x[m] @ y[m] / (x[m] @ x[m]))
        r = y[m] - slope * x[m]
        sse = float(np.sqrt((r @ r) / (m.sum() - 1) / (x[m] @ x[m])))
        oexc = float(np.mean([c["oracle_excess"] for c in cells]))
        print(f"[wp1g] {vname}: pd_slope={slope:.3f}±{sse:.3f} "
              f"oracle_excess={oexc:+.4f}")
        all_rows.append({"variant": vname, "part": "summary",
                         "pd_slope": slope, "pd_slope_se": sse,
                         "score_cells": cellcount,
                         "oracle_excess_mean": oexc})
    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp1g_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
