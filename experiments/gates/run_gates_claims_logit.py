"""Gates 1-5 for the claims-logit variant (Block B3). Re-gated from
scratch; G3 and G5 behaving differently from the tree family is part of
the result. Run: python experiments/gates/run_gates_claims_logit.py"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus.envs import claims                       # noqa: E402
from cus.envs.claims import generate, gold        # noqa: E402
from cus.envs.claims_logit import ClaimsLogitEnv  # noqa: E402

GATE_CONFIG = {
    "experiment": "env_claims_logit_gates",
    "env_seed": 20260818, "n_demo": 6000, "expert_noise": 0.06,
    "induction_family": "logistic scorer, s = max(p, 1-p), dec = p >= 0.5",
    "tilt_features": ["inconsistency", "severity"],
    "primary_tilt": "inconsistency",
    "beta_grid": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "g1_n": 50000, "g1_min_solvable": 0.98,
    "g2_n": 200000, "g2_n_sigma": 5.0,
    "g3_n": 60000, "g3_min_auc_lift": 0.02,
    "g3_audit": "decision-conditional",
    "g4_n": 40000, "g4_min_region_share": 0.05,
    "g5_n_lambda": 400, "g5_min_cells": 20,
    "seed": 20260819,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def main():
    h = config_hash(GATE_CONFIG)
    freeze = json.loads((ROOT / "registrations" / "env_claims_logit.json").read_text())
    if freeze.get("gate_config_hash") != h:
        raise SystemExit(f"Gate config hash {h} != frozen {freeze.get('gate_config_hash')}")
    print(f"[freeze] claims_logit: gate config {h} matches")
    cfg = GATE_CONFIG
    env = ClaimsLogitEnv.induce(cfg["env_seed"], cfg["n_demo"], cfg["expert_noise"])
    rng = np.random.default_rng([cfg["seed"], 77])
    report = {"gate_config": cfg}

    X, u = generate(np.random.default_rng([cfg["seed"], 78]), cfg["g1_n"])
    g1 = float(np.mean(gold(X, u) == gold(X, u)))
    report["G1"] = {"deterministic": g1, "pass": bool(g1 >= 0.98)}
    print(f"[G1] {g1:.4f} -> {'PASS' if g1 >= 0.98 else 'FAIL'}")

    bmax = max(cfg["beta_grid"])
    g2_all = True
    for feat in cfg["tilt_features"]:
        j = claims.BOUNDED[feat]
        Xs, _ = generate(rng, cfg["g2_n"])
        w = np.exp(bmax * Xs[:, j]); wn = w / w.mean()
        Xr, _ = env.draw_instances(rng, cfg["g2_n"], bmax, feat)
        ok = True
        for gs, gr in ((Xs[:, j], Xr[:, j]), (Xs[:, j]**2, Xr[:, j]**2)):
            src = wn * gs
            tol = cfg["g2_n_sigma"] * float(np.hypot(src.std()/np.sqrt(len(src)),
                                                     gr.std()/np.sqrt(len(gr))))
            ok &= abs(float(src.mean()) - float(gr.mean())) <= tol
        g2_all &= ok
        print(f"[G2] {feat}: {'PASS' if ok else 'FAIL'}")
    report["G2"] = {"pass": bool(g2_all)}

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ct = env.case_table(rng, cfg["g3_n"])
    dec, _ = env.route(ct.X)
    half = cfg["g3_n"] // 2
    tr, te = slice(0, half), slice(half, None)
    def auc(cols):
        Z = np.column_stack(cols)
        clf = LogisticRegression(max_iter=2000).fit(Z[tr], ct.wrong[tr])
        return float(roc_auc_score(ct.wrong[te], clf.predict_proba(Z[te])[:, 1]))
    base = auc([ct.s, dec])
    lifts = {}
    for feat in cfg["tilt_features"]:
        j = claims.BOUNDED[feat]
        lifts[feat] = auc([ct.s, dec, ct.X[:, j], dec * ct.X[:, j]]) - base
        print(f"[G3] {feat}: lift {lifts[feat]:+.4f}")
    g3 = max(lifts.values()) >= cfg["g3_min_auc_lift"]
    report["G3"] = {"lifts": lifts, "pass": bool(g3)}
    print(f"[G3] -> {'PASS' if g3 else 'FAIL'}")

    g4_all = True
    shares = []
    for b in cfg["beta_grid"]:
        ctb = env.case_table(rng, cfg["g4_n"], beta=b, feature=cfg["primary_tilt"])
        share = float(ctb.region.mean())
        shares.append(share)
        g4_all &= share >= cfg["g4_min_region_share"]
    report["G4"] = {"shares": shares, "pass": bool(g4_all)}
    print(f"[G4] {shares[0]:.3f} to {shares[-1]:.3f} -> {'PASS' if g4_all else 'FAIL'}")

    lams = np.linspace(0, 1, cfg["g5_n_lambda"])
    ct5 = env.case_table(rng, 50000)
    cells = len(np.unique(np.searchsorted(lams, np.unique(np.round(ct5.s, 8)))))
    g5 = cells >= cfg["g5_min_cells"]
    report["G5"] = {"distinct_grid_cells": cells, "pass": bool(g5)}
    print(f"[G5] {cells} cells -> {'PASS' if g5 else 'FAIL'}")

    report["all_pass"] = bool(report["G1"]["pass"] and g2_all and g3 and g4_all and g5)
    print(f"[gates] claims_logit: {'ALL PASS' if report['all_pass'] else 'FAILED'}")
    out = ROOT / "artifacts" / f"gates_claims_logit_{h}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
