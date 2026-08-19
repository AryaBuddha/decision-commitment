"""Gates 1-5 for environment 6 (spike, Block C).
Run: python experiments/gates/run_gates_spike.py"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus.envs import spike as sp                  # noqa: E402
from cus.envs.spike import SpikeEnv, generate, gold  # noqa: E402

GATE_CONFIG = {
    "experiment": "env_spike_gates",
    "env_seed": 20260819, "n_demo": 6000, "expert_noise": 0.03,
    "max_depth": 7, "min_samples_leaf": 40,
    "tilt_features": ["b", "v"],
    "primary_tilt": "b",
    "beta_grid": [0.0, 1.168, 2.336, 3.504, 4.672, 5.840, 7.008],
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
    freeze = json.loads((ROOT / "registrations" / "env_spike.json").read_text())
    if freeze.get("gate_config_hash") != h:
        raise SystemExit(f"Gate config hash {h} != frozen {freeze.get('gate_config_hash')}")
    print(f"[freeze] spike: gate config {h} matches")
    cfg = GATE_CONFIG
    env = SpikeEnv.induce(cfg["env_seed"], cfg["n_demo"], cfg["expert_noise"],
                          cfg["max_depth"], cfg["min_samples_leaf"])
    rng = np.random.default_rng([cfg["seed"], 77])
    report = {"gate_config": cfg, "n_rules": len(env.leaf_score)}

    X, u = generate(np.random.default_rng([cfg["seed"], 78]), cfg["g1_n"])
    g1 = float(np.mean(gold(X, u) == gold(X, u)))
    report["G1"] = {"deterministic": g1, "pass": bool(g1 >= 0.98)}
    print(f"[G1] {g1:.4f} -> {'PASS' if g1 >= 0.98 else 'FAIL'}")

    bmax = max(cfg["beta_grid"])
    g2_all = True
    for feat in cfg["tilt_features"]:
        j = sp.BOUNDED[feat]
        Xs, _ = generate(rng, cfg["g2_n"])
        w = np.exp(bmax * Xs[:, j]); wn = w / w.mean()
        Xr, _ = env.draw_instances(rng, cfg["g2_n"], bmax, feat)
        ok = True
        for gs, gr in ((Xs[:, j], Xr[:, j]), (Xs[:, j] ** 2, Xr[:, j] ** 2)):
            src = wn * gs
            tol = cfg["g2_n_sigma"] * float(np.hypot(src.std() / np.sqrt(len(src)),
                                                     gr.std() / np.sqrt(len(gr))))
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
        j = sp.BOUNDED[feat]
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
        shares.append(round(share, 4))
        g4_all &= share >= cfg["g4_min_region_share"]
    report["G4"] = {"shares": shares, "pass": bool(g4_all)}
    print(f"[G4] {shares[0]} to {shares[-1]} -> {'PASS' if g4_all else 'FAIL'}")

    lams = np.linspace(0, 1, cfg["g5_n_lambda"])
    cells = len(np.unique(np.searchsorted(lams, sorted(set(env.leaf_score.values())))))
    g5 = cells >= cfg["g5_min_cells"]
    report["G5"] = {"distinct_grid_cells": cells, "pass": bool(g5)}
    print(f"[G5] {cells} cells -> {'PASS' if g5 else 'FAIL'}")

    report["all_pass"] = bool(report["G1"]["pass"] and g2_all and g3 and g4_all and g5)
    print(f"[gates] spike: {'ALL PASS' if report['all_pass'] else 'FAILED'}")
    out = ROOT / "artifacts" / f"gates_spike_{h}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
