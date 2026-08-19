"""Validation gates 1-5 for the environment family (tickets, fraud,
moderation, compliance). Same five gates as claims, decision-conditional
G3 form (env_claims amendment 2), enforced against the per-environment
freeze document registrations/env_<name>.json.

Run: python experiments/gates/run_gates_family.py --env tickets
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cus.envs.family import GenEnv, SPECS   # noqa: E402


BETA_GRIDS = {
    "tickets": [0.0, 1.196, 2.393, 3.589, 4.785, 5.982, 7.178],
    "fraud": [0.0, 0.786, 1.572, 2.358, 3.144, 3.930, 4.716],
    "moderation": [0.0, 0.821, 1.641, 2.462, 3.283, 4.103, 4.924],
    "compliance": [0.0, 1.101, 2.202, 3.304, 4.405, 5.506, 6.607],
}


def gate_config(name: str) -> dict:
    spec = SPECS[name]
    return {
        "experiment": f"env_{name}_gates",
        "env_seed": spec.env_seed,
        "n_demo": spec.n_demo,
        "expert_noise": spec.expert_noise,
        "max_depth": spec.max_depth,
        "min_samples_leaf": spec.min_samples_leaf,
        "tilt_features": spec.tilt_features,
        "primary_tilt": spec.primary_tilt,
        "beta_grid": BETA_GRIDS[name],
        "g1_n": 50000, "g1_min_solvable": 0.98,
        "g2_n": 200000, "g2_n_sigma": 5.0,
        "g3_n": 60000, "g3_min_auc_lift": 0.02,
        "g3_audit": "decision-conditional: wrong ~ (s, dec) vs + (x_j, dec*x_j)",
        "g4_n": 40000, "g4_min_region_share": 0.05,
        "g5_n_lambda": 400, "g5_min_cells": 20,
        "seed": 20260819,
    }


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def main() -> None:
    name = sys.argv[sys.argv.index("--env") + 1]
    spec = SPECS[name]
    cfg = gate_config(name)
    root = pathlib.Path(__file__).resolve().parents[2]
    freeze = json.loads((root / "registrations" / f"env_{name}.json").read_text())
    h = config_hash(cfg)
    if freeze.get("gate_config_hash") != h:
        raise SystemExit(f"Gate config hash {h} does not match frozen "
                         f"{freeze.get('gate_config_hash')}.")
    print(f"[freeze] {name}: gate config {h} matches registration")

    env = GenEnv.induce(name)
    rng = np.random.default_rng([cfg["seed"], 77])
    report = {"gate_config": cfg, "n_rules": len(env.leaf_score)}

    # G1
    X, u = spec.generate(np.random.default_rng([cfg["seed"], 78]), cfg["g1_n"])
    frac_det = float(np.mean(spec.gold(X, u) == spec.gold(X, u)))
    frac_def = float(np.mean(np.isfinite(X).all(axis=1)))
    g1 = frac_det >= cfg["g1_min_solvable"] and frac_def >= cfg["g1_min_solvable"]
    report["G1"] = {"deterministic": frac_det, "well_defined": frac_def,
                    "pass": bool(g1)}
    print(f"[G1] deterministic={frac_det:.4f} defined={frac_def:.4f} "
          f"-> {'PASS' if g1 else 'FAIL'}")

    # G2 at beta_max per tilt feature
    bmax = max(cfg["beta_grid"])
    g2_all = True
    g2 = {}
    for feat in cfg["tilt_features"]:
        j = spec.bounded[feat]
        Xs, _ = spec.generate(rng, cfg["g2_n"])
        w = np.exp(bmax * Xs[:, j])
        wn = w / w.mean()
        Xr, _ = env.draw_instances(rng, cfg["g2_n"], bmax, feat)
        other = (j + 1) % spec.n_features
        ok_all = True
        checks = {}
        for gname, gs, gr in [("phi", Xs[:, j], Xr[:, j]),
                              ("phi2", Xs[:, j] ** 2, Xr[:, j] ** 2),
                              ("nuisance", Xs[:, other], Xr[:, other])]:
            src = wn * gs
            m_s, se_s = float(src.mean()), float(src.std() / np.sqrt(len(src)))
            m_r, se_r = float(gr.mean()), float(gr.std() / np.sqrt(len(gr)))
            tol = cfg["g2_n_sigma"] * float(np.hypot(se_s, se_r))
            ok = abs(m_s - m_r) <= tol
            ok_all &= ok
            checks[gname] = {"diff": abs(m_s - m_r), "tol": tol, "ok": bool(ok)}
        g2[feat] = {"checks": checks, "pass": bool(ok_all)}
        g2_all &= ok_all
        print(f"[G2] {feat}: {'PASS' if ok_all else 'FAIL'}")
    report["G2"] = {**g2, "pass": bool(g2_all)}

    # G3 decision-conditional
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
    g3, best = {}, 0.0
    for feat in cfg["tilt_features"]:
        j = spec.bounded[feat]
        x = ct.X[:, j]
        lift = auc([ct.s, dec, x, dec * x]) - base
        g3[feat] = {"auc_lift": lift}
        best = max(best, lift)
        print(f"[G3] + ({feat}, dec*{feat}): lift {lift:+.4f}")
    g3_pass = best >= cfg["g3_min_auc_lift"]
    report["G3"] = {"auc_s_dec": base, "per_feature": g3, "best_lift": best,
                    "pass": bool(g3_pass)}
    print(f"[G3] -> {'PASS' if g3_pass else 'FAIL'}")

    # G4 region mass across the grid
    g4, g4_pass = {}, True
    for b in cfg["beta_grid"]:
        ctb = env.case_table(rng, cfg["g4_n"], beta=b,
                             feature=cfg["primary_tilt"])
        share = float(ctb.region.mean())
        ok = share >= cfg["g4_min_region_share"]
        g4[str(b)] = {"region1_share": share, "ok": bool(ok)}
        g4_pass &= ok
    report["G4"] = {**g4, "pass": bool(g4_pass)}
    print(f"[G4] shares {g4[str(cfg['beta_grid'][0])]['region1_share']:.3f} "
          f"to {g4[str(bmax)]['region1_share']:.3f} "
          f"-> {'PASS' if g4_pass else 'FAIL'}")

    # G5 spread
    lams = np.linspace(0.0, 1.0, cfg["g5_n_lambda"])
    cells = len(np.unique(np.searchsorted(
        lams, sorted(set(env.leaf_score.values())))))
    g5 = cells >= cfg["g5_min_cells"]
    report["G5"] = {"distinct_grid_cells": cells, "pass": bool(g5)}
    print(f"[G5] {cells} cells -> {'PASS' if g5 else 'FAIL'}")

    all_pass = bool(g1 and g2_all and g3_pass and g4_pass and g5)
    report["all_pass"] = all_pass
    print(f"[gates] {name}: {'ALL PASS' if all_pass else 'FAILED'}")

    out = root / "artifacts" / f"gates_{name}_{h}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
