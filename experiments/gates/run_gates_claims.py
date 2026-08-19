"""Validation gates 1-5 for environment 1 (claims triage), RESEARCH_PLAN 3.3.

The gate config below is frozen in registrations/env_claims.json (hash
enforced); thresholds are set BEFORE the gates run. The report is archived
immutably under artifacts/ and a committed copy goes to docs/gates/.

  G1  oracle solvability   gold recomputable and deterministic on the
                           actual case sample, >= 0.98 (here: exact replay).
  G2  shift fidelity       rung-2 rejection draws match importance-weighted
                           source moments within 5 combined MC SE, for every
                           registered tilt feature at beta_max. Mostly an
                           extractor-bug check; rung 2 makes it hold by
                           construction if the code is right.
  G3  evidence blindness   held-out AUC lift of wrong ~ (s, x_j) over
                           wrong ~ s must be >= 0.02 for at least one
                           registered tilt candidate, else this environment
                           cannot exhibit the failure WP1 measures and is
                           redirected, with this audit published.
  G4  region mass          expected minority share >= 0.05 (>= 50 cases per
                           1000-draw) at EVERY beta in the sweep grid.
  G5  evidence spread      induced rule scores occupy >= 20 cells of the
                           400-point lambda grid.

Run: python experiments/gates/run_gates_claims.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cus.envs import claims                                    # noqa: E402
from cus.envs.claims import ClaimsEnv, generate, gold          # noqa: E402


GATE_CONFIG = {
    "experiment": "env_claims_gates",
    "env_seed": 20260818,
    "n_demo": 6000,
    "expert_noise": 0.06,
    "tilt_features": ["severity", "inconsistency", "doc_completeness"],
    "primary_tilt_feature": "severity",
    "beta_grid": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "g1_n": 50000, "g1_min_solvable": 0.98,
    "g2_n": 200000, "g2_n_sigma": 5.0,
    "g3_n": 60000, "g3_min_auc_lift": 0.02,
    "g4_n": 40000, "g4_min_region_share": 0.05,
    "g5_n_lambda": 400, "g5_min_cells": 20,
    "seed": 20260818,
}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    freeze = json.loads((root / "registrations" / "env_claims.json").read_text())
    h = config_hash(GATE_CONFIG)
    if freeze.get("gate_config_hash") != h:
        raise SystemExit(f"Gate config hash {h} does not match frozen "
                         f"{freeze.get('gate_config_hash')}.")
    print(f"[freeze] gate config {h} matches registrations/env_claims.json")

    cfg = GATE_CONFIG
    env = ClaimsEnv.induce(cfg["env_seed"], cfg["n_demo"], cfg["expert_noise"])
    rng = np.random.default_rng([cfg["seed"], 77])
    report = {"gate_config": cfg, "n_rules": len(env.leaf_score)}

    # G1: oracle solvability, exact recompute.
    X, u = generate(rng, cfg["g1_n"])
    g1a, g1b = gold(X, u), gold(X, u)
    frac_det = float(np.mean(g1a == g1b))
    frac_def = float(np.mean(np.isfinite(X).all(axis=1)))
    g1_pass = frac_det >= cfg["g1_min_solvable"] and frac_def >= cfg["g1_min_solvable"]
    report["G1"] = {"deterministic": frac_det, "well_defined": frac_def,
                    "pass": bool(g1_pass)}
    print(f"[G1] solvability: deterministic={frac_det:.4f} "
          f"defined={frac_def:.4f} -> {'PASS' if g1_pass else 'FAIL'}")

    # G2: rung-2 fidelity at beta_max per tilt feature.
    beta_max = max(cfg["beta_grid"])
    g2 = {}
    g2_pass = True
    for feat in cfg["tilt_features"]:
        j = claims.BOUNDED[feat]
        Xs, _ = generate(rng, cfg["g2_n"])
        w = np.exp(env.tilt_logweight(Xs, beta_max, feat))
        wn = w / w.mean()
        Xr, _ = env.draw_instances(rng, cfg["g2_n"], beta_max, feat)
        other = claims.BOUNDED["adjuster_load"]
        checks, ok_all = {}, True
        for name, gs, gr in [
            ("phi", Xs[:, j], Xr[:, j]),
            ("phi2", Xs[:, j] ** 2, Xr[:, j] ** 2),
            ("nuisance", Xs[:, other], Xr[:, other]),
        ]:
            src = wn * gs
            m_s, se_s = float(src.mean()), float(src.std() / np.sqrt(len(src)))
            m_r, se_r = float(gr.mean()), float(gr.std() / np.sqrt(len(gr)))
            tol = cfg["g2_n_sigma"] * float(np.hypot(se_s, se_r))
            ok = abs(m_s - m_r) <= tol
            ok_all &= ok
            checks[name] = {"iw": m_s, "rejection": m_r,
                            "diff": abs(m_s - m_r), "tol": tol, "ok": bool(ok)}
        g2[feat] = {"checks": checks, "pass": bool(ok_all)}
        g2_pass &= ok_all
        print(f"[G2] {feat}: {'PASS' if ok_all else 'FAIL'} "
              f"(worst diff/tol { max(c['diff']/c['tol'] for c in checks.values()):.2f})")
    report["G2"] = {**g2, "pass": bool(g2_pass)}

    # G3: evidence-blindness audit.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ct = env.case_table(rng, cfg["g3_n"])
    half = cfg["g3_n"] // 2
    tr, te = slice(0, half), slice(half, None)
    y_tr, y_te = ct.wrong[tr], ct.wrong[te]

    def auc(cols):
        Z = np.column_stack(cols)
        clf = LogisticRegression(max_iter=2000).fit(Z[tr], y_tr)
        return float(roc_auc_score(y_te, clf.predict_proba(Z[te])[:, 1]))

    auc_s = auc([ct.s])
    g3, best = {}, 0.0
    for feat in cfg["tilt_features"]:
        j = claims.BOUNDED[feat]
        lift = auc([ct.s, ct.X[:, j]]) - auc_s
        g3[feat] = {"auc_lift": lift}
        best = max(best, lift)
        print(f"[G3] wrong ~ (s, {feat}) AUC lift over wrong ~ s: {lift:+.4f}")
    auc_full = auc([ct.s] + [ct.X[:, k] for k in range(ct.X.shape[1])])
    g3_pass = best >= cfg["g3_min_auc_lift"]
    report["G3"] = {"auc_s_only": auc_s, "auc_full_X": auc_full,
                    "per_feature": g3, "best_lift": best, "pass": bool(g3_pass)}
    print(f"[G3] s-only AUC {auc_s:.4f}, full-X AUC {auc_full:.4f} "
          f"-> {'PASS' if g3_pass else 'FAIL'}")

    # G4: region mass at every beta.
    g4, g4_pass = {}, True
    for b in cfg["beta_grid"]:
        ctb = env.case_table(rng, cfg["g4_n"], beta=b,
                             feature=cfg["primary_tilt_feature"])
        share = float(ctb.region.mean())
        ok = share >= cfg["g4_min_region_share"]
        g4[str(b)] = {"region1_share": share, "ok": bool(ok)}
        g4_pass &= ok
        print(f"[G4] beta={b}: region-1 share {share:.4f} {'ok' if ok else 'FAIL'}")
    report["G4"] = {**g4, "pass": bool(g4_pass)}

    # G5: evidence-score spread.
    lams = np.linspace(0.0, 1.0, cfg["g5_n_lambda"])
    cells = len(np.unique(np.searchsorted(
        lams, sorted(set(env.leaf_score.values())))))
    g5_pass = cells >= cfg["g5_min_cells"]
    report["G5"] = {"distinct_grid_cells": cells, "pass": bool(g5_pass)}
    print(f"[G5] {cells} distinct lambda-grid cells "
          f"-> {'PASS' if g5_pass else 'FAIL'}")

    all_pass = bool(g1_pass and g2_pass and g3_pass and g4_pass and g5_pass)
    report["all_pass"] = all_pass
    print(f"\n[gates] environment claims: {'ALL PASS' if all_pass else 'FAILED'}")

    out_dir = root / "artifacts" / f"gates_claims_{h}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()
