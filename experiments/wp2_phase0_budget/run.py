"""WP2 Phase 0: m(alpha, n_cal) on claims. EVIDENCE TIER.

Carries the phase's RISKY prediction: if the leading m-mechanism candidate
(H2, empirical-crossing noise against a convex loss curve) is real, the
amplification m must fall toward a limit as n_cal grows, because the
threshold noise that Jensen-amplifies shrinks like 1/sqrt(n_cal * local
mass). If m is flat in n_cal, H2 is dead before Phase 1 starts.

Grid: alpha {0.05, 0.10} x n_cal {100, 250, 1000, 4000, 10000}, temper
family, fresh seed 20260820. Also measures sd(lambda_hat) per cell, the
H2 ingredient, directly.

Run:   python experiments/wp2_phase0_budget/run.py
Smoke: add --smoke 10
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import crc, tests                 # noqa: E402
from cus.envs.claims import ClaimsEnv      # noqa: E402


CONFIG = {
    "experiment": "wp2_phase0_budget",
    "environment": "claims",
    "tilt_feature": "inconsistency",
    "alphas": [0.05, 0.10],
    "n_cals": [100, 250, 1000, 4000, 10000],
    "betas": [3.0, 5.0],
    "temper_gammas": [0.0, 0.25, 0.5, 0.75, 0.8, 1.0, 1.25, 1.5],
    "n_eval": 1000, "n_trials": 300, "n_lambda": 400,
    "fd_window": 0.05, "min_abs_a": 0.003, "min_slope_what": 0.02,
    "z_one_sided": 1.645, "delta_control": 0.005,
    "seed": 20260820,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp2_phase0_budget.json").read_text())
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
    env = ClaimsEnv.induce()
    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    feat = CONFIG["tilt_feature"]
    h = CONFIG["fd_window"]
    print(f"[wp2p0b] {'PILOT SMOKE' if smoke else 'REAL (evidence tier)'}")

    cells, mgrid = [], []
    for ai, alpha in enumerate(CONFIG["alphas"]):
        for ni, ncal in enumerate(CONFIG["n_cals"]):
            t0 = time.time()
            grid_cells = []
            for gi, g in enumerate(CONFIG["temper_gammas"]):
                for bi, beta in enumerate(CONFIG["betas"]):
                    rng = np.random.default_rng(
                        [CONFIG["seed"], 50, ai, ni, gi, bi])
                    aligned, paired, ro_, sw_, sh_, lam_sd = \
                        [], [], [], [], [], []
                    for _ in range(n_trials):
                        cal = env.case_table(rng, ncal)
                        ev = env.case_table(rng, CONFIG["n_eval"],
                                            beta=beta, feature=feat)
                        w = np.exp(env.tilt_logweight(cal.X, beta, feat))
                        wev = np.exp(env.tilt_logweight(ev.X, beta, feat))
                        what, wevhat = w ** g, wev ** g
                        losses = crc.commit_error_losses(cal.s, cal.wrong,
                                                         lambdas)
                        lam_o = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
                        ls = float(np.mean(lam_o))
                        ro = float(((ev.s >= np.asarray(lam_o)) & ev.wrong).mean())
                        lam_e = crc.lhat_prop2(losses, lambdas, alpha,
                                               what, wevhat)
                        le = float(np.mean(lam_e))
                        re = float(((ev.s >= np.asarray(lam_e)) & ev.wrong).mean())
                        wn, hn = w / w.mean(), what / what.mean()
                        L = ((cal.s >= ls) & cal.wrong).astype(float)
                        aligned.append(float(np.mean((wn - hn) * L)))
                        paired.append(re - ro)
                        ro_.append(ro)
                        lam_sd.append(le)
                        Llo = ((cal.s >= ls - h) & cal.wrong).astype(float)
                        Lhi = ((cal.s >= ls + h) & cal.wrong).astype(float)
                        sw_.append(float(np.mean(wn * (Llo - Lhi)) / (2 * h)))
                        sh_.append(float(np.mean(hn * (Llo - Lhi)) / (2 * h)))
                    n = len(paired)
                    mw, mh = float(np.mean(sw_)), float(np.mean(sh_))
                    c = {"alpha": alpha, "n_cal": ncal, "gamma": g,
                         "beta": beta,
                         "aligned_mean": float(np.mean(aligned)),
                         "aligned_se": float(np.std(aligned, ddof=1) / np.sqrt(n)),
                         "paired_diff_mean": float(np.mean(paired)),
                         "paired_diff_se": float(np.std(paired, ddof=1) / np.sqrt(n)),
                         "risk_oracle_mean": float(np.mean(ro_)),
                         "risk_oracle_se": float(np.std(ro_, ddof=1) / np.sqrt(n)),
                         "lambda_hat_sd": float(np.std(lam_sd, ddof=1)),
                         "kappa_pred": mw / mh if mh > CONFIG["min_slope_what"] else None,
                         "slope_what": mh}
                    grid_cells.append(c)
            guard = [c for c in grid_cells
                     if c["kappa_pred"] is not None
                     and abs(c["aligned_mean"]) >= CONFIG["min_abs_a"]
                     and c["slope_what"] >= CONFIG["min_slope_what"]]
            row = {"alpha": alpha, "n_cal": ncal, "n_guarded": len(guard),
                   "b_env": float(np.mean([c["risk_oracle_mean"]
                                           for c in grid_cells])) - alpha,
                   "lambda_hat_sd_mean": float(np.mean(
                       [c["lambda_hat_sd"] for c in grid_cells]))}
            if len(guard) >= 4:
                kp = np.array([c["kappa_pred"] for c in guard])
                km = np.array([c["paired_diff_mean"] / c["aligned_mean"]
                               for c in guard])
                m = float(kp @ km / (kp @ kp))
                r = km - m * kp
                row["m"] = m
                row["m_se"] = float(np.sqrt((r @ r) / (len(kp) - 1) / (kp @ kp)))
            else:
                row["m"] = row["m_se"] = None
            mgrid.append(row)
            cells.extend(grid_cells)
            m_str = "None" if row["m"] is None else f"{row['m']:.3f}±{row['m_se']:.3f}"
            print(f"[wp2p0b] alpha={alpha:<5} n_cal={ncal:<6} m={m_str} "
                  f"b={row['b_env']:+.4f} lam_sd={row['lambda_hat_sd_mean']:.4f} "
                  f"guarded={row['n_guarded']} ({time.time() - t0:.0f}s)",
                  flush=True)

    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2p0b_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps(
        {"cells": cells, "m_grid": mgrid}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
