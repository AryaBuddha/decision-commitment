"""WP2 Phase 5, stage B: the prospective sweep against the registered
envelope. EVIDENCE TIER. The capstone.

Realized risk of the DEPLOYED pipeline (the same estimator spec whose
certificate produced the registered envelope) across the same
(feature, beta) grid, 300 trials per cell, plus the oracle arm. The
verdicts compare realized excess to the envelope registered BEFORE this
file ever ran.

Run:   python experiments/wp2_prospective/run.py
Smoke: add --smoke 10
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import crc, shift, tests            # noqa: E402
from cus.envs.returns import ReturnsEnv      # noqa: E402
sys.path.insert(0, str(ROOT / "experiments" / "wp2_prospective"))
from envelope import CONFIG, config_hash     # noqa: E402


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp2_prospective.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")
    return reg


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    reg = check_registration()
    envlp = {(r["feature"], r["beta"]): r for r in reg["envelope"]["rows"]}
    env = ReturnsEnv.induce()
    n_trials = smoke if smoke is not None else CONFIG["n_trials_sweep"]
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    alpha = CONFIG["alpha"]
    de = CONFIG["deployed_estimator"]
    print(f"[wp2p5] {'PILOT SMOKE' if smoke else 'REAL (evidence tier), the capstone'}")
    cells = []
    for fi, (feat, betas) in enumerate(CONFIG["features"].items()):
        for bi, beta in enumerate(betas):
            t0 = time.time()
            rng = np.random.default_rng([CONFIG["seed"], 980, fi, bi])
            exc_e, exc_o, defer_e = [], [], []
            for _ in range(n_trials):
                cal = env.case_table(rng, CONFIG["n_cal"])
                ev = env.case_table(rng, CONFIG["n_eval"], beta=beta,
                                    feature=feat)
                Xs, _ = env.draw_instances(rng, de["n_fit"])
                Xt, _ = env.draw_instances(rng, de["n_fit"], beta, feat)
                w_fn = shift.fit_ratio(Xs, Xt, clip=tuple(de["clip"]),
                                       C=de["C"])
                wh, whe = w_fn(cal.X), w_fn(ev.X)
                losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
                lam_e = crc.lhat_prop2(losses, lambdas, alpha, wh, whe)
                exc_e.append(float(((ev.s >= np.asarray(lam_e)) & ev.wrong).mean()) - alpha)
                defer_e.append(float((ev.s < np.asarray(lam_e)).mean()))
                w = np.exp(env.tilt_logweight(cal.X, beta, feat))
                wev = np.exp(env.tilt_logweight(ev.X, beta, feat))
                lam_o = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
                exc_o.append(float(((ev.s >= np.asarray(lam_o)) & ev.wrong).mean()) - alpha)
            n = len(exc_e)
            e = envlp[(feat, beta)]
            em, ese = float(np.mean(exc_e)), float(np.std(exc_e, ddof=1) / np.sqrt(n))
            om, ose = float(np.mean(exc_o)), float(np.std(exc_o, ddof=1) / np.sqrt(n))
            covered = em <= e["bound_median"]
            cells.append({"feature": feat, "beta": beta,
                          "excess_est_mean": em, "excess_est_se": ese,
                          "excess_oracle_mean": om, "excess_oracle_se": ose,
                          "deferral_est_mean": float(np.mean(defer_e)),
                          "bound_median_registered": e["bound_median"],
                          "covered": bool(covered),
                          "price": e["bound_median"] - em})
            print(f"[wp2p5] {feat:<15} beta={beta}: excess {em:+.4f}±{ese:.4f}"
                  f" vs envelope {e['bound_median']:+.4f} "
                  f"{'COVERED' if covered else 'BREACHED'} "
                  f"(oracle {om:+.4f}) ({time.time() - t0:.0f}s)", flush=True)
    cov = sum(c["covered"] for c in cells)
    print(f"[wp2p5] envelope coverage {cov}/{len(cells)}; "
          f"mean price {np.mean([c['price'] for c in cells]):+.4f}")
    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2p5_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps({"cells": cells}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
