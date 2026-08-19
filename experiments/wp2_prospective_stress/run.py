"""WP2 Phase 5b, stage B: the stressed prospective sweep. EVIDENCE TIER.

Realized risk of the twelve DEGRADED deployments against the envelope
registered before this file ever ran. The claim under test is coverage
under stress: the envelope approached and not crossed.

Run:   python experiments/wp2_prospective_stress/run.py
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

from cus import crc, tests                   # noqa: E402
from cus.envs.returns import ReturnsEnv      # noqa: E402
sys.path.insert(0, str(ROOT / "experiments" / "wp2_prospective_stress"))
from envelope import CONFIG, config_hash, build_what_fn   # noqa: E402


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" /
                      "wp2_prospective_stress.json").read_text())
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
    envlp = {(r["kind"], r["feature"], r["beta"]): r
             for r in reg["envelope"]["rows"]}
    env = ReturnsEnv.induce()
    n_trials = smoke if smoke is not None else CONFIG["n_trials_sweep"]
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    alpha = CONFIG["alpha"]
    print(f"[wp2p5s] {'PILOT SMOKE' if smoke else 'REAL (evidence tier), coverage under stress'}")
    cells = []
    for si, spec in enumerate(CONFIG["stress_cells"]):
        kind, feat, beta = spec["kind"], spec["feature"], spec["beta"]
        t0 = time.time()
        rng = np.random.default_rng([CONFIG["seed"], 991, si])
        exc_e, exc_o = [], []
        for _ in range(n_trials):
            what_fn = build_what_fn(env, kind, feat, beta, rng, CONFIG)
            cal = env.case_table(rng, CONFIG["n_cal"])
            ev = env.case_table(rng, CONFIG["n_eval"], beta=beta, feature=feat)
            wh, whe = what_fn(cal), what_fn(ev)
            losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
            lam_e = crc.lhat_prop2(losses, lambdas, alpha, wh, whe)
            exc_e.append(float(((ev.s >= np.asarray(lam_e)) & ev.wrong).mean()) - alpha)
            w = np.exp(env.tilt_logweight(cal.X, beta, feat))
            wev = np.exp(env.tilt_logweight(ev.X, beta, feat))
            lam_o = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
            exc_o.append(float(((ev.s >= np.asarray(lam_o)) & ev.wrong).mean()) - alpha)
        n = len(exc_e)
        e = envlp[(kind, feat, beta)]
        em, ese = float(np.mean(exc_e)), float(np.std(exc_e, ddof=1) / np.sqrt(n))
        om, ose = float(np.mean(exc_o)), float(np.std(exc_o, ddof=1) / np.sqrt(n))
        covered = em <= e["bound_median"]
        approach = em / e["bound_median"] if e["bound_median"] > 0 else None
        cells.append({"kind": kind, "feature": feat, "beta": beta,
                      "excess_est_mean": em, "excess_est_se": ese,
                      "excess_oracle_mean": om, "excess_oracle_se": ose,
                      "bound_median_registered": e["bound_median"],
                      "covered": bool(covered),
                      "approach_ratio": approach,
                      "price": e["bound_median"] - em})
        print(f"[wp2p5s] {kind:<19} {feat:<15} beta={beta}: "
              f"excess {em:+.4f}±{ese:.4f} vs envelope {e['bound_median']:+.4f} "
              f"{'COVERED' if covered else 'BREACHED'} "
              f"approach {approach and round(approach, 2)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    cov = sum(c["covered"] for c in cells)
    appr = max(c["approach_ratio"] for c in cells if c["approach_ratio"] is not None)
    print(f"[wp2p5s] coverage {cov}/{len(cells)}; max approach ratio {appr:.2f}; "
          f"min price {min(c['price'] for c in cells):+.4f}")
    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2p5s_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps({"cells": cells}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
