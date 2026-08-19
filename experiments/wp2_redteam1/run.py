"""WP2 Phase 4, round 1: break the frozen certificate. EVIDENCE TIER.

Three registered attacks on the certificate frozen in
registrations/wp2_certificate.json:

  R1-1  audit-blindness x blind estimator (surfaces i + ii): claims,
        inconsistency tilt at beta 6.5 (beyond every archived battery),
        ratio estimator fully deprived of the tilt feature. The target
        commit mass moves into interaction cells the logistic audit
        model cannot represent, and thin source bins weaken CalErr_loc.
  R1-2  per-draw coverage under estimator variance: the starvation
        n_fit = 50 cells (tickets and compliance, top archived betas).
        The validation compares per-draw bounds to mean-over-fits
        excess; deployments live per draw. 200 paired draws, each
        scoring the SAME fitted w_hat's certificate against ITS OWN
        realized excess on a 10000-case evaluation window.
  R1-3  post-audit drift (surface iii): tickets (no reconstruction
        defense), everything audited at beta 3.589, deployed at beta
        {5.982, 7.178, 9.5}; the certificate keeps its stale target
        sample, as a deployment with a frozen audit would.

Run:   python experiments/wp2_redteam1/run.py
Smoke: add --smoke 5
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

from cus import crc, shift, tests                 # noqa: E402
from cus.certificate import Audit, certificate    # noqa: E402
from cus.envs.claims import ClaimsEnv             # noqa: E402
from cus.envs.family import GenEnv                # noqa: E402


CONFIG = {
    "experiment": "wp2_redteam1",
    "alpha": 0.10,
    "n_cal": 1000, "n_src": 10000, "n_tgt": 10000, "n_audit": 60000,
    "n_eval_perdraw": 10000, "n_lambda": 400,
    "r11": {"env": "claims", "feature": "inconsistency", "beta": 6.5,
            "n_draws": 30, "n_excess_trials": 200},
    "r12": [{"env": "tickets", "feature": "frustration", "beta": 5.982,
             "n_fit": 50, "n_draws": 200},
            {"env": "compliance", "feature": "redline_density",
             "beta": 6.607, "n_fit": 50, "n_draws": 200}],
    "r13": {"env": "tickets", "feature": "frustration", "beta_audit": 3.589,
            "betas_deploy": [5.982, 7.178, 9.5], "n_draws": 30,
            "n_excess_trials": 200},
    "seed": 20260821,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp2_redteam1.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def make_env(name):
    return ClaimsEnv.induce() if name == "claims" else GenEnv.induce(name)


def tdim(env, ename, feat):
    if ename == "claims":
        from cus.envs.claims import BOUNDED
        return BOUNDED[feat]
    return env.spec.bounded[feat]


def pop_excess(env, what_fn, feat, beta, cfg, rng, n_trials, alpha):
    lambdas = np.linspace(0, 1, cfg["n_lambda"])
    exc = []
    for _ in range(n_trials):
        cal = env.case_table(rng, cfg["n_cal"])
        ev = env.case_table(rng, 1000, beta=beta, feature=feat)
        wh, whe = what_fn(cal, rng), what_fn(ev, rng)
        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        lam = crc.lhat_prop2(losses, lambdas, alpha, wh, whe)
        exc.append(float(((ev.s >= np.asarray(lam)) & ev.wrong).mean()) - alpha)
    return float(np.mean(exc)), float(np.std(exc, ddof=1) / np.sqrt(len(exc)))


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration()
    alpha = CONFIG["alpha"]
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    out = {}
    sc = (lambda n: min(n, smoke * 10) if smoke else n)
    scd = (lambda n: min(n, smoke) if smoke else n)

    # ---- R1-1 ----
    c = CONFIG["r11"]
    env = make_env(c["env"])
    rng = np.random.default_rng([CONFIG["seed"], 950])
    aud = Audit(env, rng, CONFIG["n_audit"])
    k = tdim(env, c["env"], c["feature"])

    def blind(pool, r):
        keep = [j for j in range(pool.X.shape[1]) if j != k]
        Xs, _ = env.draw_instances(r, 1000)
        Xt, _ = env.draw_instances(r, 1000, c["beta"], c["feature"])
        return shift.fit_ratio(Xs[:, keep], Xt[:, keep])(pool.X[:, keep])
    e, ese = pop_excess(env, blind, c["feature"], c["beta"], CONFIG, rng,
                        sc(c["n_excess_trials"]), alpha)
    bounds = [certificate(env, aud, alpha, rng, c["feature"], c["beta"],
                          lambda p: blind(p, rng), n_src=CONFIG["n_src"],
                          n_tgt=CONFIG["n_tgt"])["excess_bound"]
              for _ in range(scd(c["n_draws"]))]
    out["r11"] = {"excess": e, "excess_se": ese,
                  "bound_median": float(np.median(bounds)),
                  "bound_min": float(np.min(bounds)),
                  "breach_median": e - float(np.median(bounds)),
                  "covered_draws": int(sum(b >= e for b in bounds)),
                  "n_draws": len(bounds)}
    print(f"[rt1-1] excess {e:+.4f} bound_med {np.median(bounds):+.4f} "
          f"breach {out['r11']['breach_median']:+.4f} "
          f"covered {out['r11']['covered_draws']}/{len(bounds)}", flush=True)

    # ---- R1-2 ----
    out["r12"] = []
    for spec in CONFIG["r12"]:
        env = make_env(spec["env"])
        rng = np.random.default_rng([CONFIG["seed"], 951,
                                     abs(hash(spec["env"])) % 1000])
        aud = Audit(env, rng, CONFIG["n_audit"])
        feat, beta = spec["feature"], spec["beta"]
        cov, breaches = 0, []
        t0 = time.time()
        nd = scd(spec["n_draws"])
        for _ in range(nd):
            Xs, _ = env.draw_instances(rng, spec["n_fit"])
            Xt, _ = env.draw_instances(rng, spec["n_fit"], beta, feat)
            w_fn = shift.fit_ratio(Xs, Xt)
            what_fn = lambda pool: w_fn(pool.X)          # noqa: E731
            res = certificate(env, aud, alpha, rng, feat, beta, what_fn,
                              n_cal=CONFIG["n_cal"], n_src=CONFIG["n_src"],
                              n_tgt=CONFIG["n_tgt"])
            # same w_hat's own realized excess on a wide window
            cal = env.case_table(rng, CONFIG["n_cal"])
            ev = env.case_table(rng, CONFIG["n_eval_perdraw"], beta=beta,
                                feature=feat)
            losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
            lam = crc.lhat_prop2(losses, lambdas, alpha, w_fn(cal.X),
                                 w_fn(ev.X))
            exc = float(((ev.s >= np.asarray(lam)) & ev.wrong).mean()) - alpha
            cov += res["excess_bound"] >= exc
            breaches.append(exc - res["excess_bound"])
        row = {"env": spec["env"], "coverage": cov, "n_draws": nd,
               "breach_p90": float(np.percentile(breaches, 90)),
               "breach_max": float(np.max(breaches))}
        out["r12"].append(row)
        print(f"[rt1-2] {spec['env']}: per-draw coverage {cov}/{nd} "
              f"breach_p90 {row['breach_p90']:+.4f} ({time.time()-t0:.0f}s)",
              flush=True)

    # ---- R1-3 ----
    c = CONFIG["r13"]
    env = make_env(c["env"])
    rng = np.random.default_rng([CONFIG["seed"], 952])
    aud = Audit(env, rng, CONFIG["n_audit"])

    def stale(pool, r):
        Xs, _ = env.draw_instances(r, 1000)
        Xt, _ = env.draw_instances(r, 1000, c["beta_audit"], c["feature"])
        return shift.fit_ratio(Xs, Xt)(pool.X)
    out["r13"] = []
    for bd in c["betas_deploy"]:
        e, ese = pop_excess(env, stale, c["feature"], bd, CONFIG, rng,
                            sc(c["n_excess_trials"]), alpha)
        bounds = [certificate(env, aud, alpha, rng, c["feature"],
                              c["beta_audit"], lambda p: stale(p, rng),
                              n_src=CONFIG["n_src"],
                              n_tgt=CONFIG["n_tgt"])["excess_bound"]
                  for _ in range(scd(c["n_draws"]))]
        row = {"beta_deploy": bd, "excess": e, "excess_se": ese,
               "bound_median": float(np.median(bounds)),
               "breach_median": e - float(np.median(bounds)),
               "covered_draws": int(sum(b >= e for b in bounds)),
               "n_draws": len(bounds)}
        out["r13"].append(row)
        print(f"[rt1-3] deploy beta {bd}: excess {e:+.4f} "
              f"bound_med {row['bound_median']:+.4f} "
              f"breach {row['breach_median']:+.4f} "
              f"covered {row['covered_draws']}/{row['n_draws']}", flush=True)

    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2rt1_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
