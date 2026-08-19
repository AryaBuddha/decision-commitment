"""WP2 Phase 1: the m decomposition switchboard. PILOT TIER (analytic world).

The amplification m is measured on the analytic world with each piece of
the empirical machinery switched on one at a time, against quadrature-
exact population baselines. In this world the population secant identity
excess = kappa * a is EXACT (H4 absent by construction), so whatever
amplification appears belongs to the machinery:

  level 0  pop      population crossing, continuous lambda, no B-term.
                    m = 1 identically; computed as an instrument check.
  level 1  +grid    the finite 400-point grid inf and the B/(n+1)
                    pseudo-loss charge, still deterministic (H1 alone).
  level 2  +noise   empirical calibration draws decide the threshold
                    (grid and B-term kept), ONE global threshold using
                    the population mean test weight, and the realized
                    risk is evaluated by QUADRATURE at the realized
                    threshold, so no evaluation-window or wrongness
                    noise enters (H1 + H2).
  level 3  +percov  the literal Prop 2 per-test-covariate thresholds on
                    a finite test batch, conditional-exact evaluation
                    q(x) * 1{s >= lam(x)} (H1 + H2 + H3). This is the
                    full deployed procedure minus only the evaluation
                    window's Bernoulli noise.

Per (alpha, n_cal, gamma, b): the paired difference (temper arm minus
oracle arm) at each level, the exact a and kappa from quadrature, and
the per-level m = pd / (kappa_exact * a_exact). The alpha grid extends
to 0.02 and n_cal to 100000 (the H2 discriminator: m(level 2) -> 1).

Run:   python experiments/wp2_mechanism_switchboard/run.py
Smoke: add --smoke 10
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import crc, tests                    # noqa: E402
from cus.envs import analytic as A            # noqa: E402


CONFIG = {
    "experiment": "wp2_mechanism_switchboard",
    "tier": "pilot-analytic",
    "alphas": [0.02, 0.05, 0.10, 0.15],
    "n_cals": [250, 1000, 10000, 100000],
    "b_tilt": 0.8,
    "gammas": [0.0, 0.4, 0.7, 1.3, 1.6],
    "n_test": 1000,
    "n_trials": 300, "n_trials_ncal_100000": 60,
    "n_lambda": 400,
    "seed": 20260820,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" /
                      "wp2_mechanism_switchboard.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def grid_threshold_pop(alpha, c, n, w_test, lambdas):
    """Level 1: the empirical formula's structure applied to the exact
    population curve. Deterministic."""
    S_w = float(n)                       # weights normalized to mean 1
    for lam in lambdas:
        bound = (S_w * A.Rw(lam, c) + w_test) / (S_w + w_test)
        if bound <= alpha:
            return float(lam)
    return float(lambdas[-1])


def mean_test_weight(c_hat, b):
    """E_Q[w_hat] for w_hat = e^{c x - c^2/2} under Q = N(b, 1):
    exp(c*b) is the closed form."""
    return float(np.exp(c_hat * b))


def run_cell(env, alpha, ncal, gamma, b, cfg, n_trials, rng):
    lambdas = np.linspace(0, 1, cfg["n_lambda"])
    c_hat = gamma * b
    # exact population quantities
    lam_star = A.lam_pop(alpha, b)
    kap = A.kappa_exact(alpha, b, gamma)
    a = A.a_exact(alpha, b, gamma)
    # level 0: population crossings (instrument check)
    lam_o0 = lam_star
    lam_e0 = A.lam_pop(alpha, c_hat)
    pd0 = A.rho(lam_e0, b) - A.rho(lam_o0, b)
    # level 1: + grid and B-term, deterministic
    wt_o, wt_e = mean_test_weight(b, b), mean_test_weight(c_hat, b)
    lam_o1 = grid_threshold_pop(alpha, b, ncal, wt_o, lambdas)
    lam_e1 = grid_threshold_pop(alpha, c_hat, ncal, wt_e, lambdas)
    pd1 = A.rho(lam_e1, b) - A.rho(lam_o1, b)
    # levels 2 and 3: empirical
    pd2_, pd3_ = [], []
    for _ in range(n_trials):
        cal = env.case_table(rng, ncal)
        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        w = np.exp(env.tilt_logweight(cal.X, b))
        what = np.exp(env.tilt_logweight(cal.X, c_hat))
        # level 2: global threshold, exact rho evaluation
        lam_o2 = crc.lhat_weighted_global(losses, lambdas, alpha, w, wt_o)
        lam_e2 = crc.lhat_weighted_global(losses, lambdas, alpha, what, wt_e)
        pd2_.append(A.rho(lam_e2, b) - A.rho(lam_o2, b))
        # level 3: literal Prop 2 on a finite test batch, conditional-exact
        Xt, _ = env.draw_instances(rng, cfg["n_test"], beta=b)
        st = norm.cdf(Xt[:, 0])
        qt = A.g_fn(Xt[:, 0]) * A.h_fn(Xt[:, 1])
        wev = np.exp(env.tilt_logweight(Xt, b))
        wevhat = np.exp(env.tilt_logweight(Xt, c_hat))
        lam_o3 = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
        lam_e3 = crc.lhat_prop2(losses, lambdas, alpha, what, wevhat)
        r_o3 = float(np.mean(qt * (st >= np.asarray(lam_o3))))
        r_e3 = float(np.mean(qt * (st >= np.asarray(lam_e3))))
        pd3_.append(r_e3 - r_o3)
    n = len(pd2_)
    out = {"alpha": alpha, "n_cal": ncal, "gamma": gamma, "b": b,
           "a_exact": a, "kappa_exact": kap, "lam_star": lam_star,
           "pd_pop": pd0, "pd_grid": pd1,
           "pd_noise_mean": float(np.mean(pd2_)),
           "pd_noise_se": float(np.std(pd2_, ddof=1) / np.sqrt(n)),
           "pd_percov_mean": float(np.mean(pd3_)),
           "pd_percov_se": float(np.std(pd3_, ddof=1) / np.sqrt(n))}
    if abs(a) > 1e-12:
        ka = kap * a
        out["m_pop"] = pd0 / ka
        out["m_grid"] = pd1 / ka
        out["m_noise"] = out["pd_noise_mean"] / ka
        out["m_noise_se"] = out["pd_noise_se"] / abs(ka)
        out["m_percov"] = out["pd_percov_mean"] / ka
        out["m_percov_se"] = out["pd_percov_se"] / abs(ka)
    return out


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    st = A.selftest(n=200_000)
    if not st["ok"]:
        raise SystemExit(f"[selftest] analytic quadrature vs MC FAILED: {st}")
    print("[selftest] analytic rung-1 exactness: PASS")
    check_registration()
    print(f"[wp2sw] {'PILOT SMOKE' if smoke else 'FULL (pilot-analytic tier)'}")
    env = A.AnalyticEnv()
    b = CONFIG["b_tilt"]
    cells = []
    for ai, alpha in enumerate(CONFIG["alphas"]):
        for ni, ncal in enumerate(CONFIG["n_cals"]):
            t0 = time.time()
            for gi, gamma in enumerate(CONFIG["gammas"]):
                n_tr = smoke if smoke is not None else (
                    CONFIG["n_trials_ncal_100000"] if ncal == 100000
                    else CONFIG["n_trials"])
                rng = np.random.default_rng([CONFIG["seed"], 300, ai, ni, gi])
                c = run_cell(env, alpha, ncal, gamma, b, CONFIG, n_tr, rng)
                cells.append(c)
            sub = [c for c in cells if c["alpha"] == alpha
                   and c["n_cal"] == ncal and "m_noise" in c]
            if sub:
                mg = float(np.mean([c["m_grid"] for c in sub]))
                mn = float(np.mean([c["m_noise"] for c in sub]))
                mp = float(np.mean([c["m_percov"] for c in sub]))
                print(f"[wp2sw] alpha={alpha:<5} n_cal={ncal:<7} "
                      f"m_grid={mg:.3f} m_noise={mn:.3f} m_percov={mp:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2sw_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps({"cells": cells}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
