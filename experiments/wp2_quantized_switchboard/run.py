"""WP2 Phase 1: the quantized switchboard. PILOT TIER (analytic world).

The smooth switchboard (wp2sw) left the gated-tier plateau with no owner:
H1/H2 die with budget, H3 never appears, H4 is structurally absent, yet
claims/spike/tickets plateau at m 1.1 to 1.8. The surviving structural
difference is SCORE DISCRETENESS. On step-shaped risk curves the two
arms cross on DIFFERENT plateaus, and

    excess = a(lam_e) + overshoot_e,   a(lam_e) = a(lam*) * G(lam_e)/G(lam*)

so the lambda*-referenced m carries the plateau ratio G(lam_e)/G(lam*)
> 1 plus the overshoot asymmetry: an amplification that survives
infinite calibration, grows as alpha shrinks relative to plateau loss
mass, and depends on environment geometry. This runner tests that
arithmetic two ways:

  Part A (factorized quantized world, sampled): scores snapped to K
    levels. Factorization makes the FD kappa EXACTLY H(b)/H(gamma b)
    (no instrument bias possible), so measured m isolates the
    PROCEDURAL crossing arithmetic. Per cell: m_measured (full Prop 2
    machinery, conditional-exact evaluation) vs m_derived (population
    grid crossing with the B/(n+1) charge, quadrature-exact).

  Part B (rank-2 quantized curves, deterministic): non-factorized
    geometry where the FD window CAN mislabel kappa on steps. The gap
    between rank-2 m_pop (FD kappa) and the factorized m_derived at
    matched (K, alpha, gamma) measures the instrument's contribution.

Run:   python experiments/wp2_quantized_switchboard/run.py
Smoke: add --smoke 5
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np
from scipy import integrate
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import crc, tests                    # noqa: E402
from cus.envs import analytic as A            # noqa: E402


CONFIG = {
    "experiment": "wp2_quantized_switchboard",
    "tier": "pilot-analytic",
    "Ks": [8, 16, 32, 64, 0],
    "alphas": [0.05, 0.10],
    "n_cals": [1000, 10000, 100000],
    "b_tilt": 0.8,
    "gammas": [0.0, 0.4, 0.7, 1.3, 1.6],
    "n_test": 1000,
    "n_trials": 300, "n_trials_ncal_100000": 60,
    "n_lambda": 400,
    "fd_window": 0.05,
    "seed": 20260820,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" /
                      "wp2_quantized_switchboard.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def Gq(lam, K):
    """Quantized loss-tail: with s = (floor(Phi(x0) K) + 0.5)/K,
    1{s >= lam} iff Phi(x0) >= k0/K, k0 = ceil(lam K - 0.5)."""
    if K == 0:
        return A.G_int(lam)
    k0 = int(np.ceil(lam * K - 0.5))
    if k0 <= 0:
        return A.G_int(0.0)
    if k0 >= K:
        return 0.0
    return A.G_int(k0 / K)


def Rwq(lam, c, K):
    return Gq(lam, K) * A.H_int(c)


def grid_cross(alpha, c, K, n, w_test, lambdas):
    """Population grid crossing with the B/(n+1) charge."""
    S_w = float(n)
    for lam in lambdas:
        if (S_w * Rwq(lam, c, K) + w_test) / (S_w + w_test) <= alpha:
            return float(lam)
    return float(lambdas[-1])


def derived_cell(alpha, K, ncal, gamma, b, lambdas):
    """Quadrature-exact population quantities for the cell."""
    ch = gamma * b
    kap = A.H_int(b) / A.H_int(ch)          # exact; factorization kills FD bias
    wt_o, wt_e = float(np.exp(b * b)), float(np.exp(ch * b))
    lam_o = grid_cross(alpha, b, K, ncal, wt_o, lambdas)
    lam_e = grid_cross(alpha, ch, K, ncal, wt_e, lambdas)
    a_star = (A.H_int(b) - A.H_int(ch)) * Gq(lam_o, K)
    pd_pop = Rwq(lam_e, b, K) - Rwq(lam_o, b, K)
    out = {"lam_o_pop": lam_o, "lam_e_pop": lam_e,
           "kappa_exact": kap, "a_star": a_star, "pd_pop": pd_pop}
    if abs(a_star) > 1e-12:
        out["m_derived"] = pd_pop / (kap * a_star)
        g_ratio = Gq(lam_e, K) / Gq(lam_o, K) if Gq(lam_o, K) > 0 else None
        out["g_ratio"] = g_ratio
    return out


def measured_cell(env, alpha, K, ncal, gamma, b, cfg, n_trials, rng):
    lambdas = np.linspace(0, 1, cfg["n_lambda"])
    ch = gamma * b
    h = cfg["fd_window"]
    pd_, sw_, sh_ = [], [], []
    for _ in range(n_trials):
        cal = env.case_table(rng, ncal)
        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        w = np.exp(env.tilt_logweight(cal.X, b))
        what = np.exp(env.tilt_logweight(cal.X, ch))
        Xt, _ = env.draw_instances(rng, cfg["n_test"], beta=b)
        st = norm.cdf(Xt[:, 0])
        if K:
            st = (np.floor(st * K) + 0.5) / K
        qt = A.g_fn(Xt[:, 0]) * A.h_fn(Xt[:, 1])
        wev = np.exp(env.tilt_logweight(Xt, b))
        wevhat = np.exp(env.tilt_logweight(Xt, ch))
        lam_o = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
        lam_e = crc.lhat_prop2(losses, lambdas, alpha, what, wevhat)
        r_o = float(np.mean(qt * (st >= np.asarray(lam_o))))
        r_e = float(np.mean(qt * (st >= np.asarray(lam_e))))
        pd_.append(r_e - r_o)
        ls = float(np.mean(lam_o))
        wn, hn = w / w.mean(), what / what.mean()
        Llo = ((cal.s >= ls - h) & cal.wrong).astype(float)
        Lhi = ((cal.s >= ls + h) & cal.wrong).astype(float)
        sw_.append(float(np.mean(wn * (Llo - Lhi)) / (2 * h)))
        sh_.append(float(np.mean(hn * (Llo - Lhi)) / (2 * h)))
    n = len(pd_)
    mw, mh = float(np.mean(sw_)), float(np.mean(sh_))
    return {"pd_mean": float(np.mean(pd_)),
            "pd_se": float(np.std(pd_, ddof=1) / np.sqrt(n)),
            "kappa_pred_fd": mw / mh if mh > 0.02 else None}


# ---- Part B: rank-2 quantized curves, deterministic ----------------------

def _g1(u): return 0.45 * (1 - norm.cdf(1.1 * u))
def _h1(v): return 0.10 + 0.55 * norm.cdf(v)
def _g2(u): return 0.5 * np.exp(-((u - 0.6) / 0.35) ** 2)
def _h2(v): return 0.75 * norm.cdf(3.0 * (v - 0.8))


def _tail(f, t):
    v, _ = integrate.quad(lambda u: f(u) * norm.pdf(u), t, 12.0, limit=400)
    return float(v)


def _loc(f, c):
    v, _ = integrate.quad(lambda u: f(u) * norm.pdf(u, loc=c),
                          c - 12.0, c + 12.0, limit=400)
    return float(v)


_T1, _T2 = {}, {}


def R2q(lam, c, K):
    if K == 0:
        t = max(norm.ppf(max(lam, 1e-300)), -12.0)
    else:
        k0 = int(np.ceil(lam * K - 0.5))
        if k0 >= K:
            return 0.0
        t = -12.0 if k0 <= 0 else max(norm.ppf(k0 / K), -12.0)
    key = round(t, 12)
    if key not in _T1:
        _T1[key] = _tail(_g1, t)
        _T2[key] = _tail(_g2, t)
    return _T1[key] * _loc(_h1, c) + _T2[key] * _loc(_h2, c)


def rank2_cell(alpha, K, gamma, b, lambdas, h):
    ch = gamma * b
    lam_o = next((float(l) for l in lambdas if R2q(l, b, K) <= alpha),
                 float(lambdas[-1]))
    lam_e = next((float(l) for l in lambdas if R2q(l, ch, K) <= alpha),
                 float(lambdas[-1]))
    a_star = R2q(lam_o, b, K) - R2q(lam_o, ch, K)
    num = R2q(lam_o - h, b, K) - R2q(lam_o + h, b, K)
    den = R2q(lam_o - h, ch, K) - R2q(lam_o + h, ch, K)
    if abs(a_star) < 1e-12 or den <= 1e-12:
        return None
    kap_fd = num / den
    pd_pop = R2q(lam_e, b, K) - R2q(lam_o, b, K)
    return {"m_pop_fdkappa": pd_pop / (kap_fd * a_star),
            "kappa_fd": kap_fd, "a_star": a_star}


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    st = A.selftest(n=200_000)
    if not st["ok"]:
        raise SystemExit(f"[selftest] analytic exactness FAILED: {st}")
    print("[selftest] analytic rung-1 exactness: PASS")
    check_registration()
    print(f"[wp2qsw] {'PILOT SMOKE' if smoke else 'FULL (pilot-analytic tier)'}")
    b = CONFIG["b_tilt"]
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    cells = []
    for K in CONFIG["Ks"]:
        env = A.AnalyticEnv(quantize=K or None)
        for alpha in CONFIG["alphas"]:
            for ncal in CONFIG["n_cals"]:
                t0 = time.time()
                for gi, gamma in enumerate(CONFIG["gammas"]):
                    n_tr = smoke if smoke is not None else (
                        CONFIG["n_trials_ncal_100000"] if ncal == 100000
                        else CONFIG["n_trials"])
                    rng = np.random.default_rng(
                        [CONFIG["seed"], 400, K, int(alpha * 1000),
                         ncal, gi])
                    der = derived_cell(alpha, K, ncal, gamma, b, lambdas)
                    mea = measured_cell(env, alpha, K, ncal, gamma, b,
                                        CONFIG, n_tr, rng)
                    cell = {"K": K, "alpha": alpha, "n_cal": ncal,
                            "gamma": gamma, **der, **mea}
                    if "m_derived" in der and abs(der["a_star"]) > 1e-12:
                        ka = der["kappa_exact"] * der["a_star"]
                        cell["m_measured"] = mea["pd_mean"] / ka
                        cell["m_measured_se"] = mea["pd_se"] / abs(ka)
                    cells.append(cell)
                sub = [c for c in cells if c["K"] == K and c["alpha"] == alpha
                       and c["n_cal"] == ncal and "m_measured" in c]
                if sub:
                    mm = float(np.mean([c["m_measured"] for c in sub]))
                    md = float(np.mean([c["m_derived"] for c in sub]))
                    print(f"[wp2qsw] K={K:<3} alpha={alpha:<5} n_cal={ncal:<7}"
                          f" m_measured={mm:.3f} m_derived={md:.3f}"
                          f" ({time.time() - t0:.0f}s)", flush=True)
    rank2 = []
    for K in CONFIG["Ks"]:
        for alpha in CONFIG["alphas"]:
            vals = []
            for gamma in CONFIG["gammas"]:
                r = rank2_cell(alpha, K, gamma, b, lambdas,
                               CONFIG["fd_window"])
                if r:
                    rank2.append({"K": K, "alpha": alpha, "gamma": gamma, **r})
                    vals.append(r["m_pop_fdkappa"])
            if vals:
                print(f"[wp2qsw/B] K={K:<3} alpha={alpha:<5} rank2 "
                      f"m_pop(fd kappa)={float(np.mean(vals)):.3f}", flush=True)

    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2qsw_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps(
        {"cells": cells, "rank2": rank2}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
