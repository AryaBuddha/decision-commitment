"""WP2 Phase 0: the two cheap closures. EVIDENCE TIER.

Part L, the A1 locality closure. Block A's unification test found the
lambda*-referenced coordinate a(lambda*) saturates when an arm's operating
threshold sits far from the oracle's (compliance sweep: a flat at 0.017
while excess climbed to 0.038; 4 of 42 archived cells off-curve, all
above). The exact decomposition at the arm's OWN threshold,

    excess(lam_u) = E_P0[(w - 1) L(lam_u)] + (rho0(lam_u) - alpha)
                  =        a_own           +        b_own,

is algebraically exact in population for the unweighted arm; measuring
both right-hand terms on an INDEPENDENT source sample per trial makes it
a testable reconstruction with unit slope and no m amplification. If
amplification appears even in this coordinate, m is not an artifact of
lambda*-referencing and every Phase 1 mechanism needs rework.

Part R, the ledgered B4 follow-up. Block B4's corrected diagnosis says
the sklearn L2 default (C = 1.0) shrinks extreme log-odds in the ratio
fit and drives the top-tilt non-equivalence of correctly specified
estimators on the two blind-driver environments. Sweep C in
{1, 10, 100, unregularized} at the affected cells. If unregularized
logistic does NOT recover oracle-equivalence, the B4 diagnosis is
refuted verbatim and the non-equivalence needs a third owner.

Run:   python experiments/wp2_phase0_closures/run.py
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

from cus import crc, shift, tests            # noqa: E402
from cus.envs.claims import ClaimsEnv        # noqa: E402
from cus.envs.family import GenEnv           # noqa: E402


CONFIG = {
    "experiment": "wp2_phase0_closures",
    "alpha": 0.10,
    "n_cal": 1000, "n_eval": 1000, "n_own": 4000, "n_fit": 1000,
    "n_trials": 300, "n_lambda": 400,
    "locality_cells": [
        {"env": "compliance", "feature": "redline_density",
         "betas": [1.101, 2.202, 3.304, 4.405, 5.506, 6.607]},
        {"env": "claims", "feature": "severity",
         "betas": [1.49, 3.406, 6.163]},
        {"env": "moderation", "feature": "toxicity",
         "betas": [1.641, 3.283, 4.924]},
        {"env": "moderation", "feature": "sarcasm",
         "betas": [2.483, 4.8, 6.581]},
    ],
    "reg_cells": [
        {"env": "tickets", "feature": "frustration",
         "betas": [5.982, 7.178]},
        {"env": "compliance", "feature": "redline_density",
         "betas": [5.506, 6.607]},
    ],
    "reg_Cs": [1.0, 10.0, 100.0, None],
    "weight_clip": [0.01, 0.99],
    "z_one_sided": 1.645, "delta_oracle": 0.005,
    "residual_tol_floor": 0.0075,
    "seed": 20260820,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp2_phase0_closures.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def make_env(name):
    return ClaimsEnv.induce() if name == "claims" else GenEnv.induce(name)


def locality_cell(env, ename, feat, beta, cfg, n_trials, rng):
    """Unweighted arm; per-trial own-threshold decomposition on an
    independent source sample."""
    lambdas = np.linspace(0, 1, cfg["n_lambda"])
    alpha = cfg["alpha"]
    exc_, a_own_, b_own_ = [], [], []
    for _ in range(n_trials):
        cal = env.case_table(rng, cfg["n_cal"])
        ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feat)
        ind = env.case_table(rng, cfg["n_own"])          # independent P0 draw
        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        lam_u = crc.lhat_unweighted(losses, lambdas, alpha)
        exc_.append(float(((ev.s >= lam_u) & ev.wrong).mean()) - alpha)
        w_ind = np.exp(env.tilt_logweight(ind.X, beta, feat))
        wn = w_ind / w_ind.mean()
        L = ((ind.s >= lam_u) & ind.wrong).astype(float)
        a_own_.append(float(np.mean((wn - 1.0) * L)))
        b_own_.append(float(L.mean()) - alpha)
    n = len(exc_)

    def ms(v):
        return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(n)))
    exc, exc_se = ms(exc_)
    a_own, a_se = ms(a_own_)
    b_own, b_se = ms(b_own_)
    resid = exc - (a_own + b_own)
    return {"part": "locality", "env": ename, "feature": feat, "beta": beta,
            "excess_mean": exc, "excess_se": exc_se,
            "a_own_mean": a_own, "a_own_se": a_se,
            "b_own_mean": b_own, "b_own_se": b_se,
            "residual": resid}


def reg_cell(env, ename, feat, beta, C, cfg, n_trials, rng):
    """Correctly specified estimated arm at swept regularization C."""
    lambdas = np.linspace(0, 1, cfg["n_lambda"])
    alpha = cfg["alpha"]
    paired, ro_, re_, ess_ = [], [], [], []
    for _ in range(n_trials):
        cal = env.case_table(rng, cfg["n_cal"])
        ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feat)
        w = np.exp(env.tilt_logweight(cal.X, beta, feat))
        wev = np.exp(env.tilt_logweight(ev.X, beta, feat))
        Xs, _ = env.draw_instances(rng, cfg["n_fit"])
        Xt, _ = env.draw_instances(rng, cfg["n_fit"], beta, feat)
        w_fn = shift.fit_ratio(Xs, Xt, method="logistic",
                               clip=tuple(cfg["weight_clip"]), C=C)
        what, wevhat = w_fn(cal.X), w_fn(ev.X)
        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        lam_o = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
        ro = float(((ev.s >= np.asarray(lam_o)) & ev.wrong).mean())
        lam_e = crc.lhat_prop2(losses, lambdas, alpha, what, wevhat)
        re = float(((ev.s >= np.asarray(lam_e)) & ev.wrong).mean())
        paired.append(re - ro)
        ro_.append(ro)
        re_.append(re)
        ess_.append(crc.effective_sample_size(what))
    n = len(paired)
    dm = float(np.mean(paired))
    dse = float(np.std(paired, ddof=1) / np.sqrt(n))
    z = cfg["z_one_sided"]
    return {"part": "regularization", "env": ename, "feature": feat,
            "beta": beta, "C": "none" if C is None else C,
            "paired_diff_mean": dm, "paired_diff_se": dse,
            "oracle_equiv": bool(-cfg["delta_oracle"] <= dm - z * dse
                                 and dm + z * dse <= cfg["delta_oracle"]),
            "risk_oracle_mean": float(np.mean(ro_)),
            "risk_est_mean": float(np.mean(re_)),
            "ess_mean": float(np.mean(ess_))}


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration()
    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    print(f"[wp2p0c] {'PILOT SMOKE' if smoke else 'REAL (evidence tier)'}")

    rows = []
    envs = {}
    for li, spec in enumerate(CONFIG["locality_cells"]):
        ename = spec["env"]
        envs.setdefault(ename, make_env(ename))
        for bi, beta in enumerate(spec["betas"]):
            rng = np.random.default_rng([CONFIG["seed"], 70, li, bi])
            t0 = time.time()
            c = locality_cell(envs[ename], ename, spec["feature"], beta,
                              CONFIG, n_trials, rng)
            rows.append(c)
            tol = max(3 * c["excess_se"], CONFIG["residual_tol_floor"])
            print(f"[wp2p0c/L] {ename:<11} {spec['feature']:<16} beta={beta:<6}"
                  f" exc={c['excess_mean']:+.4f} a_own={c['a_own_mean']:+.4f}"
                  f" b_own={c['b_own_mean']:+.4f} resid={c['residual']:+.4f}"
                  f" (tol {tol:.4f}) ({time.time() - t0:.0f}s)", flush=True)
    # own-coordinate slope across locality cells
    x = np.array([c["a_own_mean"] for c in rows if c["part"] == "locality"])
    y = np.array([c["excess_mean"] - c["b_own_mean"]
                  for c in rows if c["part"] == "locality"])
    slope = float(x @ y / (x @ x))
    r = y - slope * x
    slope_se = float(np.sqrt((r @ r) / (len(x) - 1) / (x @ x)))
    print(f"[wp2p0c/L] own-threshold slope {slope:.3f}±{slope_se:.3f}")

    for ri, spec in enumerate(CONFIG["reg_cells"]):
        ename = spec["env"]
        envs.setdefault(ename, make_env(ename))
        for bi, beta in enumerate(spec["betas"]):
            for ci, C in enumerate(CONFIG["reg_Cs"]):
                rng = np.random.default_rng([CONFIG["seed"], 80, ri, bi, ci])
                t0 = time.time()
                c = reg_cell(envs[ename], ename, spec["feature"], beta, C,
                             CONFIG, n_trials, rng)
                rows.append(c)
                print(f"[wp2p0c/R] {ename:<11} beta={beta:<6} C={c['C']:<5}"
                      f" pd={c['paired_diff_mean']:+.4f}±{c['paired_diff_se']:.4f}"
                      f" equiv={c['oracle_equiv']} ess={c['ess_mean']:.0f}"
                      f" ({time.time() - t0:.0f}s)", flush=True)

    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2p0c_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps(
        {"rows": rows, "own_slope": slope, "own_slope_se": slope_se},
        indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
