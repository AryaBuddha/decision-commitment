"""WP2 Phase 1: gated-tier confirmation of the discrete-crossing
mechanism. EVIDENCE TIER.

The quantized analytic world (wp2qsw) tests whether m is DERIVED by the
plateau arithmetic in a world where everything is exact. This experiment
is the same test on the four gated environments: per temper cell,
Identity 1 gives

    pd = a(lam_e) + b_e - b_o        (all three terms measured per trial
                                      on an INDEPENDENT source draw)

so the identity-implied amplification is

    m_derived = pd_identity / (kappa_pred * a(lam*)),

computable without ever looking at the evaluation windows. If the
mechanism is right, m_derived reproduces the measured
m_cell = pd_eval / (kappa_pred * a(lam*)) cell by cell, and the
environment ordering of m at alpha 0.05 (logit < claims < tickets) is a
property of their loss-curve plateau geometry, not of anything else.

Run:   python experiments/wp2_mechanism_confirm/run.py
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

from cus import crc, tests                        # noqa: E402
from cus.envs.claims import ClaimsEnv             # noqa: E402
from cus.envs.claims_logit import ClaimsLogitEnv  # noqa: E402
from cus.envs.spike import SpikeEnv               # noqa: E402
from cus.envs.family import GenEnv                # noqa: E402


CONFIG = {
    "experiment": "wp2_mechanism_confirm",
    "environments": {
        "claims": {"tilt_feature": "inconsistency", "betas": [3.0, 5.0]},
        "claims_logit": {"tilt_feature": "inconsistency", "betas": [3.0, 5.0]},
        "spike": {"tilt_feature": "b", "betas": [3.854, 7.008]},
        "tickets": {"tilt_feature": "frustration", "betas": [3.589, 5.982]},
    },
    "alphas": [0.05, 0.10],
    "temper_gammas": [0.0, 0.25, 0.5, 0.75, 0.8, 1.0, 1.25, 1.5],
    "n_cal": 1000, "n_eval": 1000, "n_ind": 4000,
    "n_trials": 300, "n_lambda": 400,
    "fd_window": 0.05, "min_abs_a": 0.003, "min_slope_what": 0.02,
    "seed": 20260821,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" /
                      "wp2_mechanism_confirm.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def make_env(name):
    if name == "claims":
        return ClaimsEnv.induce()
    if name == "claims_logit":
        return ClaimsLogitEnv.induce()
    if name == "spike":
        return SpikeEnv.induce()
    return GenEnv.induce(name)


def run_cell(env, feat, alpha, gamma, beta, cfg, n_trials, rng):
    lambdas = np.linspace(0, 1, cfg["n_lambda"])
    h = cfg["fd_window"]
    pd_eval_, pd_id_, a_star_, sw_, sh_ = [], [], [], [], []
    for _ in range(n_trials):
        cal = env.case_table(rng, cfg["n_cal"])
        ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feat)
        ind = env.case_table(rng, cfg["n_ind"])
        w = np.exp(env.tilt_logweight(cal.X, beta, feat))
        wev = np.exp(env.tilt_logweight(ev.X, beta, feat))
        what, wevhat = w ** gamma, wev ** gamma
        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        lam_o = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
        lam_e = crc.lhat_prop2(losses, lambdas, alpha, what, wevhat)
        lo, le = float(np.mean(lam_o)), float(np.mean(lam_e))
        ro = float(((ev.s >= np.asarray(lam_o)) & ev.wrong).mean())
        re = float(((ev.s >= np.asarray(lam_e)) & ev.wrong).mean())
        pd_eval_.append(re - ro)
        # identity terms on the independent draw
        w_ind = np.exp(env.tilt_logweight(ind.X, beta, feat))
        wn_i = w_ind / w_ind.mean()
        hn_i = wn_i ** 0.0 if gamma == 0.0 else (w_ind ** gamma) / (w_ind ** gamma).mean()
        Le = ((ind.s >= le) & ind.wrong).astype(float)
        Lo = ((ind.s >= lo) & ind.wrong).astype(float)
        a_le = float(np.mean((wn_i - hn_i) * Le))
        b_e = float(np.mean(hn_i * Le))          # R_what at lam_e
        b_o = float(np.mean(wn_i * Lo))          # R_w at lam_o
        pd_id_.append(a_le + (b_e - alpha) - (b_o - alpha))
        # a at lambda* and FD slopes on the CAL draw (deployed instrument)
        wn, hn = w / w.mean(), what / what.mean()
        L = ((cal.s >= lo) & cal.wrong).astype(float)
        a_star_.append(float(np.mean((wn - hn) * L)))
        Llo = ((cal.s >= lo - h) & cal.wrong).astype(float)
        Lhi = ((cal.s >= lo + h) & cal.wrong).astype(float)
        sw_.append(float(np.mean(wn * (Llo - Lhi)) / (2 * h)))
        sh_.append(float(np.mean(hn * (Llo - Lhi)) / (2 * h)))
    n = len(pd_eval_)
    mw, mh = float(np.mean(sw_)), float(np.mean(sh_))
    return {
        "alpha": alpha, "gamma": gamma, "beta": beta,
        "pd_eval_mean": float(np.mean(pd_eval_)),
        "pd_eval_se": float(np.std(pd_eval_, ddof=1) / np.sqrt(n)),
        "pd_id_mean": float(np.mean(pd_id_)),
        "pd_id_se": float(np.std(pd_id_, ddof=1) / np.sqrt(n)),
        "a_star_mean": float(np.mean(a_star_)),
        "kappa_pred": mw / mh if mh > CONFIG["min_slope_what"] else None,
        "slope_what": mh,
    }


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration()
    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    print(f"[wp2mc] {'PILOT SMOKE' if smoke else 'REAL (evidence tier)'}")
    cells = []
    for ei, (ename, espec) in enumerate(CONFIG["environments"].items()):
        env = make_env(ename)
        for ai, alpha in enumerate(CONFIG["alphas"]):
            t0 = time.time()
            for gi, gamma in enumerate(CONFIG["temper_gammas"]):
                for bi, beta in enumerate(espec["betas"]):
                    rng = np.random.default_rng(
                        [CONFIG["seed"], 500, ei, ai, gi, bi])
                    c = run_cell(env, espec["tilt_feature"], alpha, gamma,
                                 beta, CONFIG, n_trials, rng)
                    c["env"] = ename
                    cells.append(c)
            guard = [c for c in cells
                     if c["env"] == ename and c["alpha"] == alpha
                     and c["kappa_pred"] is not None
                     and abs(c["a_star_mean"]) >= CONFIG["min_abs_a"]
                     and c["slope_what"] >= CONFIG["min_slope_what"]]
            if len(guard) >= 4:
                x = np.array([c["pd_id_mean"] / (c["kappa_pred"] * c["a_star_mean"])
                              for c in guard])
                y = np.array([c["pd_eval_mean"] / (c["kappa_pred"] * c["a_star_mean"])
                              for c in guard])
                slope = float(x @ y / (x @ x))
                r = y - slope * x
                se = float(np.sqrt((r @ r) / (len(x) - 1) / (x @ x)))
                ss = float(((y - y.mean()) ** 2).sum())
                r2 = 1 - float(r @ r) / ss if ss > 0 else None
                print(f"[wp2mc] {ename:<12} alpha={alpha:<5} "
                      f"m_derived-vs-measured slope={slope:.3f}±{se:.3f} "
                      f"R2={r2 if r2 is None else round(r2, 3)} n={len(x)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2mc_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps({"cells": cells}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
