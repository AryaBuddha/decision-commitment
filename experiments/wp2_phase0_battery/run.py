"""WP2 Phase 0: the confirmatory synthesis battery. EVIDENCE TIER.

The affine law, per-cell kappa_pred, and m(alpha) were DISCOVERED through
inverted predictions and exploratory analysis (Blocks A-C). By this
project's rules they are not results until they survive a registration
written in advance and fresh data. This battery re-derives every headline
quantity on FRESH SEEDS (seed 20260820: new instance draws, new trial
streams; the frozen environments themselves are untouched) across four
environments and an alpha grid whose 0.02 endpoint the law has never seen.

Law under test, per cell:  paired_diff = m(alpha) * kappa_pred * a
  a           = E_P0[(w - w_hat) L(lambda*)], signed aligned error
  kappa_pred  = local loss-curve slope ratio at lambda*, ratio of
                trial-mean FD slopes (the A3/Block-C machinery)
  m(alpha)    = through-origin regression of measured kappa (pd/a) on
                kappa_pred over guarded TEMPER cells, per env-alpha
  b(env)      = oracle plateau margin, mean oracle excess over temper cells

Run:   python experiments/wp2_phase0_battery/run.py
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

from cus import crc, shift, tests                 # noqa: E402
from cus.envs.claims import ClaimsEnv             # noqa: E402
from cus.envs.claims_logit import ClaimsLogitEnv  # noqa: E402
from cus.envs.spike import SpikeEnv               # noqa: E402
from cus.envs.family import GenEnv                # noqa: E402


CONFIG = {
    "experiment": "wp2_phase0_battery",
    "environments": {
        "claims": {"tilt_feature": "inconsistency", "tilt_dim": 8,
                   "betas": [3.0, 5.0]},
        "claims_logit": {"tilt_feature": "inconsistency", "tilt_dim": 8,
                         "betas": [3.0, 5.0]},
        "spike": {"tilt_feature": "b", "tilt_dim": 1,
                  "betas": [3.854, 7.008]},
        "tickets": {"tilt_feature": "frustration", "tilt_dim": 7,
                    "betas": [3.589, 5.982]},
    },
    "alphas": [0.02, 0.05, 0.10, 0.15],
    "temper_gammas": [0.0, 0.25, 0.5, 0.75, 0.8, 1.0, 1.25, 1.5],
    "axes": {"deprivation": {"view_rho": [0.0, 0.35]},
             "starvation": {"n_fit": [50, 250]}},
    "n_cal": 1000, "n_eval": 1000, "default_n_fit": 1000,
    "n_trials": 300, "n_lambda": 400,
    "weight_estimator": "logistic", "weight_clip": [0.01, 0.99],
    "ratio_C": 1.0,
    "fd_window": 0.05, "min_abs_a": 0.003, "min_slope_what": 0.02,
    "degenerate_pin_frac": 0.5,
    "noise_floor_mult": 2.0,
    "z_one_sided": 1.645, "delta_control": 0.005,
    "residual_tol_floor": 0.0075,
    "seed": 20260820,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp2_phase0_battery.json").read_text())
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


def make_view(kind, setting, tilt_dim, n_features, rng):
    """The ratio estimator's degraded view (fit AND application time)."""
    if kind != "deprivation":
        return lambda X: X
    rho = setting["view_rho"]
    if rho == 0.0:
        keep = [j for j in range(n_features) if j != tilt_dim]
        return lambda X: X[:, keep]

    def view(X):
        X = X.copy()
        X[:, tilt_dim] = rho * X[:, tilt_dim] + \
            np.sqrt(1 - rho ** 2) * rng.standard_normal(len(X))
        return X
    return view


def run_cell(env, ename, espec, alpha, kind, setting, beta, cfg, n_trials,
             rng):
    lambdas = np.linspace(0, 1, cfg["n_lambda"])
    feat = espec["tilt_feature"]
    h = cfg["fd_window"]
    n_feats = None
    aligned, paired, ro_, re_, sw_, sh_ = [], [], [], [], [], []
    pin_lo, pin_hi, ess_ = [], [], []
    for _ in range(n_trials):
        cal = env.case_table(rng, cfg["n_cal"])
        ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feat)
        w = np.exp(env.tilt_logweight(cal.X, beta, feat))
        wev = np.exp(env.tilt_logweight(ev.X, beta, feat))
        if kind == "temper":
            g = setting["gamma"]
            what, wevhat = w ** g, wev ** g
        else:
            if n_feats is None:
                n_feats = cal.X.shape[1]
            n_fit = setting.get("n_fit", cfg["default_n_fit"])
            view = make_view(kind, setting, espec["tilt_dim"], n_feats, rng)
            Xs, _ = env.draw_instances(rng, n_fit)
            Xt, _ = env.draw_instances(rng, n_fit, beta, feat)
            w_fn = shift.fit_ratio(view(Xs), view(Xt),
                                   method=cfg["weight_estimator"],
                                   clip=tuple(cfg["weight_clip"]),
                                   C=cfg["ratio_C"])
            what, wevhat = w_fn(view(cal.X)), w_fn(view(ev.X))
        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        lam_o = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
        ls = float(np.mean(lam_o))
        ro = float(((ev.s >= np.asarray(lam_o)) & ev.wrong).mean())
        lam_e = crc.lhat_prop2(losses, lambdas, alpha, what, wevhat)
        le = float(np.mean(lam_e))
        re = float(((ev.s >= np.asarray(lam_e)) & ev.wrong).mean())
        pin_lo.append(le <= lambdas[0] + 1e-12)
        pin_hi.append(le >= lambdas[-1] - 1e-12)
        wn, hn = w / w.mean(), what / what.mean()
        L = ((cal.s >= ls) & cal.wrong).astype(float)
        aligned.append(float(np.mean((wn - hn) * L)))
        paired.append(re - ro)
        ro_.append(ro)
        re_.append(re)
        Llo = ((cal.s >= ls - h) & cal.wrong).astype(float)
        Lhi = ((cal.s >= ls + h) & cal.wrong).astype(float)
        sw_.append(float(np.mean(wn * (Llo - Lhi)) / (2 * h)))
        sh_.append(float(np.mean(hn * (Llo - Lhi)) / (2 * h)))
        ess_.append(crc.effective_sample_size(what))
    n = len(paired)
    mw, mh = float(np.mean(sw_)), float(np.mean(sh_))
    a_m = float(np.mean(aligned))
    a_se = float(np.std(aligned, ddof=1) / np.sqrt(n))
    pd_m = float(np.mean(paired))
    pd_se = float(np.std(paired, ddof=1) / np.sqrt(n))
    return {
        "env": ename, "alpha": alpha, "kind": kind,
        "setting": json.dumps(setting), "beta": beta,
        "aligned_mean": a_m, "aligned_se": a_se,
        "paired_diff_mean": pd_m, "paired_diff_se": pd_se,
        "risk_oracle_mean": float(np.mean(ro_)),
        "risk_oracle_se": float(np.std(ro_, ddof=1) / np.sqrt(n)),
        "risk_est_mean": float(np.mean(re_)),
        "risk_est_se": float(np.std(re_, ddof=1) / np.sqrt(n)),
        "kappa_pred": mw / mh if mh > CONFIG["min_slope_what"] else None,
        "slope_w": mw, "slope_what": mh,
        "pin_lo_frac": float(np.mean(pin_lo)),
        "pin_hi_frac": float(np.mean(pin_hi)),
        "ess_mean": float(np.mean(ess_)),
    }


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else 0.0


def analyse(cells, cfg, ename, alpha):
    """Per env-alpha: b, m fit on guarded temper cells, collapse residuals."""
    temper = [c for c in cells if c["kind"] == "temper"]
    b_env = float(np.mean([c["risk_oracle_mean"] for c in temper])) - alpha
    guard = [c for c in temper
             if c["kappa_pred"] is not None
             and abs(c["aligned_mean"]) >= cfg["min_abs_a"]
             and c["slope_what"] >= cfg["min_slope_what"]
             and c["pin_lo_frac"] <= cfg["degenerate_pin_frac"]
             and c["pin_hi_frac"] <= cfg["degenerate_pin_frac"]]
    out = {"env": ename, "alpha": alpha, "b_env": b_env,
           "n_guarded": len(guard)}
    if len(guard) >= 4:
        kp = np.array([c["kappa_pred"] for c in guard])
        km = np.array([c["paired_diff_mean"] / c["aligned_mean"]
                       for c in guard])
        m = float(kp @ km / (kp @ kp))
        r = km - m * kp
        out["m"] = m
        out["m_se"] = float(np.sqrt((r @ r) / (len(kp) - 1) / (kp @ kp)))
        ss = float(((km - km.mean()) ** 2).sum())
        out["m_r2"] = 1 - float(r @ r) / ss if ss > 0 else None
        out["kp_min"] = float(kp.min())
        out["kp_max"] = float(kp.max())
    else:
        m = None
        out["m"] = out["m_se"] = out["m_r2"] = None
    # collapse residuals over ALL cells against the fitted temper law
    n_pass = n_test = 0
    for c in cells:
        if m is None or c["kappa_pred"] is None:
            continue
        tol = max(3 * c["paired_diff_se"], cfg["residual_tol_floor"])
        e = abs(c["paired_diff_mean"] - m * c["kappa_pred"] * c["aligned_mean"])
        n_test += 1
        n_pass += bool(e <= tol)
    out["residual_pass"] = n_pass
    out["residual_tested"] = n_test
    # rank collapse over cells with |a| above the noise floor (F13 lesson)
    live = [c for c in cells
            if abs(c["aligned_mean"]) >= cfg["noise_floor_mult"] * c["aligned_se"]]
    if len(live) >= 5:
        out["spearman"] = spearman(
            np.array([c["aligned_mean"] for c in live]),
            np.array([c["paired_diff_mean"] for c in live]))
        out["n_rank_cells"] = len(live)
    else:
        out["spearman"] = None
        out["n_rank_cells"] = len(live)
    return out


def iter_cells(cfg, espec):
    for gi, g in enumerate(cfg["temper_gammas"]):
        for bi, beta in enumerate(espec["betas"]):
            yield "temper", {"gamma": g}, beta, (0, gi, bi)
    for ai, (axis, spec) in enumerate(cfg["axes"].items()):
        key = list(spec.keys())[0]
        for si, val in enumerate(spec[key]):
            for bi, beta in enumerate(espec["betas"]):
                yield axis, {key: val}, beta, (1 + ai, si, bi)


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration()
    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    print(f"[wp2p0] {'PILOT SMOKE' if smoke else 'REAL (evidence tier)'}"
          f" fresh seed {CONFIG['seed']}")
    all_cells, analyses = [], []
    for ei, (ename, espec) in enumerate(CONFIG["environments"].items()):
        env = make_env(ename)
        for ai, alpha in enumerate(CONFIG["alphas"]):
            t0 = time.time()
            cells = []
            for kind, setting, beta, idx in iter_cells(CONFIG, espec):
                rng = np.random.default_rng(
                    [CONFIG["seed"], ei, ai, *idx])
                c = run_cell(env, ename, espec, alpha, kind, setting,
                             beta, CONFIG, n_trials, rng)
                cells.append(c)
            an = analyse(cells, CONFIG, ename, alpha)
            analyses.append(an)
            all_cells.extend(cells)
            m_str = "None" if an["m"] is None else f"{an['m']:.3f}±{an['m_se']:.3f}"
            print(f"[wp2p0] {ename:<12} alpha={alpha:<5} b={an['b_env']:+.4f} "
                  f"m={m_str} r2={an['m_r2'] if an['m_r2'] is None else round(an['m_r2'], 3)} "
                  f"guarded={an['n_guarded']} resid={an['residual_pass']}/{an['residual_tested']} "
                  f"rho={an['spearman'] if an['spearman'] is None else round(an['spearman'], 3)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2p0_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps(
        {"cells": all_cells, "analyses": analyses}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
