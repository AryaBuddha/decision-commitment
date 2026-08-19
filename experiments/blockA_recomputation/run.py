"""Block A: recomputation from archived artifacts (NEXT_BLOCK.md).

No new sweeps. Three parts, one registration, one config hash:

  a1  Unification test. Replay every archived evidence-tier
      unweighted-arm cell (7 levels x 5 sweeps; all 12 tilt-location
      cells across both environments and all four tilt features) with
      identical rng streams, compute a = E_P0[(w_norm - 1) L(lambda*)],
      assert the replayed unweighted risk reproduces the archived mean
      (drift <= 1e-9), and judge each cell against the ARCHIVED collapse
      isotonic of its environment (fit not refit). Cells whose a falls
      beyond the archived knot range are judged against the linear
      extension through the end knot at the archived pooled slope, with
      a wider tolerance, and labelled extension cells.
  a2  Slope anatomy on the archived 290 collapse cells: sign-split
      through-origin slopes and near-origin local slopes per environment
      (pure arithmetic on cells.json; nothing replayed).
  a3  Per-cell kappa. Replay each collapse battery (58 cells x 5 envs)
      with identical streams, estimate the local loss-curve slopes at
      lambda* under the true and estimated weightings by central
      difference (window h registered), form kappa_pred per cell, and
      test measured excess/a against it. Includes the TOST equivalence
      analysis that replaces the retracted 'statistically
      indistinguishable' slope claim.

Run:  python experiments/blockA_recomputation/run.py --part a1
      python experiments/blockA_recomputation/run.py --part a2
      python experiments/blockA_recomputation/run.py --part a3 --env claims
      python experiments/blockA_recomputation/run.py --part a3-analyze
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "gates"))

from cus import crc, shift                      # noqa: E402
from cus.envs.claims import ClaimsEnv           # noqa: E402
from cus.envs.family import GenEnv, SPECS       # noqa: E402
from run_gates_family import BETA_GRIDS         # noqa: E402


CONFIG = {
    "experiment": "blockA_recomputation",
    "drift_tolerance": 1e-9,
    "a1": {
        "sources": {
            "sweeps": {
                "claims": {"artifact": "wp1c_ab56864e6cfb3400", "seed": 20260818,
                           "feature": "inconsistency",
                           "betas": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
                "tickets": {"artifact": "wp1f_tickets_d8b48cff3ab2e336", "seed": 20260819},
                "fraud": {"artifact": "wp1f_fraud_2f83073cb88bea1e", "seed": 20260819},
                "moderation": {"artifact": "wp1f_moderation_bc612e3b59311f78", "seed": 20260819},
                "compliance": {"artifact": "wp1f_compliance_99e91888f277db19", "seed": 20260819},
            },
            "tilt_location": {
                "claims": {"artifact": "wp1tl_3879b17b6fa4ae6f", "seed": 20260819,
                           "rng_base": 0, "features": ["inconsistency", "severity"],
                           "levels": [[2.0, 1.49], [4.0, 3.406], [6.0, 6.163]]},
                "moderation": {"artifact": "wp1tlm_bf521b1f263f8cc8", "seed": 20260819,
                               "rng_base": 40, "features": ["toxicity", "sarcasm"],
                               "levels": [[1.641, 2.483], [3.283, 4.800], [4.924, 6.581]]},
            },
            "collapse_curves": {
                "claims": "wp1mc_56704982681d6960",
                "tickets": "wp1mf_tickets_85e921864acbedcc",
                "fraud": "wp1mf_fraud_047610e36449b1c7",
                "moderation": "wp1mf_moderation_037ea765a0016581",
                "compliance": "wp1mf_compliance_c3609a222408ec0f",
            },
        },
        "pooled_slopes": {"claims": 0.862, "tickets": 0.963, "fraud": 0.741,
                          "moderation": 0.939, "compliance": 0.798},
        "in_range_tol": "max(3*SE_cell, 0.0075)",
        "extension_tol": "max(3*SE_cell, 0.01)",
    },
    "a2": {
        "local_windows": [0.005, 0.01],
    },
    "a3": {
        "fd_window": 0.05,
        "fd_window_sensitivity": [0.03, 0.08],
        "min_abs_a": 0.003,
        "min_slope_what": 0.02,
        "regression": "through-origin OLS of kappa_meas on kappa_pred",
        "tost_margin": 0.25,
    },
    "n_lambda": 400,
    "alpha": 0.10,
    "seed_note": "all replays use the archived experiments' own seeds",
}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration() -> None:
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "blockA_recomputation.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} does not match registered "
                         f"{reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def get_env(name: str):
    return ClaimsEnv.induce() if name == "claims" else GenEnv.induce(name)


def yardstick(envname: str):
    d = ROOT / "artifacts" / CONFIG["a1"]["sources"]["collapse_curves"][envname]
    ana = json.loads((d / "collapse.json").read_text())
    pts = sorted(ana["isotonic_fit"])
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return xs, ys


def judge(a, excess, se, envname):
    xs, ys = yardstick(envname)
    slope = CONFIG["a1"]["pooled_slopes"][envname]
    if xs[0] <= a <= xs[-1]:
        pred = float(np.interp(a, xs, ys))
        tol = max(3 * se, 0.0075)
        return "in_range", pred, tol, abs(excess - pred) <= tol
    if a > xs[-1]:
        pred = float(ys[-1] + slope * (a - xs[-1]))
    else:
        pred = float(ys[0] + slope * (a - xs[0]))
    tol = max(3 * se, 0.01)
    return "extension", pred, tol, abs(excess - pred) <= tol


def a_unweighted(env, cal, ev, beta, feature, lambdas, alpha):
    w_cal = np.exp(env.tilt_logweight(cal.X, beta, feature))
    w_ev = np.exp(env.tilt_logweight(ev.X, beta, feature))
    losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
    lam_o = crc.lhat_prop2(losses, lambdas, alpha, w_cal, w_ev)
    lam_star = float(np.mean(lam_o))
    lam_u = crc.lhat_unweighted(losses, lambdas, alpha)
    risk_u = float(((ev.s >= lam_u) & ev.wrong).mean())
    wn = w_cal / w_cal.mean()
    L = ((cal.s >= lam_star) & cal.wrong).astype(float)
    return float(np.mean((wn - 1.0) * L)), risk_u


def part_a1():
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    alpha = CONFIG["alpha"]
    out = []

    for envname, src in CONFIG["a1"]["sources"]["sweeps"].items():
        env = get_env(envname)
        arch = json.loads((ROOT / "artifacts" / src["artifact"] / "results.json").read_text())
        unw = {r["beta"]: r for r in arch if r["arm"] == "unweighted"}
        betas = src.get("betas", BETA_GRIDS.get(envname))
        feature = src.get("feature", SPECS[envname].primary_tilt if envname != "claims" else None)
        n_fit = 1000
        for li, beta in enumerate(betas):
            rng = np.random.default_rng([src["seed"], li])
            avals, risks = [], []
            for _ in range(500):
                cal = env.case_table(rng, 1000)
                ev = env.case_table(rng, 1000, beta=beta, feature=feature)
                env.draw_instances(rng, n_fit)
                env.draw_instances(rng, n_fit, beta, feature)
                a, r = a_unweighted(env, cal, ev, beta, feature, lambdas, alpha)
                avals.append(a); risks.append(r)
            drift = abs(float(np.mean(risks)) - unw[beta]["marginal_risk_mean"])
            if drift > CONFIG["drift_tolerance"]:
                raise SystemExit(f"[a1] DRIFT {drift:.2e} at {envname} sweep beta={beta}")
            a_m = float(np.mean(avals))
            a_se = float(np.std(avals, ddof=1) / np.sqrt(len(avals)))
            excess = unw[beta]["marginal_risk_mean"] - alpha
            se = unw[beta]["marginal_risk_se"]
            zone, pred, tol, ok = judge(a_m, excess, se, envname)
            out.append({"source": "sweep", "env": envname, "feature": feature,
                        "beta": beta, "a": a_m, "a_se": a_se, "excess": excess,
                        "se": se, "zone": zone, "curve_pred": pred, "tol": tol,
                        "on_curve": bool(ok), "cross_direction": False})
            print(f"[a1] sweep {envname:<11} beta={beta:<7} a={a_m:+.4f} "
                  f"excess={excess:+.4f} pred={pred:+.4f} [{zone:<9}] "
                  f"{'ON' if ok else 'OFF'}")

    for envname, src in CONFIG["a1"]["sources"]["tilt_location"].items():
        env = get_env(envname)
        arch = json.loads((ROOT / "artifacts" / src["artifact"] / "results.json").read_text())
        battery_feat = "inconsistency" if envname == "claims" else "toxicity"
        for li, level in enumerate(src["levels"]):
            for fi, feature in enumerate(src["features"]):
                beta = level[fi]
                ref = [r for r in arch if r["arm"] == "unweighted"
                       and r["feature"] == feature and r["beta"] == beta][0]
                rng = np.random.default_rng([src["seed"], src["rng_base"] + li, fi])
                avals, risks = [], []
                for _ in range(500):
                    cal = env.case_table(rng, 1000)
                    ev = env.case_table(rng, 1000, beta=beta, feature=feature)
                    a, r = a_unweighted(env, cal, ev, beta, feature, lambdas, alpha)
                    avals.append(a); risks.append(r)
                drift = abs(float(np.mean(risks)) - ref["marginal_risk_mean"])
                if drift > CONFIG["drift_tolerance"]:
                    raise SystemExit(f"[a1] DRIFT {drift:.2e} at {envname}/{feature}/{beta}")
                a_m = float(np.mean(avals))
                excess = ref["marginal_risk_mean"] - alpha
                se = ref["marginal_risk_se"]
                zone, pred, tol, ok = judge(a_m, excess, se, envname)
                cross = feature != battery_feat
                out.append({"source": "tilt_location", "env": envname,
                            "feature": feature, "beta": beta, "a": a_m,
                            "a_se": float(np.std(avals, ddof=1) / np.sqrt(len(avals))),
                            "excess": excess, "se": se, "zone": zone,
                            "curve_pred": pred, "tol": tol, "on_curve": bool(ok),
                            "cross_direction": bool(cross)})
                print(f"[a1] tilt  {envname:<11} {feature:<14} beta={beta:<6} "
                      f"a={a_m:+.4f} excess={excess:+.4f} pred={pred:+.4f} "
                      f"[{zone:<9}]{' X-DIR' if cross else '      '} "
                      f"{'ON' if ok else 'OFF'}")

    inr = [c for c in out if c["zone"] == "in_range"]
    ext = [c for c in out if c["zone"] == "extension"]
    xd = [c for c in out if c["cross_direction"]]
    summary = {
        "n_cells": len(out),
        "in_range": {"n": len(inr), "on_curve": sum(c["on_curve"] for c in inr)},
        "extension": {"n": len(ext), "on_curve": sum(c["on_curve"] for c in ext)},
        "cross_direction": {"n": len(xd), "on_curve": sum(c["on_curve"] for c in xd)},
    }
    print(f"[a1] in-range {summary['in_range']['on_curve']}/{summary['in_range']['n']}  "
          f"extension {summary['extension']['on_curve']}/{summary['extension']['n']}  "
          f"cross-direction {summary['cross_direction']['on_curve']}/{summary['cross_direction']['n']}")

    d = ROOT / "artifacts" / f"blockA_{config_hash(CONFIG)}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a1_cells.json").write_text(json.dumps(out, indent=2))
    (d / "a1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[out] {d}/a1_*.json")


def _slope(x, y):
    if len(x) < 3 or float(x @ x) == 0.0:
        return None, None, len(x)
    b = float(x @ y / (x @ x))
    r = y - b * x
    se = float(np.sqrt((r @ r) / (len(x) - 1) / (x @ x)))
    return b, se, len(x)


def part_a2():
    res = {}
    for envname, art in CONFIG["a1"]["sources"]["collapse_curves"].items():
        cells = json.loads((ROOT / "artifacts" / art / "cells.json").read_text())
        x = np.array([c["aligned_mean"] for c in cells])
        y = np.array([c["excess_mean"] for c in cells])
        r = {"pooled": _slope(x, y)}
        r["pos"] = _slope(x[x > 0], y[x > 0])
        r["neg"] = _slope(x[x < 0], y[x < 0])
        for w in CONFIG["a2"]["local_windows"]:
            m = (np.abs(x) <= w)
            r[f"pos_loc{w}"] = _slope(x[m & (x > 0)], y[m & (x > 0)])
            r[f"neg_loc{w}"] = _slope(x[m & (x < 0)], y[m & (x < 0)])
        res[envname] = {k: {"slope": v[0], "se": v[1], "n": v[2]} for k, v in r.items()}
        print(f"[a2] {envname:<11} pooled={r['pooled'][0]:.3f}±{r['pooled'][1]:.3f} "
              f"pos={r['pos'][0]:.3f}±{r['pos'][1]:.3f} (n={r['pos'][2]}) "
              f"neg={r['neg'][0]:.3f}±{r['neg'][1]:.3f} (n={r['neg'][2]})")
    d = ROOT / "artifacts" / f"blockA_{config_hash(CONFIG)}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a2_slopes.json").write_text(json.dumps(res, indent=2))
    print(f"[out] {d}/a2_slopes.json")


def part_a3(envname: str):
    """Replay one collapse battery, computing kappa_pred per cell."""
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    alpha = CONFIG["alpha"]
    h = CONFIG["a3"]["fd_window"]
    hs = [h] + CONFIG["a3"]["fd_window_sensitivity"]

    if envname == "claims":
        sys.path.insert(0, str(ROOT / "experiments" / "wp1_misspec_claims"))
        import run as bat
        cfg = bat.CONFIG
        art = ROOT / "artifacts" / "wp1mc_56704982681d6960"
        env = ClaimsEnv.induce()
        tilt_col = cfg["tilt_dim"]
    else:
        sys.path.insert(0, str(ROOT / "experiments" / "wp1_misspec_family"))
        import run as bat
        cfg = bat.build_config(envname)
        art = ROOT / "artifacts" / f"wp1mf_{envname}_{bat.config_hash(cfg)}"
        env = GenEnv.induce(envname)
        tilt_col = env.spec.bounded[cfg["tilt_feature"]]

    arch = json.loads((art / "results.json").read_text())
    est = {(r["name"], r["setting"], r["beta"]): r for r in arch
           if r["arm"] == "estimated"}
    feat = cfg["tilt_feature"]
    out = []
    for tup in bat.iter_cells(cfg):
        if envname == "claims":
            group, name, gi, si, setting, scale = tup
        else:
            group, name, gi, si, setting, scale = tup
        label = bat.setting_label(setting)
        rng = np.random.default_rng([cfg["seed"], 10 * (group == "family") + gi,
                                     si, int(scale * 1000)])
        if envname == "claims":
            view = bat.make_view(group, name, setting, cfg, rng)
        else:
            view = bat.make_view(env, group, name, setting, cfg, rng)
        n_fit = setting.get("n_fit", cfg["default_n_fit"])
        clip = tuple(setting.get("clip", cfg["default_clip"]))

        risks_e, sw, swh = [], {hh: [] for hh in hs}, {hh: [] for hh in hs}
        for _ in range(cfg["n_trials"]):
            cal = env.case_table(rng, cfg["n_cal"])
            ev = env.case_table(rng, cfg["n_eval"], beta=scale, feature=feat)
            w_cal = np.exp(env.tilt_logweight(cal.X, scale, feat))
            w_ev = np.exp(env.tilt_logweight(ev.X, scale, feat))
            if group == "family":
                p = next(iter(setting.values()))
                if name == "temper":
                    w_hat, w_ev_hat = w_cal ** p, w_ev ** p
                else:
                    w_hat = w_cal * np.exp(p * cal.X[:, tilt_col])
                    w_ev_hat = w_ev * np.exp(p * ev.X[:, tilt_col])
            else:
                Xs, _ = env.draw_instances(rng, n_fit)
                Xt, _ = env.draw_instances(rng, n_fit, scale, feat)
                w_fn = shift.fit_ratio(view(Xs), view(Xt),
                                       method=cfg["weight_estimator"], clip=clip)
                w_hat, w_ev_hat = w_fn(view(cal.X)), w_fn(view(ev.X))

            losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
            lam_o = crc.lhat_prop2(losses, lambdas, alpha, w_cal, w_ev)
            lam_star = float(np.mean(lam_o))
            lam_e = crc.lhat_prop2(losses, lambdas, alpha, w_hat, w_ev_hat)
            risks_e.append(float(((ev.s >= np.asarray(lam_e)) & ev.wrong).mean()))

            wn = w_cal / w_cal.mean()
            hn = w_hat / w_hat.mean()
            for hh in hs:
                lo, hi = lam_star - hh, lam_star + hh
                Llo = ((cal.s >= lo) & cal.wrong).astype(float)
                Lhi = ((cal.s >= hi) & cal.wrong).astype(float)
                sw[hh].append(float(np.mean(wn * (Llo - Lhi)) / (2 * hh)))
                swh[hh].append(float(np.mean(hn * (Llo - Lhi)) / (2 * hh)))

        key = (name, label, scale)
        ref = est[key]
        drift = abs(float(np.mean(risks_e)) - ref["marginal_risk_mean"])
        if drift > CONFIG["drift_tolerance"]:
            raise SystemExit(f"[a3] DRIFT {drift:.2e} at {envname}/{key}")
        row = {"env": envname, "name": name, "setting": label, "beta": scale,
               "excess": ref["excess_marginal_risk_mean"],
               "se": ref["marginal_risk_se"],
               "a": ref["aligned_mean"], "a_se": ref["aligned_se"]}
        for hh in hs:
            mw, mh = float(np.mean(sw[hh])), float(np.mean(swh[hh]))
            row[f"slope_w_h{hh}"] = mw
            row[f"slope_what_h{hh}"] = mh
            row[f"kappa_pred_h{hh}"] = mw / mh if mh > 0 else None
        out.append(row)
        kp = row[f"kappa_pred_h{h}"]
        print(f"[a3:{envname}] {name:<11} {label:<16} beta={scale:<7} "
              f"kappa_pred={kp if kp is None else round(kp, 3)}")

    d = ROOT / "artifacts" / f"blockA_{config_hash(CONFIG)}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"a3_{envname}.json").write_text(json.dumps(out, indent=2))
    print(f"[out] {d}/a3_{envname}.json")


def part_a3_analyze():
    h = CONFIG["a3"]["fd_window"]
    d = ROOT / "artifacts" / f"blockA_{config_hash(CONFIG)}"
    rows = []
    for envname in CONFIG["a1"]["sources"]["collapse_curves"]:
        rows += json.loads((d / f"a3_{envname}.json").read_text())
    keep = [r for r in rows
            if abs(r["a"]) >= CONFIG["a3"]["min_abs_a"]
            and r[f"slope_what_h{h}"] >= CONFIG["a3"]["min_slope_what"]
            and r[f"kappa_pred_h{h}"] is not None]
    dropped = len(rows) - len(keep)
    km = np.array([r["excess"] / r["a"] for r in keep])
    kp = np.array([r[f"kappa_pred_h{h}"] for r in keep])
    envs = [r["env"] for r in keep]

    b = float(kp @ km / (kp @ kp))
    resid = km - b * kp
    se_b = float(np.sqrt((resid @ resid) / (len(kp) - 1) / (kp @ kp)))
    ybar = km.mean()
    ss_tot = float(((km - ybar) ** 2).sum())
    r2_pred = 1.0 - float(((km - b * kp) ** 2).sum()) / ss_tot
    env_means = {e: km[[i for i, x in enumerate(envs) if x == e]].mean()
                 for e in set(envs)}
    fit_env = np.array([env_means[e] for e in envs])
    r2_env = 1.0 - float(((km - fit_env) ** 2).sum()) / ss_tot

    # TOST on the a2 pooled slopes, margin registered.
    slopes = json.loads((d / "a2_slopes.json").read_text())
    from scipy.stats import norm
    margin = CONFIG["a3"]["tost_margin"]
    tost = {}
    names = list(slopes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s1, e1 = slopes[names[i]]["pooled"]["slope"], slopes[names[i]]["pooled"]["se"]
            s2, e2 = slopes[names[j]]["pooled"]["slope"], slopes[names[j]]["pooled"]["se"]
            sed = float(np.hypot(e1, e2))
            z1 = (s1 - s2 + margin) / sed
            z2 = (margin - (s1 - s2)) / sed
            p = float(max(norm.sf(z1), norm.sf(z2)))
            power = float(norm.cdf(margin / sed - norm.ppf(0.95))
                          + norm.cdf(margin / sed + norm.ppf(0.95)) - 1.0)
            tost[f"{names[i]}|{names[j]}"] = {
                "diff": s1 - s2, "se": sed, "p_tost": p,
                "equivalent_at_margin": bool(p < 0.05), "power": power}

    ana = {"n_cells_used": len(keep), "n_dropped_by_guards": dropped,
           "regression_slope": b, "regression_slope_se": se_b,
           "r2_kappa_pred": r2_pred, "r2_env_label": r2_env,
           "tost_margin": margin, "tost_pairs": tost}
    (d / "a3_analysis.json").write_text(json.dumps(ana, indent=2))
    print(f"[a3] n={len(keep)} (dropped {dropped})  "
          f"regression slope={b:.3f}±{se_b:.3f}  "
          f"R2(kappa_pred)={r2_pred:.3f} vs R2(env)={r2_env:.3f}")
    eq = sum(1 for v in tost.values() if v["equivalent_at_margin"])
    print(f"[a3] TOST at margin {margin}: {eq}/{len(tost)} pairs equivalent; "
          f"powers {sorted(round(v['power'], 2) for v in tost.values())}")
    print(f"[out] {d}/a3_analysis.json")


if __name__ == "__main__":
    check_registration()
    part = sys.argv[sys.argv.index("--part") + 1]
    if part == "a1":
        part_a1()
    elif part == "a2":
        part_a2()
    elif part == "a3":
        part_a3(sys.argv[sys.argv.index("--env") + 1])
    elif part == "a3-analyze":
        part_a3_analyze()
