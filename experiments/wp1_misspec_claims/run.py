"""Q1 at EVIDENCE TIER: misspecification + aligned-error collapse on the
gated claims environment.

The placeholder established (fa8459eb3cb50722, fd2279c8f7dc2df6) that
excess marginal risk under estimated-ratio Proposition 2 is one-dimensional
in signed aligned weight error a = E[(w - w_hat) L(lambda*)]. This
experiment asks whether that collapse survives contact with a real
rule-induction environment, in one registered run combining both designs:
realistic degradation axes AND synthetic estimators, with aligned error
recorded per trial from the start.

REALISTIC AXES (shift fixed at beta in {3, 5} on the blind feature):
  deprivation  the ratio fit's view of inconsistency is degraded
               (rho = 1.0 full, 0.7 / 0.35 noisy proxy, 0.0 dropped).
               On this environment the rho = 0 cell mirrors reality
               exactly: the feature that was never logged for rule
               induction is also missing from the ratio fit.
  starvation   n_fit in {50, 100, 250, 1000}.
  inflation    {0, 20, 50, 100} pure-noise columns appended to the
               estimator's view (wide logging schemas).
  mismatch     clip in {(0.01,0.99), (0.05,0.95), (0.2,0.8)}. The tilt is
               log-linear in the manifest, so the logistic estimator is
               well-specified and these cells are expected to pad the
               collapse origin.

SYNTHETIC FAMILIES (exact unnormalised ratio, no classifier):
  temper       w^gamma, gamma in {0 .. 1.5} (0.8 included for the
               coincidence check with directional delta = -1 at beta 5).
  directional  w * exp(delta * inconsistency), delta in {-2 .. 2}.

ALIGNED ERROR, preregistered: a = mean over calibration of
(w_norm - w_hat_norm) * 1{s >= lambda* and wrong}, lambda* the trial mean
of oracle Prop 2 thresholds on evaluation points. Both weight vectors are
normalised to mean 1 ON THE CALIBRATION SAMPLE, so the exact unnormalised
rung-2 ratio enters a directly and no MC normaliser touches a or the
|a| <= L1 arithmetic; the MC-estimated normaliser (with SE) is used only
for chi2 axis labels.

Run:   python experiments/wp1_misspec_claims/run.py
Smoke: python experiments/wp1_misspec_claims/run.py --smoke 20
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cus import crc, metrics, shift, tests   # noqa: E402
from cus.envs.claims import ClaimsEnv        # noqa: E402


CONFIG = {
    "experiment": "wp1_misspec_claims",
    "environment": "claims",
    "env_freeze_gate_hash": "5459d3b5a7b3c1a1",
    "alpha": 0.10,
    "n_cal": 1000,
    "n_eval": 1000,
    "n_trials": 300,
    "n_lambda": 400,
    "tilt_feature": "inconsistency",
    "tilt_dim": 8,
    "beta_scales": [3.0, 5.0],
    "weight_estimator": "logistic",
    "default_clip": [0.01, 0.99],
    "default_n_fit": 1000,
    "axes": {
        "deprivation": {"view_rho": [1.0, 0.7, 0.35, 0.0]},
        "starvation": {"n_fit": [50, 100, 250, 1000]},
        "inflation": {"extra_dims": [0, 20, 50, 100]},
        "mismatch": {"clip": [[0.01, 0.99], [0.05, 0.95], [0.2, 0.8]]},
    },
    "families": {
        "temper": {"gamma": [0.0, 0.25, 0.5, 0.75, 0.8, 1.0, 1.25, 1.5]},
        "directional": {"delta": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]},
    },
    "aligned_error_def": ("mean over calibration of (w_norm - w_hat_norm) * "
                          "1{s >= lambda_star and wrong}; lambda_star = trial "
                          "mean of oracle Prop 2 thresholds on eval points; "
                          "weights normalised to mean 1 on the calibration sample"),
    "collapse_criteria": {
        "spearman_min": 0.9,
        "residual_tol": "max(3*SE_cell, 0.0075)",
        "residual_pass_fraction": 0.9,
        "fit": "unweighted isotonic regression on cell means, increasing",
        "slope_window": 0.02,
    },
    "mc_divergence_n": 400000,
    "z_one_sided": 1.645,
    "delta_control": 0.005,
    "delta_oracle": 0.005,
    "seed": 20260819,
}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration(cfg: dict, reg_dir: pathlib.Path) -> None:
    h = config_hash(cfg)
    path = reg_dir / f"{cfg['experiment']}.json"
    if not path.exists():
        raise SystemExit(f"No registration at {path}. Write one, commit it, then run.")
    reg = json.loads(path.read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(
            f"Config hash {h} does not match registered {reg.get('config_hash')}.\n"
            "Update the registration deliberately (with an amendment) or revert."
        )
    print(f"[prereg] config {h} matches registration")


def verdict(mean: float, se: float, cfg: dict) -> str:
    z, a, d = cfg["z_one_sided"], cfg["alpha"], cfg["delta_control"]
    if mean - z * se > a:
        return "VIOLATION"
    if mean + z * se <= a + d:
        return "consistent"
    return "inconclusive"


def aligned_error(cal, w_true, w_hat, lam_star: float) -> float:
    wn = w_true / w_true.mean()
    hn = w_hat / w_hat.mean()
    L = ((cal.s >= lam_star) & cal.wrong).astype(float)
    return float(np.mean((wn - hn) * L))


def iter_cells(cfg: dict):
    for ai, (axis, spec) in enumerate(cfg["axes"].items()):
        key = list(spec.keys())[0]
        for si, val in enumerate(spec[key]):
            for scale in cfg["beta_scales"]:
                yield "axis", axis, ai, si, {key: val}, scale
    for fi, (family, spec) in enumerate(cfg["families"].items()):
        key = list(spec.keys())[0]
        for si, val in enumerate(spec[key]):
            for scale in cfg["beta_scales"]:
                yield "family", family, fi, si, {key: val}, scale


def setting_label(setting: dict) -> str:
    k, v = next(iter(setting.items()))
    if k == "clip":
        return f"clip={v[0]}-{v[1]}"
    return f"{k}={v}"


def make_view(group: str, name: str, setting: dict, cfg: dict,
              rng: np.random.Generator):
    """The estimator's degraded view; applied at fit AND application time."""
    if group != "axis" or name in ("starvation", "mismatch"):
        return lambda X: X
    if name == "deprivation":
        rho = setting["view_rho"]
        k = cfg["tilt_dim"]
        if rho == 1.0:
            return lambda X: X
        if rho == 0.0:
            keep = [j for j in range(13) if j != k]
            return lambda X: X[:, keep]

        def view(X):
            X = X.copy()
            X[:, k] = rho * X[:, k] + np.sqrt(1 - rho ** 2) * \
                rng.standard_normal(len(X))
            return X
        return view
    extra = setting["extra_dims"]                       # inflation
    if extra == 0:
        return lambda X: X
    return lambda X: np.hstack([X, rng.standard_normal((len(X), extra))])


def run_cell(env, group, name, gi, si, setting, scale, cfg, n_trials):
    k = cfg["tilt_dim"]
    feat = cfg["tilt_feature"]
    n_fit = setting.get("n_fit", cfg["default_n_fit"])
    clip = tuple(setting.get("clip", cfg["default_clip"]))
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda"])
    rng = np.random.default_rng([cfg["seed"], 10 * (group == "family") + gi,
                                 si, int(scale * 1000)])
    view = make_view(group, name, setting, cfg, rng)

    trials = {"oracle": [], "estimated": []}
    battery, aligned, paired = [], [], []

    for _ in range(n_trials):
        cal = env.case_table(rng, cfg["n_cal"])
        ev = env.case_table(rng, cfg["n_eval"], beta=scale, feature=feat)
        w_cal_true = np.exp(env.tilt_logweight(cal.X, scale, feat))
        w_ev_true = np.exp(env.tilt_logweight(ev.X, scale, feat))

        if group == "family":
            p = next(iter(setting.values()))
            if name == "temper":
                w_cal_hat, w_ev_hat = w_cal_true ** p, w_ev_true ** p
            else:
                w_cal_hat = w_cal_true * np.exp(p * cal.X[:, k])
                w_ev_hat = w_ev_true * np.exp(p * ev.X[:, k])
        else:
            Xs, _ = env.draw_instances(rng, n_fit)
            Xt, _ = env.draw_instances(rng, n_fit, scale, feat)
            w_fn = shift.fit_ratio(view(Xs), view(Xt),
                                   method=cfg["weight_estimator"], clip=clip)
            w_cal_hat, w_ev_hat = w_fn(view(cal.X)), w_fn(view(ev.X))

        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        lam_o = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                               w_cal_true, w_ev_true)
        lam_star = float(np.mean(lam_o))
        res_o = metrics.evaluate(ev.s, ev.wrong, ev.region, lam_o, cfg["alpha"])
        res_o["ess"] = crc.effective_sample_size(w_cal_true)
        trials["oracle"].append(res_o)

        lam_e = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                               w_cal_hat, w_ev_hat)
        res_e = metrics.evaluate(ev.s, ev.wrong, ev.region, lam_e, cfg["alpha"])
        res_e["ess"] = crc.effective_sample_size(w_cal_hat)
        trials["estimated"].append(res_e)

        battery.append(shift.ratio_error_battery(w_cal_hat, w_cal_true))
        aligned.append(aligned_error(cal, w_cal_true, w_cal_hat, lam_star))
        paired.append(res_e["marginal_risk"] - res_o["marginal_risk"])

    a = np.asarray(aligned)
    d = np.asarray(paired)
    dm, dse = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    z = cfg["z_one_sided"]

    rows = []
    for arm, tr in trials.items():
        summ = metrics.summarise(tr)
        summ["verdict"] = verdict(summ["marginal_risk_mean"],
                                  summ["marginal_risk_se"], cfg)
        if arm == "estimated":
            for key in battery[0]:
                summ[f"ratio_{key}_mean"] = float(np.mean([x[key] for x in battery]))
            summ["aligned_mean"] = float(a.mean())
            summ["aligned_se"] = float(a.std(ddof=1) / np.sqrt(len(a)))
            summ["paired_diff_mean"] = dm
            summ["paired_diff_se"] = dse
            summ["oracle_equiv"] = bool(
                -cfg["delta_oracle"] <= dm - z * dse
                and dm + z * dse <= cfg["delta_oracle"])
            summ["envelope_ok"] = bool(
                summ["excess_marginal_risk_mean"] - z * summ["marginal_risk_se"]
                <= summ["ratio_w_l1_mean"])
            summ["aligned_le_l1"] = bool(
                abs(summ["aligned_mean"]) <= summ["ratio_w_l1_mean"] + 1e-12)
        rows.append({"group": group, "name": name,
                     "setting": setting_label(setting), "beta": scale,
                     "arm": arm, **summ})
    return rows


def collapse_analysis(cells, cfg):
    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression
    x = np.array([c["aligned_mean"] for c in cells])
    y = np.array([c["excess_mean"] for c in cells])
    se = np.array([c["risk_se"] for c in cells])
    rho = float(spearmanr(x, y).statistic)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    fit = iso.fit_transform(x, y)
    resid = y - fit
    tol = np.maximum(3.0 * se, 0.0075)
    ok = np.abs(resid) <= tol
    win = cfg["collapse_criteria"]["slope_window"]
    m = np.abs(x) <= win
    slope = float(x[m] @ y[m] / (x[m] @ x[m])) if m.sum() >= 3 else None
    crit = cfg["collapse_criteria"]
    return {
        "n_cells": len(cells),
        "spearman": rho,
        "residual_pass_fraction": float(ok.mean()),
        "origin_slope": slope,
        "origin_n": int(m.sum()),
        "failing_cells": [
            {"cell": f"{c['name']}/{c['setting']}/beta={c['beta']}",
             "aligned": c["aligned_mean"], "residual": float(r),
             "tol": float(t)}
            for c, r, t, o in zip(cells, resid, tol, ok) if not o],
        "collapse": bool(rho >= crit["spearman_min"]
                         and float(ok.mean()) >= crit["residual_pass_fraction"]),
        "isotonic_fit": [[float(p), float(q)] for p, q in zip(x, fit)],
    }


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])

    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Proposition 2 reduces to unweighted at w == 1: PASS")
    check_registration(CONFIG, root / "registrations")
    env = ClaimsEnv.induce()

    dr = np.random.default_rng([CONFIG["seed"], 900])
    Xd, _ = env.draw_instances(dr, CONFIG["mc_divergence_n"])
    chi2 = {}
    for b in CONFIG["beta_scales"]:
        w = np.exp(env.tilt_logweight(Xd, b, CONFIG["tilt_feature"]))
        wn = w / w.mean()
        chi2[b] = (float((wn ** 2).mean() - 1.0),
                   float((wn ** 2).std() / np.sqrt(len(wn))))
        print(f"[mc] chi2(beta={b}) = {chi2[b][0]:.4f} ± {chi2[b][1]:.4f}")

    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    tag = "PILOT SMOKE" if smoke else "REAL (confirmatory, evidence tier)"
    print(f"[wp1mc] {tag}: n_trials={n_trials}")

    all_rows, cells = [], []
    for group, name, gi, si, setting, scale in iter_cells(CONFIG):
        rows = run_cell(env, group, name, gi, si, setting, scale,
                        CONFIG, n_trials)
        all_rows.extend(rows)
        for r in rows:
            if r["arm"] == "estimated":
                cells.append({
                    "group": r["group"], "name": r["name"],
                    "setting": r["setting"], "beta": r["beta"],
                    "excess_mean": r["excess_marginal_risk_mean"],
                    "risk_se": r["marginal_risk_se"],
                    "aligned_mean": r["aligned_mean"],
                    "aligned_se": r["aligned_se"],
                    "l1_mean": r["ratio_w_l1_mean"],
                })
                extra = (f" aligned={r['aligned_mean']:+.4f}"
                         f" L1={r['ratio_w_l1_mean']:.3f}"
                         f" {'equiv' if r['oracle_equiv'] else 'NOT-equiv'}"
                         f" {'env-ok' if r['envelope_ok'] else 'ENV-BREACH'}"
                         f"{'' if r['aligned_le_l1'] else ' ALIGNED>L1-BUG'}")
            else:
                extra = ""
            print(f"[wp1mc] {r['name']:<11} {r['setting']:<16} beta={r['beta']:<4}"
                  f" {r['arm']:<9} risk={r['marginal_risk_mean']:.4f}"
                  f"±{r['marginal_risk_se']:.4f} [{r['verdict']:<12}]{extra}")

    analysis = collapse_analysis(cells, CONFIG) if smoke is None else {"skipped": "smoke"}
    if smoke is None:
        print(f"[wp1mc] collapse: n={analysis['n_cells']} "
              f"spearman={analysis['spearman']:.3f} "
              f"resid_pass={analysis['residual_pass_fraction']:.2f} "
              f"origin_slope={analysis['origin_slope']} "
              f"-> {'COLLAPSE' if analysis['collapse'] else 'NO COLLAPSE'}")

    suffix = f"_smoke{smoke}" if smoke else ""
    out_dir = root / "artifacts" / f"wp1mc_{config_hash(CONFIG)}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**CONFIG, "n_trials_effective": n_trials, "smoke": bool(smoke),
         "chi2_mc": {str(k): v for k, v in chi2.items()}}, indent=2))
    (out_dir / "results.json").write_text(json.dumps(all_rows, indent=2))
    (out_dir / "cells.json").write_text(json.dumps(cells, indent=2))
    (out_dir / "collapse.json").write_text(json.dumps(analysis, indent=2))
    print(f"\n[out] {out_dir}")


if __name__ == "__main__":
    main()
