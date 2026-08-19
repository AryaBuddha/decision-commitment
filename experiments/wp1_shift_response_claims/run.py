"""WP1 primary sweep on environment 1 (claims triage). EVIDENCE TIER.

First real-environment run of the programme: the gated claims environment
(registrations/env_claims.json, gates ALL PASS at 5459d3b5a7b3c1a1)
replaces the placeholder. Same four arms as the registered placeholder
sweep, same three-outcome decision rule, now at confirmatory trial count
with Bonferroni-adjusted verdicts across the seven shift levels
(one-sided alpha 0.05 / 7, z = 2.4573); raw per-level verdicts are also
reported, labelled exploratory.

SAMPLING, rung 2. Source draws are fresh generator output; target draws
are exact rejection samples from exp(beta * inconsistency) * P0
(inconsistency is the gate-3-verified evidence-blind feature, lift
+0.0859). The unnormalised ratio exp(beta * x_inc) is exact by
construction and Proposition 2 is scale-invariant, so the oracle arm is
exact. chi2, TV, and the Proposition 3 envelope have no closed form on
this rung; they are Monte Carlo estimates on 400k fresh draws with SE
reported, and every figure states this.

SPLITS, four-way, fresh per trial: labelled source calibration; ratio-fit
source covariates; ratio-fit target covariates (unlabelled; latents
discarded); labelled target evaluation.

Run:   python experiments/wp1_shift_response_claims/run.py
Smoke: python experiments/wp1_shift_response_claims/run.py --smoke 20
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cus import crc, metrics, shift, tests   # noqa: E402
from cus.envs.claims import ClaimsEnv, gold  # noqa: E402


CONFIG = {
    "experiment": "wp1_shift_response_claims",
    "environment": "claims",
    "env_freeze_gate_hash": "5459d3b5a7b3c1a1",
    "alpha": 0.10,
    "n_cal": 1000,
    "n_eval": 1000,
    "n_fit": 1000,
    "n_trials": 500,
    "n_lambda": 400,
    "tilt_feature": "inconsistency",
    "beta_grid": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "weight_estimator": "logistic",
    "weight_clip": [0.01, 0.99],
    "mc_divergence_n": 400000,
    "z_one_sided": 1.645,
    "z_bonferroni": 2.4573,
    "delta_control": 0.005,
    "delta_oracle": 0.005,
    "seed": 20260818,
}

ARMS = ("unweighted", "oracle", "estimated", "glob_oracle")


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


def verdict(mean: float, se: float, cfg: dict, z: float) -> str:
    a, d = cfg["alpha"], cfg["delta_control"]
    if mean - z * se > a:
        return "VIOLATION"
    if mean + z * se <= a + d:
        return "consistent"
    return "inconclusive"


def run_cell(env: ClaimsEnv, li: int, beta: float, cfg: dict, n_trials: int):
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda"])
    rng = np.random.default_rng([cfg["seed"], li])
    feat = cfg["tilt_feature"]

    # Divergences: MC on fresh draws, reported with SE (rung 2, no closed form).
    dr = np.random.default_rng([cfg["seed"], 500 + li])
    Xd, _ = env.draw_instances(dr, cfg["mc_divergence_n"])
    wd = np.exp(env.tilt_logweight(Xd, beta, feat))
    wn = wd / wd.mean()
    nmc = len(wn)
    chi2 = float((wn ** 2).mean() - 1.0)
    chi2_se = float((wn ** 2).std() / np.sqrt(nmc))
    tv = float(0.5 * np.abs(wn - 1.0).mean())
    tv_se = float(0.5 * np.abs(wn - 1.0).std() / np.sqrt(nmc))

    trials = {arm: [] for arm in ARMS}
    diag, paired = [], []
    for _ in range(n_trials):
        cal = env.case_table(rng, cfg["n_cal"])
        ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feat)
        Xs_fit, _ = env.draw_instances(rng, cfg["n_fit"])
        Xt_fit, _ = env.draw_instances(rng, cfg["n_fit"], beta=beta, feature=feat)

        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        w_cal_true = np.exp(env.tilt_logweight(cal.X, beta, feat))
        w_ev_true = np.exp(env.tilt_logweight(ev.X, beta, feat))

        risks = {}
        for arm in ARMS:
            if arm == "unweighted":
                lam = crc.lhat_unweighted(losses, lambdas, cfg["alpha"])
                w_used = np.ones(len(cal.s))
            elif arm == "oracle":
                lam = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                                     w_cal_true, w_ev_true)
                w_used = w_cal_true
            elif arm == "glob_oracle":
                lam = crc.lhat_weighted_global(losses, lambdas, cfg["alpha"],
                                               w_cal=w_cal_true,
                                               w_test=float(w_ev_true.mean()))
                w_used = w_cal_true
            else:
                w_fn = shift.fit_ratio(Xs_fit, Xt_fit,
                                       method=cfg["weight_estimator"],
                                       clip=tuple(cfg["weight_clip"]))
                w_cal_hat, w_ev_hat = w_fn(cal.X), w_fn(ev.X)
                lam = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                                     w_cal_hat, w_ev_hat)
                w_used = w_cal_hat
                diag.append(shift.ratio_error_battery(w_cal_hat, w_cal_true))

            res = metrics.evaluate(ev.s, ev.wrong, ev.region, lam, cfg["alpha"])
            res["ess"] = crc.effective_sample_size(w_used)
            risks[arm] = res["marginal_risk"]
            trials[arm].append(res)
        paired.append(risks["estimated"] - risks["oracle"])

    d = np.asarray(paired)
    dm = float(d.mean())
    dse = float(d.std(ddof=1) / np.sqrt(len(d)))

    ana = {"chi2_mc": chi2, "chi2_mc_se": chi2_se,
           "tv_mc": tv, "tv_mc_se": tv_se,
           "prop3_envelope_mc": cfg["alpha"] + 1.0 * cfg["n_cal"] * tv,
           "sampling_rung": 2}

    rows = []
    for arm in ARMS:
        summ = metrics.summarise(trials[arm])
        m, se = summ["marginal_risk_mean"], summ["marginal_risk_se"]
        summ["verdict_raw"] = verdict(m, se, cfg, cfg["z_one_sided"])
        summ["verdict_bonf"] = verdict(m, se, cfg, cfg["z_bonferroni"])
        if arm == "estimated" and diag:
            for k in diag[0]:
                summ[f"ratio_{k}_mean"] = float(np.mean([x[k] for x in diag]))
            summ["paired_diff_mean"] = dm
            summ["paired_diff_se"] = dse
            zb = cfg["z_bonferroni"]
            summ["paired_ci_bonf"] = [dm - zb * dse, dm + zb * dse]
            summ["oracle_equiv_bonf"] = bool(
                -cfg["delta_oracle"] <= dm - zb * dse
                and dm + zb * dse <= cfg["delta_oracle"])
        rows.append({"beta": beta, "arm": arm, **ana, **summ})
    return rows


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])

    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Proposition 2 reduces to unweighted at w == 1: PASS")
    check_registration(CONFIG, root / "registrations")

    env = ClaimsEnv.induce()
    # Rung-2 fidelity spot check at the largest tilt before any trial runs.
    rng0 = np.random.default_rng([CONFIG["seed"], 999])
    bmax = max(CONFIG["beta_grid"])
    Xs, _ = env.draw_instances(rng0, 200_000)
    w = np.exp(env.tilt_logweight(Xs, bmax, CONFIG["tilt_feature"]))
    wn = w / w.mean()
    Xr, _ = env.draw_instances(rng0, 200_000, bmax, CONFIG["tilt_feature"])
    j = 8
    diff = abs(float((wn * Xs[:, j]).mean()) - float(Xr[:, j].mean()))
    tol = 5.0 * float(np.hypot((wn * Xs[:, j]).std() / np.sqrt(len(wn)),
                               Xr[:, j].std() / np.sqrt(len(Xr))))
    if diff > tol:
        raise SystemExit(f"[selftest] rung-2 fidelity FAILED: {diff} > {tol}")
    print(f"[selftest] rung-2 rejection fidelity at beta={bmax}: PASS")

    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    tag = "PILOT SMOKE" if smoke else "REAL (confirmatory, evidence tier)"
    print(f"[wp1c] {tag}: n_trials={n_trials}")

    all_rows = []
    for li, beta in enumerate(CONFIG["beta_grid"]):
        rows = run_cell(env, li, beta, CONFIG, n_trials)
        all_rows.extend(rows)
        for r in rows:
            extra = ""
            if r["arm"] == "estimated" and "ratio_w_l1_mean" in r:
                eq = "equiv" if r.get("oracle_equiv_bonf") else "NOT-equiv"
                extra = f" L1={r['ratio_w_l1_mean']:.3f} {eq}"
            print(f"[wp1c] beta={r['beta']:<4} chi2={r['chi2_mc']:5.2f} "
                  f"{r['arm']:<11} risk={r['marginal_risk_mean']:.4f}"
                  f"±{r['marginal_risk_se']:.4f} "
                  f"[raw {r['verdict_raw']:<12}|bonf {r['verdict_bonf']:<12}] "
                  f"defer={r['deferral_rate_mean']:.3f} "
                  f"ess={r['ess_mean']:7.1f} "
                  f"r1={r.get('marginal_risk_region1_mean', float('nan')):.3f}"
                  f"{extra}")

    suffix = f"_smoke{smoke}" if smoke else ""
    out_dir = root / "artifacts" / f"wp1c_{config_hash(CONFIG)}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**CONFIG, "n_trials_effective": n_trials, "smoke": bool(smoke)}, indent=2))
    (out_dir / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\n[out] {out_dir}")


if __name__ == "__main__":
    main()
