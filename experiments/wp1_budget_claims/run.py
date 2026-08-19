"""Ablation C: calibration budget on the gated claims environment.
EVIDENCE TIER. Produces the certifiable-shift-vs-budget table.

n_cal in {250, 1000, 4000} crossed with beta in {0, 3, 6} on the blind
feature. The theorem holds at any n_cal; what the budget buys is
dispersion and conservatism, and the table prices both: a mean-controlled
certificate whose p95 realized risk is far above alpha is not one a
deployment can budget around.

Run:   python experiments/wp1_budget_claims/run.py
Smoke: add --smoke 20
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
    "experiment": "wp1_budget_claims",
    "environment": "claims",
    "alpha": 0.10,
    "n_cal_grid": [250, 1000, 4000],
    "n_eval": 1000,
    "n_fit": 1000,
    "n_trials": 500,
    "n_lambda": 400,
    "tilt_feature": "inconsistency",
    "beta_grid": [0.0, 3.0, 6.0],
    "weight_estimator": "logistic",
    "weight_clip": [0.01, 0.99],
    "z_one_sided": 1.645,
    "delta_control": 0.005,
    "delta_oracle": 0.005,
    "seed": 20260819,
}

ARMS = ("unweighted", "oracle", "estimated")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration(cfg, reg_dir):
    h = config_hash(cfg)
    path = reg_dir / f"{cfg['experiment']}.json"
    if not path.exists():
        raise SystemExit(f"No registration at {path}.")
    reg = json.loads(path.read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def verdict(mean, se, cfg):
    z, a, d = cfg["z_one_sided"], cfg["alpha"], cfg["delta_control"]
    if mean - z * se > a:
        return "VIOLATION"
    if mean + z * se <= a + d:
        return "consistent"
    return "inconclusive"


def run_cell(env, ni, n_cal, li, beta, cfg, n_trials):
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda"])
    rng = np.random.default_rng([cfg["seed"], 60 + ni, li])
    feat = cfg["tilt_feature"]
    trials = {arm: [] for arm in ARMS}
    paired = []
    for _ in range(n_trials):
        cal = env.case_table(rng, n_cal)
        ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feat)
        Xs, _ = env.draw_instances(rng, cfg["n_fit"])
        Xt, _ = env.draw_instances(rng, cfg["n_fit"], beta, feat)
        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        w_cal_true = np.exp(env.tilt_logweight(cal.X, beta, feat))
        w_ev_true = np.exp(env.tilt_logweight(ev.X, beta, feat))
        risks = {}
        for arm in ARMS:
            if arm == "unweighted":
                lam = crc.lhat_unweighted(losses, lambdas, cfg["alpha"])
                w_used = np.ones(n_cal)
            elif arm == "oracle":
                lam = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                                     w_cal_true, w_ev_true)
                w_used = w_cal_true
            else:
                w_fn = shift.fit_ratio(Xs, Xt, method=cfg["weight_estimator"],
                                       clip=tuple(cfg["weight_clip"]))
                w_cal_hat, w_ev_hat = w_fn(cal.X), w_fn(ev.X)
                lam = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                                     w_cal_hat, w_ev_hat)
                w_used = w_cal_hat
            res = metrics.evaluate(ev.s, ev.wrong, ev.region, lam, cfg["alpha"])
            res["ess"] = crc.effective_sample_size(w_used)
            risks[arm] = res["marginal_risk"]
            trials[arm].append(res)
        paired.append(risks["estimated"] - risks["oracle"])

    d = np.asarray(paired)
    dm, dse = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    z = cfg["z_one_sided"]
    rows = []
    for arm in ARMS:
        summ = metrics.summarise(trials[arm])
        summ["verdict"] = verdict(summ["marginal_risk_mean"],
                                  summ["marginal_risk_se"], cfg)
        summ["risk_spread_90"] = float(summ["marginal_risk_p95"]
                                       - summ["marginal_risk_p05"])
        if arm == "estimated":
            summ["paired_diff_mean"] = dm
            summ["paired_diff_se"] = dse
            summ["oracle_equiv"] = bool(
                -cfg["delta_oracle"] <= dm - z * dse
                and dm + z * dse <= cfg["delta_oracle"])
        rows.append({"n_cal": n_cal, "beta": beta, "arm": arm, **summ})
    return rows


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration(CONFIG, root / "registrations")
    env = ClaimsEnv.induce()
    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    print(f"[wp1b] {'PILOT SMOKE' if smoke else 'REAL (evidence tier)'}: "
          f"n_trials={n_trials}")

    all_rows = []
    for ni, n_cal in enumerate(CONFIG["n_cal_grid"]):
        for li, beta in enumerate(CONFIG["beta_grid"]):
            rows = run_cell(env, ni, n_cal, li, beta, CONFIG, n_trials)
            all_rows.extend(rows)
            for r in rows:
                print(f"[wp1b] n_cal={r['n_cal']:<5} beta={r['beta']:<4}"
                      f" {r['arm']:<10} risk={r['marginal_risk_mean']:.4f}"
                      f"±{r['marginal_risk_se']:.4f} [{r['verdict']:<12}]"
                      f" p95={r['marginal_risk_p95']:.4f}"
                      f" spread90={r['risk_spread_90']:.4f}"
                      f" ess={r['ess_mean']:7.1f}")

    suffix = f"_smoke{smoke}" if smoke else ""
    out_dir = root / "artifacts" / f"wp1b_{config_hash(CONFIG)}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**CONFIG, "n_trials_effective": n_trials, "smoke": bool(smoke)}, indent=2))
    (out_dir / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\n[out] {out_dir}")


if __name__ == "__main__":
    main()
