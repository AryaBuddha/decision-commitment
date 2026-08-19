"""WP1 primary sweep for environments 2-5 (family engine). EVIDENCE TIER.

Identical protocol to the claims sweep (wp1_shift_response_claims): four
arms, 500 trials per level, Bonferroni-adjusted verdicts across seven
levels, rung-2 exact rejection tilts, MC divergences with SE. One
registration per environment; each registration's shape prediction is the
APE forecast filed in cross_environment_hypothesis.json BEFORE this
runner existed for that environment.

Run:   python experiments/wp1_shift_response_family/run.py --env tickets
Smoke: add --smoke 20
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gates"))

from cus import crc, metrics, shift, tests   # noqa: E402
from cus.envs.family import GenEnv, SPECS    # noqa: E402
from run_gates_family import BETA_GRIDS      # noqa: E402


def build_config(name: str) -> dict:
    return {
        "experiment": f"wp1_shift_response_{name}",
        "environment": name,
        "alpha": 0.10,
        "n_cal": 1000,
        "n_eval": 1000,
        "n_fit": 1000,
        "n_trials": 500,
        "n_lambda": 400,
        "tilt_feature": SPECS[name].primary_tilt,
        "beta_grid": BETA_GRIDS[name],
        "weight_estimator": "logistic",
        "weight_clip": [0.01, 0.99],
        "mc_divergence_n": 400000,
        "z_one_sided": 1.645,
        "z_bonferroni": 2.4573,
        "delta_control": 0.005,
        "delta_oracle": 0.005,
        "seed": 20260819,
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
        raise SystemExit(f"Config hash {h} does not match registered "
                         f"{reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def verdict(mean, se, cfg, z):
    a, d = cfg["alpha"], cfg["delta_control"]
    if mean - z * se > a:
        return "VIOLATION"
    if mean + z * se <= a + d:
        return "consistent"
    return "inconclusive"


def run_cell(env, li, beta, cfg, n_trials):
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda"])
    rng = np.random.default_rng([cfg["seed"], li])
    feat = cfg["tilt_feature"]

    dr = np.random.default_rng([cfg["seed"], 500 + li])
    Xd, _ = env.draw_instances(dr, cfg["mc_divergence_n"])
    wd = np.exp(env.tilt_logweight(Xd, beta, feat))
    wn = wd / wd.mean()
    chi2 = float((wn ** 2).mean() - 1.0)
    chi2_se = float((wn ** 2).std() / np.sqrt(len(wn)))
    tv = float(0.5 * np.abs(wn - 1.0).mean())

    trials = {arm: [] for arm in ARMS}
    diag, paired = [], []
    for _ in range(n_trials):
        cal = env.case_table(rng, cfg["n_cal"])
        ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feat)
        Xs_fit, _ = env.draw_instances(rng, cfg["n_fit"])
        Xt_fit, _ = env.draw_instances(rng, cfg["n_fit"], beta, feat)

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
    dm, dse = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    ana = {"chi2_mc": chi2, "chi2_mc_se": chi2_se, "tv_mc": tv,
           "prop3_envelope_mc": cfg["alpha"] + cfg["n_cal"] * tv,
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
            zb = cfg["z_bonferroni"]
            summ["paired_diff_mean"] = dm
            summ["paired_diff_se"] = dse
            summ["oracle_equiv_bonf"] = bool(
                -cfg["delta_oracle"] <= dm - zb * dse
                and dm + zb * dse <= cfg["delta_oracle"])
        rows.append({"beta": beta, "arm": arm, **ana, **summ})
    return rows


def main() -> None:
    name = sys.argv[sys.argv.index("--env") + 1]
    cfg = build_config(name)
    root = pathlib.Path(__file__).resolve().parents[2]
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])

    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Proposition 2 reduces to unweighted at w == 1: PASS")
    check_registration(cfg, root / "registrations")
    env = GenEnv.induce(name)

    n_trials = smoke if smoke is not None else cfg["n_trials"]
    tag = "PILOT SMOKE" if smoke else "REAL (confirmatory, evidence tier)"
    print(f"[wp1f:{name}] {tag}: n_trials={n_trials}")

    all_rows = []
    for li, beta in enumerate(cfg["beta_grid"]):
        rows = run_cell(env, li, beta, cfg, n_trials)
        all_rows.extend(rows)
        for r in rows:
            extra = ""
            if r["arm"] == "estimated" and "ratio_w_l1_mean" in r:
                eq = "equiv" if r.get("oracle_equiv_bonf") else "NOT-equiv"
                extra = f" L1={r['ratio_w_l1_mean']:.3f} {eq}"
            print(f"[wp1f:{name}] beta={r['beta']:<7} chi2={r['chi2_mc']:5.2f} "
                  f"{r['arm']:<11} risk={r['marginal_risk_mean']:.4f}"
                  f"±{r['marginal_risk_se']:.4f} "
                  f"[raw {r['verdict_raw']:<12}|bonf {r['verdict_bonf']:<12}] "
                  f"defer={r['deferral_rate_mean']:.3f} "
                  f"ess={r['ess_mean']:7.1f}{extra}")

    suffix = f"_smoke{smoke}" if smoke else ""
    out_dir = root / "artifacts" / f"wp1f_{name}_{config_hash(cfg)}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**cfg, "n_trials_effective": n_trials, "smoke": bool(smoke)}, indent=2))
    (out_dir / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\n[out] {out_dir}")


if __name__ == "__main__":
    main()
