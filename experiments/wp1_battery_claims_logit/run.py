"""Block B3 battery: shift-response + temper collapse on the gated
claims-logit variant (continuous per-case evidence). EVIDENCE TIER.

Run:   python experiments/wp1_battery_claims_logit/run.py
Smoke: add --smoke 10
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import crc, metrics, shift, tests        # noqa: E402
from cus.envs.claims_logit import ClaimsLogitEnv  # noqa: E402


CONFIG = {
    "experiment": "wp1_battery_claims_logit",
    "environment": "claims_logit",
    "env_freeze_gate_hash": "6a1d980a7c21ed3c",
    "alpha": 0.10,
    "n_cal": 1000, "n_eval": 1000, "n_fit": 1000,
    "n_trials_sweep": 500, "n_trials_temper": 300,
    "n_lambda": 400,
    "tilt_feature": "inconsistency",
    "beta_grid": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "temper_betas": [3.0, 5.0],
    "temper_gammas": [0.0, 0.25, 0.5, 0.75, 0.8, 1.0, 1.25, 1.5],
    "weight_estimator": "logistic",
    "weight_clip": [0.01, 0.99],
    "z_one_sided": 1.645, "z_bonferroni": 2.4573,
    "delta_control": 0.005, "delta_oracle": 0.005,
    "seed": 20260819,
}

ARMS = ("unweighted", "oracle", "estimated", "glob_oracle")


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp1_battery_claims_logit.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def verdict(mean, se, cfg, z):
    a, d = cfg["alpha"], cfg["delta_control"]
    if mean - z * se > a:
        return "VIOLATION"
    if mean + z * se <= a + d:
        return "consistent"
    return "inconclusive"


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration()
    env = ClaimsLogitEnv.induce()
    ns = smoke if smoke is not None else CONFIG["n_trials_sweep"]
    nt = smoke if smoke is not None else CONFIG["n_trials_temper"]
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    alpha = CONFIG["alpha"]
    feat = CONFIG["tilt_feature"]
    cfg = CONFIG
    all_rows = []

    for li, beta in enumerate(cfg["beta_grid"]):
        rng = np.random.default_rng([cfg["seed"], 400, li])
        trials = {a: [] for a in ARMS}
        paired = []
        for _ in range(ns):
            cal = env.case_table(rng, cfg["n_cal"], feature=feat)
            ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feat)
            Xs, _ = env.draw_instances(rng, cfg["n_fit"], feature=feat)
            Xt, _ = env.draw_instances(rng, cfg["n_fit"], beta, feat)
            losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
            w = np.exp(env.tilt_logweight(cal.X, beta, feat))
            wev = np.exp(env.tilt_logweight(ev.X, beta, feat))
            risks = {}
            for arm in ARMS:
                if arm == "unweighted":
                    lam = crc.lhat_unweighted(losses, lambdas, alpha)
                    w_used = np.ones(cfg["n_cal"])
                elif arm == "oracle":
                    lam = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
                    w_used = w
                elif arm == "glob_oracle":
                    lam = crc.lhat_weighted_global(losses, lambdas, alpha,
                                                   w_cal=w, w_test=float(wev.mean()))
                    w_used = w
                else:
                    w_fn = shift.fit_ratio(Xs, Xt, method=cfg["weight_estimator"],
                                           clip=tuple(cfg["weight_clip"]))
                    w_hat, w_ev_hat = w_fn(cal.X), w_fn(ev.X)
                    lam = crc.lhat_prop2(losses, lambdas, alpha, w_hat, w_ev_hat)
                    w_used = w_hat
                res = metrics.evaluate(ev.s, ev.wrong, ev.region, lam, alpha)
                res["ess"] = crc.effective_sample_size(w_used)
                risks[arm] = res["marginal_risk"]
                trials[arm].append(res)
            paired.append(risks["estimated"] - risks["oracle"])
        d = np.asarray(paired)
        for arm in ARMS:
            summ = metrics.summarise(trials[arm])
            m, se = summ["marginal_risk_mean"], summ["marginal_risk_se"]
            summ["verdict_raw"] = verdict(m, se, cfg, cfg["z_one_sided"])
            summ["verdict_bonf"] = verdict(m, se, cfg, cfg["z_bonferroni"])
            if arm == "estimated":
                dm, dse = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
                summ["paired_diff_mean"] = dm
                summ["paired_diff_se"] = dse
                zb = cfg["z_bonferroni"]
                summ["oracle_equiv_bonf"] = bool(
                    -cfg["delta_oracle"] <= dm - zb * dse
                    and dm + zb * dse <= cfg["delta_oracle"])
            all_rows.append({"part": "sweep", "beta": beta, "arm": arm, **summ})
            print(f"[wp1bl] beta={beta:<4} {arm:<11} risk={m:.4f}±{se:.4f} "
                  f"[bonf {summ['verdict_bonf']:<12}]")

    cells = []
    for bi, beta in enumerate(cfg["temper_betas"]):
        for gi, gamma in enumerate(cfg["temper_gammas"]):
            rng = np.random.default_rng([cfg["seed"], 500, gi, bi])
            aligned, paired, ro_ = [], [], []
            for _ in range(nt):
                cal = env.case_table(rng, cfg["n_cal"], feature=feat)
                ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feat)
                w = np.exp(env.tilt_logweight(cal.X, beta, feat))
                wev = np.exp(env.tilt_logweight(ev.X, beta, feat))
                what, wevhat = w ** gamma, wev ** gamma
                losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
                lam_o = crc.lhat_prop2(losses, lambdas, alpha, w, wev)
                ls = float(np.mean(lam_o))
                ro = float(((ev.s >= np.asarray(lam_o)) & ev.wrong).mean())
                lam_e = crc.lhat_prop2(losses, lambdas, alpha, what, wevhat)
                re = float(((ev.s >= np.asarray(lam_e)) & ev.wrong).mean())
                wn = w / w.mean(); hn = what / what.mean()
                L = ((cal.s >= ls) & cal.wrong).astype(float)
                aligned.append(float(np.mean((wn - hn) * L)))
                paired.append(re - ro)
                ro_.append(ro)
            cells.append({"part": "temper", "gamma": gamma, "beta": beta,
                          "aligned_mean": float(np.mean(aligned)),
                          "paired_diff_mean": float(np.mean(paired)),
                          "paired_diff_se": float(np.std(paired, ddof=1)
                                                  / np.sqrt(len(paired))),
                          "oracle_excess": float(np.mean(ro_) - alpha)})
    all_rows.extend(cells)
    x = np.array([c["aligned_mean"] for c in cells])
    y = np.array([c["paired_diff_mean"] for c in cells])
    m = np.abs(x) > 1e-12
    slope = float(x[m] @ y[m] / (x[m] @ x[m]))
    r = y[m] - slope * x[m]
    sse = float(np.sqrt((r @ r) / (m.sum() - 1) / (x[m] @ x[m])))
    oexc = float(np.mean([c["oracle_excess"] for c in cells]))
    print(f"[wp1bl] temper pd_slope={slope:.3f}±{sse:.3f} oracle_excess={oexc:+.4f}")
    all_rows.append({"part": "summary", "pd_slope": slope, "pd_slope_se": sse,
                     "oracle_excess_mean": oexc})

    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp1bl_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
