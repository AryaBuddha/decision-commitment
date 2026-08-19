"""Block B4: clip as a swept parameter. EVIDENCE TIER.

The wp1f sweeps found the default ratio clip (0.01, 0.99) binding at
extreme bounded tilts, breaking oracle-equivalence of a correctly
specified estimator in the two blind-driver environments (tickets,
compliance P3 misses). This promotes the boundary finding to a swept
knob: widen the clip and register where equivalence is recovered and
what it costs in ESS.

Cells: {tickets, compliance} x their top two sweep levels x clip in
{(0.01, 0.99), (0.005, 0.995), (0.002, 0.998)}; arms oracle + estimated,
500 trials.

Run:   python experiments/wp1_clip_recovery/run.py
Smoke: add --smoke 15
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import crc, metrics, shift, tests   # noqa: E402
from cus.envs.family import GenEnv, SPECS    # noqa: E402


CONFIG = {
    "experiment": "wp1_clip_recovery",
    "alpha": 0.10,
    "n_cal": 1000,
    "n_eval": 1000,
    "n_fit": 1000,
    "n_trials": 500,
    "n_lambda": 400,
    "cells": {
        "tickets": {"feature": "frustration", "betas": [5.982, 7.178]},
        "compliance": {"feature": "redline_density", "betas": [5.506, 6.607]},
    },
    "clips": [[0.01, 0.99], [0.005, 0.995], [0.002, 0.998]],
    "weight_estimator": "logistic",
    "z_one_sided": 1.645,
    "delta_control": 0.005,
    "delta_oracle": 0.005,
    "seed": 20260819,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp1_clip_recovery.json").read_text())
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


def run_cell(env, ei, li, ci, beta, clip, feature, cfg, n_trials):
    lambdas = np.linspace(0, 1, cfg["n_lambda"])
    rng = np.random.default_rng([cfg["seed"], 70 + ei, li, ci])
    trials = {"oracle": [], "estimated": []}
    battery, paired = [], []
    for _ in range(n_trials):
        cal = env.case_table(rng, cfg["n_cal"])
        ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feature)
        Xs, _ = env.draw_instances(rng, cfg["n_fit"])
        Xt, _ = env.draw_instances(rng, cfg["n_fit"], beta, feature)
        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        w_cal = np.exp(env.tilt_logweight(cal.X, beta, feature))
        w_ev = np.exp(env.tilt_logweight(ev.X, beta, feature))
        lam_o = crc.lhat_prop2(losses, lambdas, cfg["alpha"], w_cal, w_ev)
        res_o = metrics.evaluate(ev.s, ev.wrong, ev.region, lam_o, cfg["alpha"])
        res_o["ess"] = crc.effective_sample_size(w_cal)
        trials["oracle"].append(res_o)
        w_fn = shift.fit_ratio(Xs, Xt, method=cfg["weight_estimator"],
                               clip=tuple(clip))
        w_hat, w_ev_hat = w_fn(cal.X), w_fn(ev.X)
        lam_e = crc.lhat_prop2(losses, lambdas, cfg["alpha"], w_hat, w_ev_hat)
        res_e = metrics.evaluate(ev.s, ev.wrong, ev.region, lam_e, cfg["alpha"])
        res_e["ess"] = crc.effective_sample_size(w_hat)
        trials["estimated"].append(res_e)
        battery.append(shift.ratio_error_battery(w_hat, w_cal))
        paired.append(res_e["marginal_risk"] - res_o["marginal_risk"])
    d = np.asarray(paired)
    dm, dse = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    z = cfg["z_one_sided"]
    rows = []
    for arm, tr in trials.items():
        summ = metrics.summarise(tr)
        summ["verdict"] = verdict(summ["marginal_risk_mean"],
                                  summ["marginal_risk_se"], cfg)
        if arm == "estimated":
            for k in battery[0]:
                summ[f"ratio_{k}_mean"] = float(np.mean([x[k] for x in battery]))
            summ["paired_diff_mean"] = dm
            summ["paired_diff_se"] = dse
            summ["oracle_equiv"] = bool(
                -cfg["delta_oracle"] <= dm - z * dse
                and dm + z * dse <= cfg["delta_oracle"])
        rows.append({"env": env.spec.name, "beta": beta,
                     "clip": f"{clip[0]}-{clip[1]}", "arm": arm, **summ})
    return rows


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration()
    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    print(f"[wp1clip] {'PILOT SMOKE' if smoke else 'REAL (evidence tier)'}: "
          f"n_trials={n_trials}")
    all_rows = []
    for ei, (envname, spec) in enumerate(CONFIG["cells"].items()):
        env = GenEnv.induce(envname)
        for li, beta in enumerate(spec["betas"]):
            for ci, clip in enumerate(CONFIG["clips"]):
                rows = run_cell(env, ei, li, ci, beta, clip,
                                spec["feature"], CONFIG, n_trials)
                all_rows.extend(rows)
                for r in rows:
                    extra = ""
                    if r["arm"] == "estimated":
                        extra = (f" pd={r['paired_diff_mean']:+.4f} "
                                 f"{'equiv' if r['oracle_equiv'] else 'NOT-equiv'}"
                                 f" ess={r['ess_mean']:.0f}")
                    print(f"[wp1clip] {r['env']:<11} beta={r['beta']:<7} "
                          f"clip={r['clip']:<12} {r['arm']:<9} "
                          f"risk={r['marginal_risk_mean']:.4f}"
                          f"±{r['marginal_risk_se']:.4f}{extra}")
    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp1clip_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "n_trials_effective": n_trials, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
