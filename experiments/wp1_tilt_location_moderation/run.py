"""Ablation A on moderation: tilt location at matched chi2, second
environment. EVIDENCE TIER.

Claims showed the visible, gold-heavy feature outdamaging the blind one
at every matched level (defense mounted, defense lost). Moderation's
pair has the opposite balance: toxicity is visible AND heavily defended
(gold-dominant, finely split), sarcasm is blind with modest gold weight.
The registered prediction, from a disclosed probe, is a CROSSOVER:
gold-relevance dominates at small shift, the defense dominates at large
shift, and blindness never defends. Together the two environments
instantiate both branches of the F10 law: damage = gold-relevance NET of
self-correction.

Run:   python experiments/wp1_tilt_location_moderation/run.py
Smoke: add --smoke 15
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cus import crc, metrics, tests          # noqa: E402
from cus.envs.family import GenEnv           # noqa: E402


CONFIG = {
    "experiment": "wp1_tilt_location_moderation",
    "environment": "moderation",
    "alpha": 0.10,
    "n_cal": 1000,
    "n_eval": 1000,
    "n_trials": 500,
    "n_lambda": 400,
    "levels": [
        {"chi2_target": 0.233, "beta_toxicity": 1.641, "beta_sarcasm": 2.483},
        {"chi2_target": 1.302, "beta_toxicity": 3.283, "beta_sarcasm": 4.800},
        {"chi2_target": 3.470, "beta_toxicity": 4.924, "beta_sarcasm": 6.581},
    ],
    "mc_divergence_n": 400000,
    "z_one_sided": 1.645,
    "z_named": 2.128,
    "delta_control": 0.005,
    "seed": 20260819,
}

ARMS = ("unweighted", "oracle")


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


def verdict(mean, se, cfg, z):
    a, d = cfg["alpha"], cfg["delta_control"]
    if mean - z * se > a:
        return "VIOLATION"
    if mean + z * se <= a + d:
        return "consistent"
    return "inconclusive"


def run_cell(env, li, fi, feature, beta, cfg, n_trials):
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda"])
    rng = np.random.default_rng([cfg["seed"], 40 + li, fi])
    dr = np.random.default_rng([cfg["seed"], 840 + li, fi])
    Xd, _ = env.draw_instances(dr, cfg["mc_divergence_n"])
    w = np.exp(env.tilt_logweight(Xd, beta, feature))
    wn = w / w.mean()
    chi2 = float((wn ** 2).mean() - 1.0)

    trials = {arm: [] for arm in ARMS}
    for _ in range(n_trials):
        cal = env.case_table(rng, cfg["n_cal"])
        ev = env.case_table(rng, cfg["n_eval"], beta=beta, feature=feature)
        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        w_cal = np.exp(env.tilt_logweight(cal.X, beta, feature))
        w_ev = np.exp(env.tilt_logweight(ev.X, beta, feature))
        for arm in ARMS:
            if arm == "unweighted":
                lam = crc.lhat_unweighted(losses, lambdas, cfg["alpha"])
                w_used = np.ones(len(cal.s))
            else:
                lam = crc.lhat_prop2(losses, lambdas, cfg["alpha"], w_cal, w_ev)
                w_used = w_cal
            res = metrics.evaluate(ev.s, ev.wrong, ev.region, lam, cfg["alpha"])
            res["ess"] = crc.effective_sample_size(w_used)
            trials[arm].append(res)

    rows = []
    for arm in ARMS:
        summ = metrics.summarise(trials[arm])
        m, se = summ["marginal_risk_mean"], summ["marginal_risk_se"]
        summ["verdict_raw"] = verdict(m, se, cfg, cfg["z_one_sided"])
        summ["verdict_named"] = verdict(m, se, cfg, cfg["z_named"])
        rows.append({"level": li, "feature": feature, "beta": beta,
                     "chi2_mc": chi2, "sampling_rung": 2, "arm": arm, **summ})
    return rows


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration(CONFIG, root / "registrations")
    env = GenEnv.induce("moderation")
    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    print(f"[wp1tlm] {'PILOT SMOKE' if smoke else 'REAL (evidence tier)'}: "
          f"n_trials={n_trials}")

    all_rows = []
    for li, level in enumerate(CONFIG["levels"]):
        for fi, feature in enumerate(("toxicity", "sarcasm")):
            beta = level[f"beta_{feature}"]
            rows = run_cell(env, li, fi, feature, beta, CONFIG, n_trials)
            all_rows.extend(rows)
            for r in rows:
                print(f"[wp1tlm] chi2~{level['chi2_target']:<6} {r['feature']:<9}"
                      f" beta={r['beta']:<6} {r['arm']:<10}"
                      f" risk={r['marginal_risk_mean']:.4f}±{r['marginal_risk_se']:.4f}"
                      f" [raw {r['verdict_raw']:<12}|named {r['verdict_named']:<12}]"
                      f" defer={r['deferral_rate_mean']:.3f}"
                      f" chi2_mc={r['chi2_mc']:.3f}")

    suffix = f"_smoke{smoke}" if smoke else ""
    out_dir = root / "artifacts" / f"wp1tlm_{config_hash(CONFIG)}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**CONFIG, "n_trials_effective": n_trials, "smoke": bool(smoke)}, indent=2))
    (out_dir / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\n[out] {out_dir}")


if __name__ == "__main__":
    main()
