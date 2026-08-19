"""WP2 Phase 5b, stage A: the certificate envelope on the holdout under
DELIBERATELY DEGRADED estimators, computed from deployment-visible data
only, before any realized-risk sweep.

The first capstone (wp2p5) covered a deployment where realized excess
was negative everywhere: it validated the workflow, not the protection.
This run is the protection test: the WP1 realistic degradation axes
(no correction, deprivation of the blind driver, deprivation of driver
AND proxy, ratio-fit starvation) instantiated on the holdout, where a
disclosed probe shows the no-correction cells realize genuinely
positive excess. Certificate form v3 (floored, 20 bins), per-draw
values stored.

Run: python experiments/wp2_prospective_stress/envelope.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import shift, tests                       # noqa: E402
from cus.certificate import Audit, certificate     # noqa: E402
from cus.envs.returns import ReturnsEnv            # noqa: E402

SERIAL, PROXY = 6, 12

CONFIG = {
    "experiment": "wp2_prospective_stress",
    "environment": "returns",
    "certificate_version": "v3 (floored at zero, 20 bins, per-draw stored)",
    "alpha": 0.10,
    "stress_cells": [
        {"kind": "no_correction", "feature": "serial_rate", "beta": 2.0},
        {"kind": "no_correction", "feature": "serial_rate", "beta": 4.0},
        {"kind": "no_correction", "feature": "serial_rate", "beta": 6.0},
        {"kind": "no_correction", "feature": "desc_vagueness", "beta": 2.0},
        {"kind": "no_correction", "feature": "desc_vagueness", "beta": 4.0},
        {"kind": "no_correction", "feature": "desc_vagueness", "beta": 6.0},
        {"kind": "deprivation", "feature": "serial_rate", "beta": 4.0},
        {"kind": "deprivation", "feature": "serial_rate", "beta": 6.0},
        {"kind": "double_deprivation", "feature": "serial_rate", "beta": 6.0},
        {"kind": "starvation50", "feature": "serial_rate", "beta": 4.0},
        {"kind": "starvation50", "feature": "serial_rate", "beta": 6.0},
        {"kind": "starvation100", "feature": "serial_rate", "beta": 6.0},
    ],
    "recipe": {"n_bins": 20, "audit_C": 1.0, "n_audit": 60000,
               "n_src": 10000, "n_tgt": 10000, "z": 1.645},
    "deployed_clip": [0.01, 0.99], "deployed_C": 1.0,
    "n_cal": 1000, "n_lambda": 400,
    "n_envelope_draws": 30,
    "n_trials_sweep": 300, "n_eval": 1000,
    "z_one_sided": 1.645, "delta_control": 0.005,
    "seed": 20260823,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def build_what_fn(env, kind, feat, beta, rng, cfg):
    """The stressed deployment's estimator, fitted per draw."""
    if kind == "no_correction":
        return lambda pool: np.ones(len(pool.s))
    n_fit = {"starvation50": 50, "starvation100": 100}.get(kind, 1000)
    if kind == "deprivation":
        keep = [j for j in range(13) if j != SERIAL]
    elif kind == "double_deprivation":
        keep = [j for j in range(13) if j not in (SERIAL, PROXY)]
    else:
        keep = list(range(13))
    Xs, _ = env.draw_instances(rng, n_fit)
    Xt, _ = env.draw_instances(rng, n_fit, beta, feat)
    w_fn = shift.fit_ratio(Xs[:, keep], Xt[:, keep],
                           clip=tuple(cfg["deployed_clip"]),
                           C=cfg["deployed_C"])
    return lambda pool: w_fn(pool.X[:, keep])


def main():
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    env = ReturnsEnv.induce()
    r = CONFIG["recipe"]
    rng = np.random.default_rng([CONFIG["seed"], 990])
    aud = Audit(env, rng, r["n_audit"])
    rows = []
    for spec in CONFIG["stress_cells"]:
        kind, feat, beta = spec["kind"], spec["feature"], spec["beta"]
        outs = []
        for _ in range(CONFIG["n_envelope_draws"]):
            what_fn = build_what_fn(env, kind, feat, beta, rng, CONFIG)
            outs.append(certificate(env, aud, CONFIG["alpha"], rng, feat,
                                    beta, what_fn, n_cal=CONFIG["n_cal"],
                                    n_src=r["n_src"], n_tgt=r["n_tgt"],
                                    n_lambda=CONFIG["n_lambda"], z=r["z"],
                                    n_bins=r["n_bins"]))
        b = [o["excess_bound"] for o in outs]
        row = {"kind": kind, "feature": feat, "beta": beta,
               "bound_median": float(np.median(b)),
               "bound_p10": float(np.percentile(b, 10)),
               "bound_p90": float(np.percentile(b, 90)),
               "a_plugin_median": float(np.median([o["a_plugin"] for o in outs])),
               "cal_err_median": float(np.median([o["cal_err_loc"] for o in outs])),
               "cal_err_conf_median": float(np.median([o["cal_err_conf"] for o in outs])),
               "b_own_median": float(np.median([o["b_own_ucb"] for o in outs])),
               "draws_bound": [round(float(x), 6) for x in b]}
        rows.append(row)
        print(f"[env-s] {kind:<19} {feat:<15} beta={beta}: alpha_cert = "
              f"{CONFIG['alpha'] + row['bound_median']:.4f} "
              f"(bound {row['bound_median']:+.4f})", flush=True)
    d = ROOT / "artifacts" / f"wp2envs_{config_hash(CONFIG)}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(CONFIG, indent=2))
    (d / "envelope.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
