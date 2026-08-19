"""WP2 Phase 5, stage A: the certificate's predicted safe operating
envelope on the HOLDOUT environment (returns), computed from
deployment-visible data ONLY, before any realized-risk sweep exists.

For each (tilt feature, beta level), the deployment's own pipeline runs:
fit the ratio estimator (logistic, full manifest view, the ledgered
defaults), compute the Prop-2 thresholds, and assemble the certificate
(recipe as revised by red-team round 2). The resulting alpha_cert
envelope goes verbatim into the Phase 5 registration as the prediction;
stage B (run.py) then sweeps realized risk against it.

Run: python experiments/wp2_prospective/envelope.py
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


CONFIG = {
    "experiment": "wp2_prospective",
    "environment": "returns",
    "features": {"serial_rate": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                 "desc_vagueness": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
    "alpha": 0.10,
    "deployed_estimator": {"method": "logistic", "view": "full",
                           "clip": [0.01, 0.99], "C": 1.0, "n_fit": 1000},
    "recipe": {"n_bins": 20, "audit_C": 1.0, "n_audit": 60000,
               "n_src": 10000, "n_tgt": 10000, "z": 1.645},
    "n_cal": 1000, "n_lambda": 400,
    "n_envelope_draws": 30,
    "n_trials_sweep": 300, "n_eval": 1000,
    "z_one_sided": 1.645, "delta_control": 0.005,
    "seed": 20260822,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def main():
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    env = ReturnsEnv.induce()
    r = CONFIG["recipe"]
    rng = np.random.default_rng([CONFIG["seed"], 970])
    aud = Audit(env, rng, r["n_audit"])
    rows = []
    for feat, betas in CONFIG["features"].items():
        for beta in betas:
            def what_fn(pool, _rng=rng, _b=beta, _f=feat):
                Xs, _ = env.draw_instances(_rng, CONFIG["deployed_estimator"]["n_fit"])
                Xt, _ = env.draw_instances(_rng, CONFIG["deployed_estimator"]["n_fit"], _b, _f)
                return shift.fit_ratio(
                    Xs, Xt, clip=tuple(CONFIG["deployed_estimator"]["clip"]),
                    C=CONFIG["deployed_estimator"]["C"])(pool.X)
            outs = [certificate(env, aud, CONFIG["alpha"], rng, feat, beta,
                                what_fn, n_cal=CONFIG["n_cal"],
                                n_src=r["n_src"], n_tgt=r["n_tgt"],
                                n_lambda=CONFIG["n_lambda"], z=r["z"],
                                n_bins=r["n_bins"])
                    for _ in range(CONFIG["n_envelope_draws"])]
            b = [o["excess_bound"] for o in outs]
            row = {"feature": feat, "beta": beta,
                   "bound_median": float(np.median(b)),
                   "bound_p10": float(np.percentile(b, 10)),
                   "bound_p90": float(np.percentile(b, 90)),
                   "a_plugin_median": float(np.median([o["a_plugin"] for o in outs])),
                   "cal_err_median": float(np.median([o["cal_err_loc"] for o in outs])),
                   "b_own_median": float(np.median([o["b_own_ucb"] for o in outs]))}
            rows.append(row)
            print(f"[env] {feat:<15} beta={beta}: alpha_cert = "
                  f"{CONFIG['alpha'] + row['bound_median']:.4f} "
                  f"(bound {row['bound_median']:+.4f})", flush=True)
    d = ROOT / "artifacts" / f"wp2env_{config_hash(CONFIG)}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(CONFIG, indent=2))
    (d / "envelope.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
