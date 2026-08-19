"""WP2 Phase 4, round 2: attack the REVISED certificate. EVIDENCE TIER.

  R2-1  re-attack the freshness window (Revision 1): tickets audited at
        beta 3.589, deployed at +1.2, +2.4 (the declared tolerance
        edge), and +3.6 beta-units. The monitor must order the drifts,
        the inside-window deployment must stay covered, and the
        outside-window deployment must be FLAGGED, converting round 1's
        silent failure into a detected one.
  R2-2  the B4 lesson hunted inside the certificate: the audit model's
        own sklearn L2 default (C = 1.0, unledgered until now). Sweep
        audit C in {1.0, unregularized} at two cells.
  R2-3  the binning default: n_bins in {5, 10, 20} at the same cells.

Run:   python experiments/wp2_redteam2/run.py
Smoke: add --smoke 5
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import crc, shift, tests                          # noqa: E402
from cus.certificate import Audit, certificate, drift_monitor  # noqa: E402
from cus.envs.claims import ClaimsEnv                      # noqa: E402
from cus.envs.family import GenEnv                         # noqa: E402


CONFIG = {
    "experiment": "wp2_redteam2",
    "alpha": 0.10,
    "n_cal": 1000, "n_src": 10000, "n_tgt": 10000, "n_audit": 60000,
    "n_lambda": 400,
    "r21": {"env": "tickets", "feature": "frustration",
            "beta_audit": 3.589, "betas_deploy": [4.8, 5.982, 7.178],
            "n_draws": 30, "n_excess_trials": 200, "n_monitor": 10000},
    "r22_cells": [
        {"env": "claims", "feature": "inconsistency", "beta": 6.5,
         "estimator": "blind_deprived"},
        {"env": "tickets", "feature": "frustration", "beta": 5.982,
         "estimator": "temper0"},
    ],
    "audit_Cs": [1.0, None],
    "n_bins_sweep": [5, 10, 20],
    "n_draws_defaults": 15,
    "seed": 20260821,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp2_redteam2.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


class AuditC(Audit):
    def __init__(self, env, rng, n_audit, C):
        pool = env.case_table(rng, n_audit)
        dec, _ = env.route(pool.X)
        if C is None:
            self.clf = LogisticRegression(max_iter=5000, penalty=None)
        else:
            self.clf = LogisticRegression(max_iter=2000, C=C)
        self.clf.fit(self._feats(pool.X, pool.s, dec), pool.wrong)


def make_env(name):
    return ClaimsEnv.induce() if name == "claims" else GenEnv.induce(name)


def estimator(env, ename, feat, beta, kind, rng):
    if kind == "temper0":
        return lambda pool: np.ones(len(pool.s))
    k = (8 if ename == "claims" else env.spec.bounded[feat])

    def blind(pool):
        keep = [j for j in range(pool.X.shape[1]) if j != k]
        Xs, _ = env.draw_instances(rng, 1000)
        Xt, _ = env.draw_instances(rng, 1000, beta, feat)
        return shift.fit_ratio(Xs[:, keep], Xt[:, keep])(pool.X[:, keep])
    return blind


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration()
    alpha = CONFIG["alpha"]
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    scd = (lambda n: min(n, smoke) if smoke else n)
    out = {}

    # ---- R2-1 ----
    c = CONFIG["r21"]
    env = make_env(c["env"])
    rng = np.random.default_rng([CONFIG["seed"], 960])
    aud = Audit(env, rng, CONFIG["n_audit"])

    def stale(pool):
        Xs, _ = env.draw_instances(rng, 1000)
        Xt, _ = env.draw_instances(rng, 1000, c["beta_audit"], c["feature"])
        return shift.fit_ratio(Xs, Xt)(pool.X)
    rows = []
    for bd in c["betas_deploy"]:
        exc = []
        for _ in range(scd(c["n_excess_trials"])):
            cal = env.case_table(rng, CONFIG["n_cal"])
            ev = env.case_table(rng, 1000, beta=bd, feature=c["feature"])
            wh, whe = stale(cal), stale(ev)
            losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
            lam = crc.lhat_prop2(losses, lambdas, alpha, wh, whe)
            exc.append(float(((ev.s >= np.asarray(lam)) & ev.wrong).mean()) - alpha)
        e = float(np.mean(exc))
        bounds, monitors = [], []
        for _ in range(scd(c["n_draws"])):
            res = certificate(env, aud, alpha, rng, c["feature"],
                              c["beta_audit"], stale,
                              n_src=CONFIG["n_src"], n_tgt=CONFIG["n_tgt"])
            Xa, _ = env.draw_instances(rng, c["n_monitor"], c["beta_audit"],
                                       c["feature"])
            Xd, _ = env.draw_instances(rng, c["n_monitor"], bd, c["feature"])
            bounds.append(res["excess_bound"])
            monitors.append(drift_monitor(Xa, Xd))
        rows.append({"beta_deploy": bd, "excess": e,
                     "bound_median": float(np.median(bounds)),
                     "covered_draws": int(sum(b >= e for b in bounds)),
                     "n_draws": len(bounds),
                     "monitor_median": float(np.median(monitors)),
                     "monitor_p10": float(np.percentile(monitors, 10)),
                     "monitor_p90": float(np.percentile(monitors, 90))})
        print(f"[rt2-1] deploy {bd}: excess {e:+.4f} "
              f"bound_med {rows[-1]['bound_median']:+.4f} "
              f"covered {rows[-1]['covered_draws']}/{rows[-1]['n_draws']} "
              f"monitor {rows[-1]['monitor_median']:.3f}", flush=True)
    out["r21"] = rows

    # ---- R2-2 and R2-3 ----
    out["r22"], out["r23"] = [], []
    for spec in CONFIG["r22_cells"]:
        env = make_env(spec["env"])
        rng = np.random.default_rng([CONFIG["seed"], 961,
                                     abs(hash(spec["env"])) % 1000])
        what_fn = estimator(env, spec["env"], spec["feature"], spec["beta"],
                            spec["estimator"], rng)
        for C in CONFIG["audit_Cs"]:
            audC = AuditC(env, np.random.default_rng([CONFIG["seed"], 962]),
                          CONFIG["n_audit"], C)
            bs = [certificate(env, audC, alpha, rng, spec["feature"],
                              spec["beta"], what_fn, n_src=CONFIG["n_src"],
                              n_tgt=CONFIG["n_tgt"])["excess_bound"]
                  for _ in range(scd(CONFIG["n_draws_defaults"]))]
            out["r22"].append({"cell": spec["env"], "C": "none" if C is None
                               else C, "bound_median": float(np.median(bs))})
            print(f"[rt2-2] {spec['env']} auditC={C}: "
                  f"bound_med {np.median(bs):+.4f}", flush=True)
        aud = Audit(env, np.random.default_rng([CONFIG["seed"], 962]),
                    CONFIG["n_audit"])
        for nb in CONFIG["n_bins_sweep"]:
            bs = [certificate(env, aud, alpha, rng, spec["feature"],
                              spec["beta"], what_fn, n_src=CONFIG["n_src"],
                              n_tgt=CONFIG["n_tgt"],
                              n_bins=nb)["excess_bound"]
                  for _ in range(scd(CONFIG["n_draws_defaults"]))]
            out["r23"].append({"cell": spec["env"], "n_bins": nb,
                               "bound_median": float(np.median(bs))})
            print(f"[rt2-3] {spec['env']} bins={nb}: "
                  f"bound_med {np.median(bs):+.4f}", flush=True)

    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2rt2_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
