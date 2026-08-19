"""WP2 Phase 3: registered in-sample validation of the certificate.

For every archived evidence-tier estimated-arm cell (the five WP1
collapse batteries and the WP2 Phase 0 battery: 674 cells), compute the
own-rule certificate from deployment-visible inputs over independent
deployment draws, and test coverage of the archived population excess.

Run:   python experiments/wp2_certificate/run.py
Smoke: add --smoke 3 (draws per cell)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import crc, shift, tests                 # noqa: E402
from cus.certificate import Audit, certificate    # noqa: E402
from cus.envs.claims import ClaimsEnv             # noqa: E402
from cus.envs.claims_logit import ClaimsLogitEnv  # noqa: E402
from cus.envs.spike import SpikeEnv               # noqa: E402
from cus.envs.family import GenEnv                # noqa: E402


CONFIG = {
    "experiment": "wp2_certificate",
    "batteries": {
        "wp1mc_56704982681d6960": "claims",
        "wp1mf_tickets_85e921864acbedcc": "tickets",
        "wp1mf_fraud_047610e36449b1c7": "fraud",
        "wp1mf_moderation_037ea765a0016581": "moderation",
        "wp1mf_compliance_c3609a222408ec0f": "compliance",
        "wp2p0_ec15383b39b52206": "multi",
    },
    "n_draws": 30, "cover_rule": 27,
    "n_cal": 1000, "n_src": 10000, "n_tgt": 10000, "n_audit": 60000,
    "n_lambda": 400, "n_bins": 10, "z": 1.645,
    "ratio_C": 1.0,
    "seed": 20260821,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp2_certificate.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def make_env(name):
    if name == "claims":
        return ClaimsEnv.induce()
    if name == "claims_logit":
        return ClaimsLogitEnv.induce()
    if name == "spike":
        return SpikeEnv.induce()
    return GenEnv.induce(name)


def tilt_dim(env, ename, feat):
    if ename == "claims" or ename == "claims_logit":
        from cus.envs.claims import BOUNDED
        return BOUNDED[feat]
    if ename == "spike":
        from cus.envs.spike import BOUNDED
        return BOUNDED[feat]
    return env.spec.bounded[feat]


def build_what_fn(env, ename, feat, beta, kind, setting, rng, cfg):
    """The cell's estimator, applied per pool. Fit-based estimators are
    fitted per deployment draw with the draw's rng."""
    k = tilt_dim(env, ename, feat)
    if kind == "temper":
        g = setting["gamma"]
        return lambda pool: np.exp(env.tilt_logweight(pool.X, beta, feat)) ** g
    if kind == "directional":
        d = setting["delta"]
        return lambda pool: (np.exp(env.tilt_logweight(pool.X, beta, feat))
                             * np.exp(d * pool.X[:, k]))
    n_fit = setting.get("n_fit", 1000)
    clip = tuple(setting.get("clip", (0.01, 0.99)))
    nf = None

    def view(X):
        nonlocal nf
        if nf is None:
            nf = X.shape[1]
        if kind == "deprivation":
            rho = setting["view_rho"]
            if rho == 1.0:
                return X
            if rho == 0.0:
                keep = [j for j in range(nf) if j != k]
                return X[:, keep]
            X = X.copy()
            X[:, k] = rho * X[:, k] + np.sqrt(1 - rho ** 2) * \
                rng.standard_normal(len(X))
            return X
        if kind == "inflation":
            extra = setting["extra_dims"]
            if extra == 0:
                return X
            return np.hstack([X, rng.standard_normal((len(X), extra))])
        return X                                    # starvation, mismatch
    Xs, _ = env.draw_instances(rng, n_fit)
    Xt, _ = env.draw_instances(rng, n_fit, beta, feat)
    w_fn = shift.fit_ratio(view(Xs), view(Xt), clip=clip, C=cfg["ratio_C"])
    return lambda pool: w_fn(view(pool.X))


def iter_cells(cfg):
    for bat, ename in cfg["batteries"].items():
        res = json.loads((ROOT / "artifacts" / bat / "results.json").read_text())
        bcfg = json.loads((ROOT / "artifacts" / bat / "config.json").read_text())
        cells = res["cells"] if isinstance(res, dict) and "cells" in res else res
        if ename != "multi":
            feat = bcfg["tilt_feature"]
            for c in cells:
                if c.get("arm") != "estimated":
                    continue
                setting = c["setting"]
                key, val = setting.split("=", 1)
                if key == "clip":
                    lo, hi = val.split("-")
                    sd = {"clip": [float(lo), float(hi)]}
                    kind = "mismatch"
                else:
                    cast = float if "." in val or key in ("gamma", "delta", "view_rho") else int
                    sd = {key: cast(val)}
                    kind = {"gamma": "temper", "delta": "directional",
                            "view_rho": "deprivation", "n_fit": "starvation",
                            "extra_dims": "inflation"}[key]
                yield (bat, ename, feat, c["beta"], 0.10, kind, sd,
                       c["marginal_risk_mean"] - 0.10)
        else:
            espec = bcfg["environments"]
            for c in cells:
                sd = json.loads(c["setting"])
                feat = espec[c["env"]]["tilt_feature"]
                yield (bat, c["env"], feat, c["beta"], c["alpha"], c["kind"],
                       sd, c["risk_est_mean"] - c["alpha"])


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration()
    n_draws = smoke if smoke is not None else CONFIG["n_draws"]
    print(f"[wp2cert] {'PILOT SMOKE' if smoke else 'REAL (evidence tier)'}"
          f" {n_draws} draws/cell")
    envs, audits = {}, {}
    rows = []
    t00 = time.time()
    for i, (bat, ename, feat, beta, alpha, kind, sd, exc) in \
            enumerate(iter_cells(CONFIG)):
        if ename not in envs:
            envs[ename] = make_env(ename)
            arng = np.random.default_rng([CONFIG["seed"], 900,
                                          abs(hash(ename)) % 10000])
            audits[ename] = Audit(envs[ename], arng, CONFIG["n_audit"])
        env, aud = envs[ename], audits[ename]
        rng = np.random.default_rng([CONFIG["seed"], 901, i])
        bounds, parts = [], []
        for _ in range(n_draws):
            what_fn = build_what_fn(env, ename, feat, beta, kind, sd, rng,
                                    CONFIG)
            out = certificate(env, aud, alpha, rng, feat, beta, what_fn,
                              n_cal=CONFIG["n_cal"], n_src=CONFIG["n_src"],
                              n_tgt=CONFIG["n_tgt"],
                              n_lambda=CONFIG["n_lambda"], z=CONFIG["z"],
                              n_bins=CONFIG["n_bins"])
            bounds.append(out["excess_bound"])
            parts.append(out)
        bounds = np.asarray(bounds)
        covered = int((bounds >= exc).sum())
        rows.append({
            "battery": bat, "env": ename, "alpha": alpha, "kind": kind,
            "setting": json.dumps(sd), "beta": beta,
            "excess_archived": exc,
            "bound_median": float(np.median(bounds)),
            "bound_mean": float(bounds.mean()),
            "bound_min": float(bounds.min()),
            "covered_draws": covered, "n_draws": n_draws,
            "cell_covered": covered >= (CONFIG["cover_rule"] if smoke is None
                                        else n_draws),
            "b_own_ucb_mean": float(np.mean([p["b_own_ucb"] for p in parts])),
            "cal_err_mean": float(np.mean([p["cal_err_loc"] for p in parts])),
            "a_plugin_mean": float(np.mean([p["a_plugin"] for p in parts])),
        })
        if i % 50 == 0:
            print(f"[wp2cert] {i:>4} cells done ({time.time() - t00:.0f}s)",
                  flush=True)
    cov = sum(r["cell_covered"] for r in rows)
    pos = [r for r in rows if r["excess_archived"] >= 0.005]
    within2 = sum(r["bound_median"] <= 2 * r["excess_archived"] for r in pos)
    price = float(np.mean([r["bound_mean"] - r["excess_archived"]
                           for r in rows]))
    print(f"[wp2cert] coverage {cov}/{len(rows)} cells; "
          f"within-2x on positive cells {within2}/{len(pos)}; "
          f"mean conservatism {price:+.4f}")
    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp2cert_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps(
        {"rows": rows, "coverage": cov, "n_cells": len(rows),
         "within2x_pos": within2, "n_pos": len(pos),
         "mean_conservatism": price}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
