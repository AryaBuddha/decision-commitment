"""Block C battery: temper collapse on the gated spike environment
(adversarial kappa). EVIDENCE TIER.

Run:   python experiments/wp1_battery_spike/run.py
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

from cus import crc, tests             # noqa: E402
from cus.envs.spike import SpikeEnv    # noqa: E402


CONFIG = {
    "experiment": "wp1_battery_spike",
    "environment": "spike",
    "env_freeze_gate_hash": "4bab082db8eba24c",
    "alpha": 0.10,
    "n_cal": 1000, "n_eval": 1000, "n_trials": 300, "n_lambda": 400,
    "tilt_feature": "b",
    "temper_betas": [3.854, 7.008],
    "temper_gammas": [0.0, 0.25, 0.5, 0.75, 0.8, 1.0, 1.25, 1.5],
    "fd_window": 0.05,
    "min_abs_a": 0.003,
    "min_slope_what": 0.02,
    "z_one_sided": 1.645,
    "seed": 20260819,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp1_battery_spike.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Prop 2 reduction: PASS")
    check_registration()
    env = SpikeEnv.induce()
    n = smoke if smoke is not None else CONFIG["n_trials"]
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    alpha = CONFIG["alpha"]
    feat = CONFIG["tilt_feature"]
    h = CONFIG["fd_window"]
    cells = []
    for bi, beta in enumerate(CONFIG["temper_betas"]):
        for gi, gamma in enumerate(CONFIG["temper_gammas"]):
            rng = np.random.default_rng([CONFIG["seed"], 600, gi, bi])
            aligned, paired, ro_, sw, sh = [], [], [], [], []
            for _ in range(n):
                cal = env.case_table(rng, CONFIG["n_cal"])
                ev = env.case_table(rng, CONFIG["n_eval"], beta=beta, feature=feat)
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
                Llo = ((cal.s >= ls - h) & cal.wrong).astype(float)
                Lhi = ((cal.s >= ls + h) & cal.wrong).astype(float)
                sw.append(float(np.mean(wn * (Llo - Lhi)) / (2 * h)))
                sh.append(float(np.mean(hn * (Llo - Lhi)) / (2 * h)))
            mw, mh = float(np.mean(sw)), float(np.mean(sh))
            cells.append({
                "gamma": gamma, "beta": beta,
                "aligned_mean": float(np.mean(aligned)),
                "paired_diff_mean": float(np.mean(paired)),
                "paired_diff_se": float(np.std(paired, ddof=1) / np.sqrt(len(paired))),
                "oracle_excess": float(np.mean(ro_) - alpha),
                "kappa_pred": mw / mh if mh > CONFIG["min_slope_what"] else None})
            kp = cells[-1]["kappa_pred"]
            print(f"[wp1sp] gamma={gamma:<5} beta={beta:<6} "
                  f"a={cells[-1]['aligned_mean']:+.4f} "
                  f"pd={cells[-1]['paired_diff_mean']:+.4f} "
                  f"kappa_pred={kp if kp is None else round(kp, 3)}")

    x = np.array([c["aligned_mean"] for c in cells])
    y = np.array([c["paired_diff_mean"] for c in cells])
    m = np.abs(x) > 1e-12
    slope = float(x[m] @ y[m] / (x[m] @ x[m]))
    r = y[m] - slope * x[m]
    sse = float(np.sqrt((r @ r) / (m.sum() - 1) / (x[m] @ x[m])))

    guard = [c for c in cells if c["kappa_pred"] is not None
             and abs(c["aligned_mean"]) >= CONFIG["min_abs_a"]]
    kp = np.array([c["kappa_pred"] for c in guard])
    km = np.array([c["paired_diff_mean"] / c["aligned_mean"] for c in guard])
    b = float(kp @ km / (kp @ kp))
    rb = km - b * kp
    seb = float(np.sqrt((rb @ rb) / (len(kp) - 1) / (kp @ kp)))
    r2 = 1 - float((rb @ rb)) / float(((km - km.mean()) ** 2).sum())
    oexc = float(np.mean([c["oracle_excess"] for c in cells]))
    print(f"[wp1sp] pooled pd_slope={slope:.3f}±{sse:.3f}  "
          f"per-cell regression: slope={b:.3f}±{seb:.3f} R2={r2:.3f} "
          f"(n={len(kp)})  oracle_excess={oexc:+.4f}")
    summary = {"pd_slope": slope, "pd_slope_se": sse,
               "percell_slope": b, "percell_slope_se": seb,
               "percell_r2": r2, "n_guarded": len(kp),
               "oracle_excess_mean": oexc}
    suffix = f"_smoke{smoke}" if smoke else ""
    d = ROOT / "artifacts" / f"wp1sp_{config_hash(CONFIG)}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({**CONFIG, "smoke": bool(smoke)}, indent=2))
    (d / "results.json").write_text(json.dumps({"cells": cells, "summary": summary}, indent=2))
    print(f"\n[out] {d}")


if __name__ == "__main__":
    main()
