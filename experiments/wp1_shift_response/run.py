"""WP1: shift-response curves. v3, revised after external review.

ARMS
  unweighted   Eq. (4) of Angelopoulos et al. (2022). Exchangeability assumed.
               Its decay is compared against the Proposition 3 TV envelope,
               E[L] <= alpha + B * sum_i TV(Z_i, Z_{n+1}), computable in
               closed form here; expected to be vacuous at n_cal = 1000,
               which is itself a reportable fact (the known bound does not
               explain the observed decay).
  oracle       Proposition 2, literal, per-test-covariate threshold, true
               ratio. THEOREM-BACKED. Any apparent failure is treated first
               as an implementation, sampling, dependence, or assumption
               error, never first as a counterexample.
  estimated    Proposition 2 with the classifier-odds ratio fitted on
               DEDICATED covariate splits, disjoint from calibration and
               evaluation. The subject of Q1, which Angelopoulos et al.
               Section 4.1 explicitly leave open.
  glob_oracle  The single-global-threshold shortcut with the true ratio.
               NOT Proposition 2. Ablation: what does the practical
               shortcut lose relative to the literal procedure?

SAMPLING. Exact analytic tilt: source P0 = N(0,I), target Q_b = N(b,I),
ratio w(x) = exp(b.x - |b|^2/2) exact and normalised. No pool, no
resampling approximation. Four fresh independent splits per trial:
labelled calibration (P0), ratio-fit source covariates (P0), ratio-fit
target covariates (Q_b), labelled evaluation (Q_b).

DECISION RULES (three outcomes, preregistered)
  violation              mean - z * SE > alpha        (one-sided, z = 1.645)
  consistent w/ control  mean + z * SE <= alpha + delta_control
  inconclusive           otherwise
Confirmatory claims on real environments apply Bonferroni across shift
levels; this placeholder pilot reports raw verdicts, labelled as such.

Run:  python experiments/wp1_shift_response/run.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cus import crc, metrics, shift, tests   # noqa: E402
from cus import synth2 as synth              # noqa: E402


CONFIG = {
    "experiment": "wp1_shift_response",
    "alpha": 0.10,
    "n_cal": 1000,
    "n_eval": 1000,
    "n_fit": 1000,
    "n_trials": 200,
    "n_lambda": 400,
    "d": 5,
    "tilt_direction": [0.0, 0.0, 1.0, 0.0, 0.0],
    "beta_scales": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "weight_estimator": "logistic",
    "weight_clip": [0.01, 0.99],
    "z_one_sided": 1.645,
    "delta_control": 0.005,
    "seed": 20260818,
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
        raise SystemExit(
            f"Config hash {h} does not match registered {reg.get('config_hash')}.\n"
            "Update the registration deliberately (with an amendment) or revert."
        )
    print(f"[prereg] config {h} matches registration")


def verdict(mean: float, se: float, cfg: dict) -> str:
    z, a, d = cfg["z_one_sided"], cfg["alpha"], cfg["delta_control"]
    if mean - z * se > a:
        return "VIOLATION"
    if mean + z * se <= a + d:
        return "consistent"
    return "inconclusive"


def run_cell(scale: float, cfg: dict):
    b = scale * np.array(cfg["tilt_direction"])
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda"])
    rng = np.random.default_rng(cfg["seed"] + int(scale * 1000))

    trials = {arm: [] for arm in ARMS}
    diag = []

    for _ in range(cfg["n_trials"]):
        # Four fresh, mutually independent splits. Sampling unit: one case.
        cal = synth.draw_cases(rng, cfg["n_cal"], cfg["d"], mean=None)
        ev = synth.draw_cases(rng, cfg["n_eval"], cfg["d"], mean=b)
        Xs_fit = rng.standard_normal((cfg["n_fit"], cfg["d"]))
        Xt_fit = rng.standard_normal((cfg["n_fit"], cfg["d"])) + b

        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        w_cal_true = shift.gaussian_tilt_ratio(cal.X, b)
        w_ev_true = shift.gaussian_tilt_ratio(ev.X, b)

        for arm in ARMS:
            if arm == "unweighted":
                lam = crc.lhat_unweighted(losses, lambdas, cfg["alpha"])
                w_used = np.ones(len(cal))
            elif arm == "oracle":
                lam = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                                     w_cal_true, w_ev_true)
                w_used = w_cal_true
            elif arm == "glob_oracle":
                lam = crc.lhat_weighted_global(losses, lambdas, cfg["alpha"],
                                               w_cal=w_cal_true,
                                               w_test=float(w_ev_true.mean()))
                w_used = w_cal_true
            else:  # estimated
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
            trials[arm].append(res)

    ana = {"chi2_analytic": shift.gaussian_tilt_chi2(b),
           "tv_analytic": shift.gaussian_tilt_tv(b),
           "prop3_envelope": cfg["alpha"] + 1.0 * cfg["n_cal"] * shift.gaussian_tilt_tv(b)}

    rows = []
    for arm in ARMS:
        summ = metrics.summarise(trials[arm])
        summ["verdict"] = verdict(summ["marginal_risk_mean"],
                                  summ["marginal_risk_se"], cfg)
        if arm == "estimated" and diag:
            for k in diag[0]:
                summ[f"ratio_{k}_mean"] = float(np.mean([x[k] for x in diag]))
        rows.append({"beta_scale": scale, "arm": arm, **ana, **summ})
    return rows


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Proposition 2 reduces to unweighted at w == 1: PASS")
    check_registration(CONFIG, root / "registrations")

    all_rows = []
    for scale in CONFIG["beta_scales"]:
        rows = run_cell(scale, CONFIG)
        all_rows.extend(rows)
        for r in rows:
            extra = (f" L1={r['ratio_w_l1_mean']:.3f}"
                     if "ratio_w_l1_mean" in r else "")
            print(f"[wp1] beta={r['beta_scale']:<5} chi2={r['chi2_analytic']:6.2f} "
                  f"{r['arm']:<11} risk={r['marginal_risk_mean']:.4f}"
                  f"±{r['marginal_risk_se']:.4f} [{r['verdict']:<12}] "
                  f"defer={r['deferral_rate_mean']:.3f} "
                  f"ess={r['ess_mean']:7.1f} "
                  f"r1={r.get('marginal_risk_region1_mean', float('nan')):.3f}"
                  f"{extra}")

    out_dir = root / "artifacts" / f"wp1_{config_hash(CONFIG)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(CONFIG, indent=2))
    (out_dir / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\n[out] {out_dir}")


if __name__ == "__main__":
    main()
