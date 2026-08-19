"""WP1 Ablation B: estimator misspecification. THE LOAD-BEARING SWEEP.

The corrected pilot showed the well-specified logistic ratio estimator
tracking the oracle to the third decimal, so Question 1 (Proposition 2 with
an ESTIMATED ratio, the case Angelopoulos et al. Section 4.1 leave open) had
no bite. This experiment gives it bite: shift is held fixed at two levels
and the estimator is degraded deliberately along four axes, with the full
guarantee-relevant error battery recorded per cell.

AXES (deployment analogue in parentheses)
  deprivation  fit_ratio's view of the tilted dim is degraded: rho = 1.0 is
               the full view, rho in (0,1) observes rho*x_2 + sqrt(1-rho^2)*
               noise (attribute logged with error), rho = 0.0 drops the
               column entirely, X[:, mask] (drifting attribute unlogged).
  inflation    d in {5, 20, 50, 100} nuisance dims at fixed n_fit
               (wide manifests, small ratio-fit budget).
  starvation   n_fit in {50, 100, 250, 1000} at d = 5
               (little unlabelled target data).
  mismatch     tilt on the BOUNDED nonlinear feature exp(beta * tanh(x_2)),
               sampled EXACTLY by rejection (rung 2, self-checked at
               startup), while the estimator stays linear-logistic on raw X;
               clip swept (model class wrong, clip knob caps expressible
               shift).

ARMS: oracle (Prop 2, true ratio) and estimated (Prop 2, w_hat) only.

SAMPLING RUNGS. Linear-tilt axes use rung 1 (exact analytic tilt, target
N(b, I)). The mismatch axis uses rung 2 (exact rejection on the bounded
tilt); the runner refuses to start if the importance-weighted-moment
self-check fails.

DELIVERABLE. Excess marginal risk vs normalized-weight L1(P0) error, points
pooled across axes, colored by axis, with y = B*L1 overlaid as the envelope
candidate. This figure is the WP2 target.

Run:   python experiments/wp1_misspecification/run.py
Smoke: python experiments/wp1_misspecification/run.py --smoke 20
       (n_trials override for pipeline debugging only; output dir is
       suffixed _smokeN and smoke results are never cited)
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
    "experiment": "wp1_misspecification",
    "alpha": 0.10,
    "n_cal": 1000,
    "n_eval": 1000,
    "n_trials": 200,
    "n_lambda": 400,
    "base_d": 5,
    "tilt_dim": 2,
    "beta_scales": [0.75, 1.25],
    "weight_estimator": "logistic",
    "default_clip": [0.01, 0.99],
    "default_n_fit": 1000,
    "axes": {
        "deprivation": {"view_rho": [1.0, 0.7, 0.35, 0.0]},
        "inflation": {"d": [5, 20, 50, 100]},
        "starvation": {"n_fit": [50, 100, 250, 1000]},
        "mismatch": {
            "tilt": "exp(beta * tanh(x_tilt_dim))",
            "sampling_rung": 2,
            "clip": [[0.01, 0.99], [0.05, 0.95], [0.2, 0.8]],
        },
    },
    "z_one_sided": 1.645,
    "delta_control": 0.005,
    "delta_oracle": 0.005,
    "seed": 20260818,
}

ARMS = ("oracle", "estimated")


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


def iter_cells(cfg: dict):
    """Yield (axis, axis_idx, setting_idx, setting_dict, beta_scale)."""
    for ai, (axis, spec) in enumerate(cfg["axes"].items()):
        if axis == "deprivation":
            settings = [{"view_rho": r} for r in spec["view_rho"]]
        elif axis == "inflation":
            settings = [{"d": d} for d in spec["d"]]
        elif axis == "starvation":
            settings = [{"n_fit": n} for n in spec["n_fit"]]
        else:  # mismatch
            settings = [{"clip": c} for c in spec["clip"]]
        for si, setting in enumerate(settings):
            for scale in cfg["beta_scales"]:
                yield axis, ai, si, setting, scale


def setting_label(axis: str, setting: dict) -> str:
    if axis == "deprivation":
        return f"rho={setting['view_rho']}"
    if axis == "inflation":
        return f"d={setting['d']}"
    if axis == "starvation":
        return f"nfit={setting['n_fit']}"
    lo, hi = setting["clip"]
    return f"clip={lo}-{hi}"


def make_view(axis: str, setting: dict, cfg: dict, rng: np.random.Generator):
    """The estimator's (possibly degraded) view of the covariates.

    Applied consistently at ratio-fit time AND when w_hat is evaluated on
    calibration and evaluation covariates: a deployment that does not log an
    attribute does not log it anywhere. Noise draws are fresh per call, the
    logged-with-error analogue.
    """
    if axis != "deprivation":
        return lambda X: X
    rho = setting["view_rho"]
    k = cfg["tilt_dim"]
    if rho == 1.0:
        return lambda X: X
    if rho == 0.0:
        keep = [j for j in range(cfg["base_d"]) if j != k]
        return lambda X: X[:, keep]

    def view(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[:, k] = rho * X[:, k] + np.sqrt(1.0 - rho ** 2) * \
            rng.standard_normal(len(X))
        return X

    return view


def run_cell(axis: str, ai: int, si: int, setting: dict, scale: float,
             cfg: dict, n_trials: int):
    d = setting.get("d", cfg["base_d"])
    n_fit = setting.get("n_fit", cfg["default_n_fit"])
    clip = tuple(setting.get("clip", cfg["default_clip"]))
    k = cfg["tilt_dim"]
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda"])
    # SeedSequence-style entropy list keeps every cell's stream distinct.
    rng = np.random.default_rng([cfg["seed"], ai, si, int(scale * 1000)])
    view = make_view(axis, setting, cfg, rng)

    if axis == "mismatch":
        rung = 2
        chi2 = shift.tanh_tilt_chi2(scale)
        w_true_fn = lambda X: shift.tanh_tilt_ratio(X, scale, k)     # noqa: E731
        draw_target_X = lambda n: shift.rejection_tilt_draw(          # noqa: E731
            rng, n, d, scale, k)
    else:
        rung = 1
        b = np.zeros(d)
        b[k] = scale
        chi2 = shift.gaussian_tilt_chi2(b)
        w_true_fn = lambda X: shift.gaussian_tilt_ratio(X, b)         # noqa: E731
        draw_target_X = lambda n: rng.standard_normal((n, d)) + b     # noqa: E731

    trials = {arm: [] for arm in ARMS}
    battery, paired_diff = [], []

    for _ in range(n_trials):
        # Four fresh, mutually independent splits. Sampling unit: one case.
        cal = synth.draw_cases(rng, cfg["n_cal"], d, mean=None)
        ev = synth.realize(draw_target_X(cfg["n_eval"]), rng)
        Xs_fit = rng.standard_normal((n_fit, d))
        Xt_fit = draw_target_X(n_fit)

        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        w_cal_true = w_true_fn(cal.X)
        w_ev_true = w_true_fn(ev.X)

        risks = {}
        for arm in ARMS:
            if arm == "oracle":
                lam = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                                     w_cal_true, w_ev_true)
                w_used = w_cal_true
            else:
                w_fn = shift.fit_ratio(view(Xs_fit), view(Xt_fit),
                                       method=cfg["weight_estimator"],
                                       clip=clip)
                w_cal_hat, w_ev_hat = w_fn(view(cal.X)), w_fn(view(ev.X))
                lam = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                                     w_cal_hat, w_ev_hat)
                w_used = w_cal_hat
                battery.append(shift.ratio_error_battery(w_cal_hat, w_cal_true))

            res = metrics.evaluate(ev.s, ev.wrong, ev.region, lam, cfg["alpha"])
            res["ess"] = crc.effective_sample_size(w_used)
            risks[arm] = res["marginal_risk"]
            trials[arm].append(res)
        paired_diff.append(risks["estimated"] - risks["oracle"])

    diff = np.asarray(paired_diff)
    dm = float(diff.mean())
    dse = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    ci = (dm - cfg["z_one_sided"] * dse, dm + cfg["z_one_sided"] * dse)

    rows = []
    for arm in ARMS:
        summ = metrics.summarise(trials[arm])
        summ["verdict"] = verdict(summ["marginal_risk_mean"],
                                  summ["marginal_risk_se"], cfg)
        if arm == "estimated":
            for key in battery[0]:
                summ[f"ratio_{key}_mean"] = float(np.mean([x[key] for x in battery]))
            summ["paired_diff_mean"] = dm
            summ["paired_diff_se"] = dse
            summ["paired_diff_ci90"] = [ci[0], ci[1]]
            summ["oracle_equiv"] = bool(
                -cfg["delta_oracle"] <= ci[0] and ci[1] <= cfg["delta_oracle"])
            z = cfg["z_one_sided"]
            summ["envelope_ok"] = bool(
                summ["excess_marginal_risk_mean"] - z * summ["marginal_risk_se"]
                <= summ["ratio_w_l1_mean"])
        rows.append({"axis": axis, "setting": setting_label(axis, setting),
                     "beta_scale": scale, "chi2": chi2, "sampling_rung": rung,
                     **{f"setting_{k2}": v for k2, v in setting.items()},
                     "arm": arm, **summ})
    return rows


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])

    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Proposition 2 reduces to unweighted at w == 1: PASS")
    for scale in CONFIG["beta_scales"]:
        rep = shift.rejection_exactness_check(
            np.random.default_rng([CONFIG["seed"], 999, int(scale * 1000)]),
            beta=scale, dim=CONFIG["tilt_dim"], d=CONFIG["base_d"])
        if not rep["ok"]:
            raise SystemExit(f"[selftest] rung-2 exactness FAILED at beta={scale}: {rep}")
        print(f"[selftest] rung-2 rejection exact at beta={scale}: PASS")
    check_registration(CONFIG, root / "registrations")

    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    tag = "PILOT SMOKE" if smoke else "pilot (placeholder env)"
    print(f"[wp1m] {tag}: n_trials={n_trials}")

    all_rows = []
    for axis, ai, si, setting, scale in iter_cells(CONFIG):
        rows = run_cell(axis, ai, si, setting, scale, CONFIG, n_trials)
        all_rows.extend(rows)
        for r in rows:
            extra = ""
            if r["arm"] == "estimated":
                eq = "equiv" if r["oracle_equiv"] else "NOT-equiv"
                env = "env-ok" if r["envelope_ok"] else "ENV-BREACH"
                extra = (f" L1={r['ratio_w_l1_mean']:.3f} {eq} {env}"
                         f" dRisk={r['paired_diff_mean']:+.4f}")
            print(f"[wp1m] {r['axis']:<11} {r['setting']:<14} beta={r['beta_scale']:<5}"
                  f" chi2={r['chi2']:5.2f} {r['arm']:<9}"
                  f" risk={r['marginal_risk_mean']:.4f}±{r['marginal_risk_se']:.4f}"
                  f" [{r['verdict']:<12}] defer={r['deferral_rate_mean']:.3f}"
                  f" ess={r['ess_mean']:7.1f}{extra}")

    suffix = f"_smoke{smoke}" if smoke else ""
    out_dir = root / "artifacts" / f"wp1m_{config_hash(CONFIG)}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**CONFIG, "n_trials_effective": n_trials, "smoke": bool(smoke)}, indent=2))
    (out_dir / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\n[out] {out_dir}")


if __name__ == "__main__":
    main()
