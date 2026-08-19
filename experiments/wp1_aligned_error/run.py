"""WP1 follow-up: is SIGNED ALIGNED weight error the one-dimensional
coordinate of estimated-shift risk control?

The misspecification sweep (fa8459eb3cb50722) answered its registered
questions: one curve does NOT map L1(P0) error to excess risk, and B*L1 is
a valid but >=5x-loose envelope. The residual structure suggested the true
driver is the signed alignment of the weight error with the loss on the
shifted direction. The four axes produced that alignment as a SIDE EFFECT
of realistic estimator failures; this experiment manipulates it DIRECTLY,
with synthetic estimators built from the oracle ratio and no classifier:

  temper       w_hat = w^gamma, gamma in [0, 1.5]. Traces a continuous path
               from unweighted CRC (gamma=0, the confirmed violation
               endpoint) through the oracle (gamma=1) into over-correction.
  directional  w_hat = w * exp(delta * x_tilt), signed delta. Isolates
               aligned error in both directions. Note both families live in
               the same exponential family along the tilt: the normalized
               estimator equals an oracle for effective tilt gamma*beta,
               resp. beta+delta; matched effective tilts must coincide
               (registered as an internal consistency check).

ALIGNED ERROR, preregistered definition, per trial:

    a = (1/n_cal) sum_i (w_norm(X_i) - w_hat_norm(X_i)) * L_i(lambda*)

with both weight vectors normalized to mean 1 on the calibration sample,
L_i(lambda*) = 1{s_i >= lambda* and wrong_i}, and lambda* the trial's mean
oracle Proposition 2 threshold over evaluation points. Positive a means the
estimator under-weights loss-bearing cases (the blindness direction).
|a| <= L1 holds by arithmetic; a violation is a code bug.

The runner also RECOMPUTES the 30 misspecification cells from their seeds
(identical rng streams, identical draw order) to extract per-trial aligned
error that the original run did not record. The recompute must reproduce
each stored estimated-arm cell mean exactly (tolerance 1e-9) or the run
aborts: the assertion is the proof that the replication is faithful. The
original artifact directory is never touched (analysis tier).

DELIVERABLE. Excess marginal risk vs signed aligned error, all 58 cells
pooled (30 misspec + 28 synthetic), with the pooled isotonic fit and the
preregistered collapse criteria. If the collapse holds, WP2's target
functional is identified.

Run:   python experiments/wp1_aligned_error/run.py
Smoke: python experiments/wp1_aligned_error/run.py --smoke 20
       (n_trials override for synthetic cells AND recompute-skip; smoke
       outputs are suffixed _smokeN and never cited)
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "wp1_misspecification"))
import run as misspec                        # noqa: E402


CONFIG = {
    "experiment": "wp1_aligned_error",
    "alpha": 0.10,
    "n_cal": 1000,
    "n_eval": 1000,
    "n_trials": 200,
    "n_lambda": 400,
    "d": 5,
    "tilt_dim": 2,
    "beta_scales": [0.75, 1.25],
    "families": {
        "temper": {"gamma": [0.0, 0.25, 0.5, 0.75, 0.8, 1.0, 1.25, 1.5]},
        "directional": {"delta": [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]},
    },
    "aligned_error_def": ("mean over calibration of (w_norm - w_hat_norm) * "
                          "1{s >= lambda_star and wrong}; lambda_star = trial "
                          "mean of oracle Prop 2 thresholds on eval points; "
                          "weights normalized to mean 1 on the calibration sample"),
    "misspec_recompute": "fa8459eb3cb50722",
    "recompute_tolerance": 1e-9,
    "collapse_criteria": {
        "spearman_min": 0.9,
        "residual_tol": "max(3*SE_cell, 0.0075)",
        "residual_pass_fraction": 0.9,
        "fit": "unweighted isotonic regression on cell means, increasing",
    },
    "family_seed_offset": 10,
    "z_one_sided": 1.645,
    "delta_control": 0.005,
    "delta_oracle": 0.005,
    "seed": 20260818,
}


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


def aligned_error(cal, w_true: np.ndarray, w_hat: np.ndarray,
                  lam_star: float) -> float:
    wn = w_true / w_true.mean()
    hn = w_hat / w_hat.mean()
    L = ((cal.s >= lam_star) & cal.wrong).astype(float)
    return float(np.mean((wn - hn) * L))


# ---------------------------------------------------------------------------
# Part 1: synthetic-estimator cells.
# ---------------------------------------------------------------------------

def run_synth_cell(family: str, fi: int, si: int, param: float, scale: float,
                   cfg: dict, n_trials: int):
    d, k = cfg["d"], cfg["tilt_dim"]
    b = np.zeros(d)
    b[k] = scale
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda"])
    rng = np.random.default_rng([cfg["seed"], cfg["family_seed_offset"] + fi,
                                 si, int(scale * 1000)])

    trials = {"oracle": [], "estimated": []}
    battery, aligned, paired_diff = [], [], []

    for _ in range(n_trials):
        cal = synth.draw_cases(rng, cfg["n_cal"], d, mean=None)
        ev = synth.draw_cases(rng, cfg["n_eval"], d, mean=b)

        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        w_cal_true = shift.gaussian_tilt_ratio(cal.X, b)
        w_ev_true = shift.gaussian_tilt_ratio(ev.X, b)

        if family == "temper":
            w_cal_hat = w_cal_true ** param
            w_ev_hat = w_ev_true ** param
        else:
            w_cal_hat = w_cal_true * np.exp(param * cal.X[:, k])
            w_ev_hat = w_ev_true * np.exp(param * ev.X[:, k])

        lam_o = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                               w_cal_true, w_ev_true)
        lam_star = float(np.mean(lam_o))
        res_o = metrics.evaluate(ev.s, ev.wrong, ev.region, lam_o, cfg["alpha"])
        res_o["ess"] = crc.effective_sample_size(w_cal_true)
        trials["oracle"].append(res_o)

        lam_e = crc.lhat_prop2(losses, lambdas, cfg["alpha"],
                               w_cal_hat, w_ev_hat)
        res_e = metrics.evaluate(ev.s, ev.wrong, ev.region, lam_e, cfg["alpha"])
        res_e["ess"] = crc.effective_sample_size(w_cal_hat)
        trials["estimated"].append(res_e)

        battery.append(shift.ratio_error_battery(w_cal_hat, w_cal_true))
        aligned.append(aligned_error(cal, w_cal_true, w_cal_hat, lam_star))
        paired_diff.append(res_e["marginal_risk"] - res_o["marginal_risk"])

    return _summarise_cell(trials, battery, aligned, paired_diff, cfg)


def _summarise_cell(trials, battery, aligned, paired_diff, cfg):
    a = np.asarray(aligned)
    diff = np.asarray(paired_diff)
    dm = float(diff.mean())
    dse = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    z = cfg["z_one_sided"]
    ci = (dm - z * dse, dm + z * dse)

    rows = []
    for arm, tr in trials.items():
        summ = metrics.summarise(tr)
        summ["verdict"] = verdict(summ["marginal_risk_mean"],
                                  summ["marginal_risk_se"], cfg)
        if arm == "estimated":
            for key in battery[0]:
                summ[f"ratio_{key}_mean"] = float(np.mean([x[key] for x in battery]))
            summ["aligned_mean"] = float(a.mean())
            summ["aligned_se"] = float(a.std(ddof=1) / np.sqrt(len(a)))
            summ["paired_diff_mean"] = dm
            summ["paired_diff_se"] = dse
            summ["paired_diff_ci90"] = [ci[0], ci[1]]
            summ["oracle_equiv"] = bool(
                -cfg["delta_oracle"] <= ci[0] and ci[1] <= cfg["delta_oracle"])
            summ["envelope_ok"] = bool(
                summ["excess_marginal_risk_mean"] - z * summ["marginal_risk_se"]
                <= summ["ratio_w_l1_mean"])
            summ["aligned_le_l1"] = bool(
                abs(summ["aligned_mean"]) <= summ["ratio_w_l1_mean"] + 1e-12)
        rows.append((arm, summ))
    return rows


# ---------------------------------------------------------------------------
# Part 2: misspecification-cell recompute (analysis tier).
#
# Replays each cell of fa8459eb3cb50722 with IDENTICAL rng streams and draw
# order, recording aligned error, and asserts the estimated-arm mean risk
# reproduces the stored value exactly. Duplicated loop rather than a
# modified misspec runner so the archived experiment file stays untouched;
# the assertion catches any replication drift.
# ---------------------------------------------------------------------------

def recompute_misspec_cell(axis, ai, si, setting, scale, mcfg):
    d = setting.get("d", mcfg["base_d"])
    n_fit = setting.get("n_fit", mcfg["default_n_fit"])
    clip = tuple(setting.get("clip", mcfg["default_clip"]))
    k = mcfg["tilt_dim"]
    lambdas = np.linspace(0.0, 1.0, mcfg["n_lambda"])
    rng = np.random.default_rng([mcfg["seed"], ai, si, int(scale * 1000)])
    view = misspec.make_view(axis, setting, mcfg, rng)

    if axis == "mismatch":
        w_true_fn = lambda X: shift.tanh_tilt_ratio(X, scale, k)      # noqa: E731
        draw_target_X = lambda n: shift.rejection_tilt_draw(          # noqa: E731
            rng, n, d, scale, k)
    else:
        b = np.zeros(d)
        b[k] = scale
        w_true_fn = lambda X: shift.gaussian_tilt_ratio(X, b)         # noqa: E731
        draw_target_X = lambda n: rng.standard_normal((n, d)) + b     # noqa: E731

    risks, aligned = [], []
    for _ in range(mcfg["n_trials"]):
        cal = synth.draw_cases(rng, mcfg["n_cal"], d, mean=None)
        ev = synth.realize(draw_target_X(mcfg["n_eval"]), rng)
        Xs_fit = rng.standard_normal((n_fit, d))
        Xt_fit = draw_target_X(n_fit)

        losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
        w_cal_true = w_true_fn(cal.X)
        w_ev_true = w_true_fn(ev.X)

        # Arm order and view-call order must mirror the original runner:
        # oracle consumes no rng; estimated views Xs_fit, Xt_fit, cal.X, ev.X.
        lam_o = crc.lhat_prop2(losses, lambdas, mcfg["alpha"],
                               w_cal_true, w_ev_true)
        lam_star = float(np.mean(lam_o))

        w_fn = shift.fit_ratio(view(Xs_fit), view(Xt_fit),
                               method=mcfg["weight_estimator"], clip=clip)
        w_cal_hat, w_ev_hat = w_fn(view(cal.X)), w_fn(view(ev.X))
        lam_e = crc.lhat_prop2(losses, lambdas, mcfg["alpha"],
                               w_cal_hat, w_ev_hat)

        committed = ev.s >= np.asarray(lam_e, dtype=float)
        risks.append(float((committed & ev.wrong).sum() / len(ev.s)))
        aligned.append(aligned_error(cal, w_cal_true, w_cal_hat, lam_star))

    r = np.asarray(risks)
    a = np.asarray(aligned)
    return {
        "risk_mean": float(r.mean()),
        "risk_se": float(r.std(ddof=1) / np.sqrt(len(r))),
        "aligned_mean": float(a.mean()),
        "aligned_se": float(a.std(ddof=1) / np.sqrt(len(a))),
    }


def recompute_misspec(root: pathlib.Path, cfg: dict) -> list[dict]:
    mcfg = misspec.CONFIG
    stored_dir = root / "artifacts" / f"wp1m_{cfg['misspec_recompute']}"
    stored = json.loads((stored_dir / "results.json").read_text())
    assert misspec.config_hash(mcfg) == cfg["misspec_recompute"], \
        "misspec CONFIG no longer matches the artifact being recomputed"

    out = []
    for axis, ai, si, setting, scale in misspec.iter_cells(mcfg):
        label = misspec.setting_label(axis, setting)
        rec = recompute_misspec_cell(axis, ai, si, setting, scale, mcfg)
        ref = [r for r in stored
               if r["arm"] == "estimated" and r["axis"] == axis
               and r["setting"] == label and r["beta_scale"] == scale]
        assert len(ref) == 1, (axis, label, scale)
        ref = ref[0]
        drift = abs(rec["risk_mean"] - ref["marginal_risk_mean"])
        if drift > cfg["recompute_tolerance"]:
            raise SystemExit(
                f"[recompute] REPLICATION DRIFT {drift:.2e} at "
                f"{axis}/{label}/beta={scale}; aborting.")
        out.append({
            "source": "misspec", "axis": axis, "setting": label,
            "beta_scale": scale,
            "excess_mean": ref["excess_marginal_risk_mean"],
            "risk_se": ref["marginal_risk_se"],
            "aligned_mean": rec["aligned_mean"],
            "aligned_se": rec["aligned_se"],
            "l1_mean": ref["ratio_w_l1_mean"],
        })
        print(f"[recompute] {axis:<11} {label:<14} beta={scale:<5} "
              f"risk reproduced (drift {drift:.1e})  "
              f"aligned={rec['aligned_mean']:+.4f}±{rec['aligned_se']:.4f}")
    return out


# ---------------------------------------------------------------------------
# Part 3: pooled collapse analysis.
# ---------------------------------------------------------------------------

def collapse_analysis(cells: list[dict], cfg: dict) -> dict:
    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression

    x = np.array([c["aligned_mean"] for c in cells])
    y = np.array([c["excess_mean"] for c in cells])
    se = np.array([c["risk_se"] for c in cells])

    rho = float(spearmanr(x, y).statistic)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    fit = iso.fit_transform(x, y)
    resid = y - fit
    tol = np.maximum(3.0 * se, 0.0075)
    ok = np.abs(resid) <= tol
    frac = float(ok.mean())
    crit = cfg["collapse_criteria"]
    return {
        "n_cells": len(cells),
        "spearman": rho,
        "residual_pass_fraction": frac,
        "worst_cells": [
            {"cell": f"{c.get('axis', c.get('family'))}/{c['setting']}"
                     f"/beta={c['beta_scale']}",
             "residual": float(r), "tol": float(t)}
            for c, r, t in sorted(zip(cells, resid, tol),
                                  key=lambda z: -abs(z[1]) / z[2])[:5]
        ],
        "collapse": bool(rho >= crit["spearman_min"]
                         and frac >= crit["residual_pass_fraction"]),
        "isotonic_fit": [[float(a), float(b)] for a, b in zip(x, fit)],
    }


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])

    tests.test_prop2_reduces_to_unweighted()
    print("[selftest] Proposition 2 reduces to unweighted at w == 1: PASS")
    check_registration(CONFIG, root / "registrations")

    n_trials = smoke if smoke is not None else CONFIG["n_trials"]
    tag = "PILOT SMOKE (recompute skipped)" if smoke else "pilot (placeholder env)"
    print(f"[wp1ae] {tag}: n_trials={n_trials}")

    all_rows, cells = [], []

    for fi, (family, spec) in enumerate(CONFIG["families"].items()):
        key = "gamma" if family == "temper" else "delta"
        for si, param in enumerate(spec[key]):
            for scale in CONFIG["beta_scales"]:
                rows = run_synth_cell(family, fi, si, param, scale,
                                      CONFIG, n_trials)
                for arm, summ in rows:
                    row = {"source": "synthetic", "family": family,
                           "setting": f"{key}={param}", key: param,
                           "beta_scale": scale, "arm": arm, **summ}
                    all_rows.append(row)
                    if arm == "estimated":
                        cells.append({
                            "source": "synthetic", "family": family,
                            "setting": f"{key}={param}", "beta_scale": scale,
                            "excess_mean": summ["excess_marginal_risk_mean"],
                            "risk_se": summ["marginal_risk_se"],
                            "aligned_mean": summ["aligned_mean"],
                            "aligned_se": summ["aligned_se"],
                            "l1_mean": summ["ratio_w_l1_mean"],
                        })
                        extra = (f" aligned={summ['aligned_mean']:+.4f}"
                                 f" L1={summ['ratio_w_l1_mean']:.3f}"
                                 f" {'equiv' if summ['oracle_equiv'] else 'NOT-equiv'}"
                                 f" {'env-ok' if summ['envelope_ok'] else 'ENV-BREACH'}"
                                 f"{'' if summ['aligned_le_l1'] else ' ALIGNED>L1-BUG'}")
                    else:
                        extra = ""
                    print(f"[wp1ae] {family:<11} {key}={param:<5} beta={scale:<5}"
                          f" {arm:<9} risk={summ['marginal_risk_mean']:.4f}"
                          f"±{summ['marginal_risk_se']:.4f}"
                          f" [{summ['verdict']:<12}]{extra}")

    if smoke is None:
        mis_cells = recompute_misspec(root, CONFIG)
        cells = mis_cells + cells
        analysis = collapse_analysis(cells, CONFIG)
        print(f"[wp1ae] collapse: n={analysis['n_cells']} "
              f"spearman={analysis['spearman']:.3f} "
              f"resid_pass={analysis['residual_pass_fraction']:.2f} "
              f"-> {'COLLAPSE' if analysis['collapse'] else 'NO COLLAPSE'}")
    else:
        analysis = {"skipped": "smoke"}

    suffix = f"_smoke{smoke}" if smoke else ""
    out_dir = root / "artifacts" / f"wp1ae_{config_hash(CONFIG)}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**CONFIG, "n_trials_effective": n_trials, "smoke": bool(smoke)}, indent=2))
    (out_dir / "results.json").write_text(json.dumps(all_rows, indent=2))
    (out_dir / "cells.json").write_text(json.dumps(cells, indent=2))
    (out_dir / "collapse.json").write_text(json.dumps(analysis, indent=2))
    print(f"\n[out] {out_dir}")


if __name__ == "__main__":
    main()
