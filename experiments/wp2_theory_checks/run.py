"""WP2 Phase 2: numerical verification of wp2_theory.md. PILOT TIER.

Deterministic (quadrature) checks; no sampling. A result in
wp2_theory.md is not believed until its check here passes.

  check L2   Lemma 2 (intercept): on quantized factorized curves, the
             crossing margin b lies in [-(one-step drop + charge), 0]
             at every (alpha, K) on a grid.
  check P3   Proposition 3 (first order): on the rank-2 SMOOTH world,
             |excess - kappa a| <= Lip (1 + |kappa|) / c^2 * a^2 with
             Lip and c measured on the crossing interval, across a
             gamma grid at three alphas.
  check L4   Lemma 4 (charge asymmetry): the predicted effective-level
             shift (1 + chi2_arm)(B - alpha)/n reproduces the
             deterministic grid-crossing shift within one grid step,
             across (alpha, n_cal, gamma).

Run: python experiments/wp2_theory_checks/run.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
from scipy import integrate
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus.envs import analytic as A                       # noqa: E402
sys.path.insert(0, str(ROOT / "experiments" / "wp2_quantized_switchboard"))
from run import Gq, Rwq, grid_cross                      # noqa: E402


CONFIG = {
    "experiment": "wp2_theory_checks",
    "tier": "pilot-analytic-deterministic",
    "l2_alphas": [0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    "l2_Ks": [8, 16, 32, 64],
    "p3_alphas": [0.05, 0.10, 0.15],
    "p3_gammas": [0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6],
    "l4_ncals": [250, 1000, 10000],
    "b_tilt": 0.8,
    "n_lambda": 400,
}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def check_registration():
    h = config_hash(CONFIG)
    reg = json.loads((ROOT / "registrations" / "wp2_theory_checks.json").read_text())
    if reg.get("config_hash") != h:
        raise SystemExit(f"Config hash {h} != registered {reg.get('config_hash')}.")
    print(f"[prereg] config {h} matches registration")


# rank-2 smooth world (as in the quantized runner's part B, K = 0)
def _g1(u): return 0.45 * (1 - norm.cdf(1.1 * u))
def _h1(v): return 0.10 + 0.55 * norm.cdf(v)
def _g2(u): return 0.5 * np.exp(-((u - 0.6) / 0.35) ** 2)
def _h2(v): return 0.75 * norm.cdf(3.0 * (v - 0.8))


def R2(lam, c):
    t = max(norm.ppf(max(lam, 1e-300)), -12.0)
    T1, _ = integrate.quad(lambda u: _g1(u) * norm.pdf(u), t, 12, limit=400)
    T2, _ = integrate.quad(lambda u: _g2(u) * norm.pdf(u), t, 12, limit=400)
    H1, _ = integrate.quad(lambda v: _h1(v) * norm.pdf(v, loc=c), c - 12, c + 12, limit=400)
    H2, _ = integrate.quad(lambda v: _h2(v) * norm.pdf(v, loc=c), c - 12, c + 12, limit=400)
    return T1 * H1 + T2 * H2


def dR2(lam, c, eps=1e-5):
    return (R2(lam + eps, c) - R2(lam - eps, c)) / (2 * eps)


def cross2(alpha, c, lo=0.0, hi=1 - 1e-9):
    if R2(lo, c) <= alpha:
        return lo
    for _ in range(80):
        mid = (lo + hi) / 2
        if R2(mid, c) <= alpha:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    check_registration()
    b = CONFIG["b_tilt"]
    lambdas = np.linspace(0, 1, CONFIG["n_lambda"])
    report = {}

    # ---- check L2 ----
    fails = 0
    n = 0
    for K in CONFIG["l2_Ks"]:
        for alpha in CONFIG["l2_alphas"]:
            Lam = grid_cross(alpha, b, K, 10**9, 0.0, lambdas)  # no charge
            i = int(np.searchsorted(lambdas, Lam))
            prev = lambdas[max(i - 1, 0)]
            bmargin = Rwq(Lam, b, K) - alpha
            step = Rwq(prev, b, K) - Rwq(Lam, b, K)
            n += 1
            if not (-step - 1e-12 <= bmargin <= 1e-12):
                fails += 1
    report["L2"] = {"cells": n, "fails": fails}
    print(f"[check L2] intercept bound: {n - fails}/{n} PASS")

    # ---- check P3 ----
    fails, n, worst = 0, 0, 0.0
    for alpha in CONFIG["p3_alphas"]:
        lam_w = cross2(alpha, b)
        if lam_w <= 0.0:
            continue
        for gamma in CONFIG["p3_gammas"]:
            ch = gamma * b
            lam_e = cross2(alpha, ch)
            if lam_e <= 0.0:
                continue
            a = R2(lam_w, b) - R2(lam_w, ch)
            if abs(a) < 1e-9:
                continue
            kap = dR2(lam_w, b) / dR2(lam_w, ch)
            exc = R2(lam_e, b) - alpha
            lhs = abs(exc - kap * a)
            lo, hi = min(lam_w, lam_e), max(lam_w, lam_e)
            grid = np.linspace(lo, hi, 30)
            slopes_w = [dR2(x, b) for x in grid]
            slopes_e = [dR2(x, ch) for x in grid]
            c_min = min(-s for s in slopes_e)
            if c_min <= 0:
                continue
            lip = 0.0
            for s in (slopes_w, slopes_e):
                d = np.abs(np.diff(s)) / np.diff(grid)
                lip = max(lip, float(d.max()))
            rhs = lip * (1 + abs(kap)) / c_min**2 * a * a
            n += 1
            worst = max(worst, lhs / rhs if rhs > 0 else np.inf)
            if lhs > rhs + 1e-12:
                fails += 1
    report["P3"] = {"cells": n, "fails": fails, "worst_ratio": worst}
    print(f"[check P3] first-order remainder bound: {n - fails}/{n} PASS "
          f"(worst lhs/rhs {worst:.3f})")

    # ---- check L4 ----
    fails, n = 0, 0
    K = 0
    step = float(lambdas[1] - lambdas[0])
    for alpha in (0.05, 0.10):
        for ncal in CONFIG["l4_ncals"]:
            for gamma in (0.0, 0.4, 1.6):
                ch = gamma * b
                wt = float(np.exp(ch * b))
                lam_c = grid_cross(alpha, ch, K, ncal, wt, lambdas)
                a_eff = alpha - (wt / (ncal + wt)) * (1.0 - alpha)
                lam_pred = grid_cross(a_eff, ch, K, 10**9, 0.0, lambdas)
                n += 1
                if abs(lam_c - lam_pred) > step + 1e-12:
                    fails += 1
    report["L4"] = {"cells": n, "fails": fails}
    print(f"[check L4] charge-asymmetry formula: {n - fails}/{n} PASS")

    d = ROOT / "artifacts" / f"wp2thk_{config_hash(CONFIG)}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[out] {d}")
    if any(v.get("fails") for v in report.values()):
        raise SystemExit("THEORY CHECK FAILURE: see report")


if __name__ == "__main__":
    main()
