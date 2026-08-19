"""The analytic world for WP2 Phase 1: rung 1, everything computable.

PILOT TIER BY DEFINITION (a designed conditional, not an induced system);
its job is mechanism identification: the m(alpha) amplification can be
decomposed exactly, term by term, because every population quantity the
CRC procedures interact with has a quadrature-exact expression here.
Confirmation of whatever mechanism survives belongs on the gated
environments, never here.

Design. x = (x0, x1) ~ N(0, I_2) under the source P0.

  score      s(x) = Phi(x0), continuous, Uniform[0,1] marginal under P0,
             so the lambda grid meets no plateaus (rho' exists everywhere;
             the discrete-score variant for the H1 story is a separate
             knob, `quantize`).
  wrongness  P(wrong | x) = g(x0) * h(x1), factorized so every population
             integral splits into two one-dimensional quadratures:
                 g(x0) = g0 * (1 - Phi(ag * x0))      evidence-aligned
                 h(x1) = h0 + h1 * Phi(x1)            evidence-BLIND driver
  shift      exponential tilt on the blind coordinate,
             w(x) = exp(b*x1 - b^2/2), so Q = N((0, b), I) EXACTLY
             (conjugacy; rung 1). chi2 = exp(b^2) - 1 in closed form.
  tempering  w_hat = w^gamma is itself the tilt at gamma*b, so every
             estimated-arm population quantity is quadrature-exact too.

Population quantities (all to quadrature tolerance ~1e-10):
  R0(lam)     = E_P0[L(lam)]                (unweighted source risk)
  Rw(lam, c)  = E_P0[e^{c x1 - c^2/2} L(lam)]  (tilted/weighted risk;
                c = b gives rho(lam) = E_Q[L(lam)], c = gamma*b gives the
                temper arm's estimating curve)
  their lam-derivatives, exact:  dG(t)/dlam = -g(t) with t = Phi^{-1}(lam)
  lam_pop(alpha, c): the population crossing Rw(lam, c) = alpha
  kappa_exact(lam*) = Rw'(lam*, b) / Rw'(lam*, c_hat)
  a_exact(lam*)     = Rw(lam*, b) - Rw(lam*, c_hat)

The sampler draws finite case tables from the same law, so the empirical
CRC machinery (grid inf, B/(n+1) charge, per-covariate Prop 2 thresholds,
calibration noise) can be switched on one term at a time against the
exact population baseline. That switchboard is the Phase 1 instrument.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.stats import norm

from cus.synth2 import Pool


G0, AG = 0.55, 1.1
H0, H1 = 0.15, 0.85


def g_fn(x0):
    return G0 * (1.0 - norm.cdf(AG * x0))


def h_fn(x1):
    return H0 + H1 * norm.cdf(x1)


_CUT = 12.0     # integrands carry a N(c,1) factor with |c| <= ~8; beyond
                # |u| = 12 they are below 1e-30, and finite bounds keep
                # scipy's quad from missing the concentrated mass on
                # semi-infinite intervals.


def _quad(f, lo=-_CUT, hi=_CUT):
    if hi <= lo:
        return 0.0
    v, _ = integrate.quad(f, lo, hi, limit=400, points=[0.0]
                          if lo < 0.0 < hi else None)
    return float(v)


def G_int(lam):
    """E_P0[ 1{Phi(x0) >= lam} g(x0) ], exact tail integral."""
    if lam >= 1.0:
        return 0.0
    t = max(norm.ppf(max(lam, 1e-300)), -_CUT)
    return _quad(lambda u: g_fn(u) * norm.pdf(u), t, _CUT)


def dG_dlam(lam):
    """d/dlam G_int = -g(Phi^{-1}(lam)); the substitution is exact."""
    t = norm.ppf(min(max(lam, 1e-300), 1 - 1e-16))
    return -g_fn(t)


def H_int(c):
    """E_P0[ e^{c x1 - c^2/2} h(x1) ] = E_{N(c,1)}[ h(x1) ], exact."""
    return _quad(lambda u: h_fn(u) * norm.pdf(u, loc=c),
                 c - _CUT, c + _CUT)


def Rw(lam, c):
    """E_P0[ e^{c x1 - c^2/2} L(lam) ] = G_int(lam) * H_int(c)."""
    return G_int(lam) * H_int(c)


def dRw_dlam(lam, c):
    return dG_dlam(lam) * H_int(c)


def lam_pop(alpha, c, lo=0.0, hi=1.0 - 1e-12, tol=1e-12):
    """Population crossing: smallest lam with Rw(lam, c) <= alpha.
    Rw is continuous and strictly decreasing here, so bisection is exact."""
    if Rw(lo, c) <= alpha:
        return lo
    if Rw(hi, c) > alpha:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Rw(mid, c) <= alpha:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def rho(lam, b):
    """Target risk E_Q[L(lam)] under the tilt b."""
    return Rw(lam, b)


def kappa_exact(alpha, b, gamma):
    """Exact local slope ratio at the oracle population threshold."""
    ls = lam_pop(alpha, b)
    return dRw_dlam(ls, b) / dRw_dlam(ls, gamma * b)


def a_exact(alpha, b, gamma):
    """Exact aligned error at the oracle population threshold.
    Weights normalized to mean 1 under P0 (E_P0[e^{cx - c^2/2}] = 1)."""
    ls = lam_pop(alpha, b)
    return Rw(ls, b) - Rw(ls, gamma * b)


def chi2_closed(b):
    return float(np.expm1(b * b))


class AnalyticEnv:
    """Sampler with the case-table contract of the gated environments."""

    spec_name = "analytic"

    def __init__(self, quantize: int | None = None):
        """quantize = K snaps scores to the K-point grid
        {(k + 0.5)/K}, the discrete-score variant for the H1 story.
        None keeps the continuous score."""
        self.quantize = quantize

    @staticmethod
    def tilt_logweight(X, beta, feature="x1"):
        return beta * X[:, 1] - 0.5 * beta * beta

    def draw_instances(self, rng, n, beta=0.0, feature="x1"):
        X = rng.standard_normal((n, 2))
        if beta != 0.0:
            X[:, 1] += beta          # conjugacy: exact draws from Q
        return X, None

    def case_table(self, rng, n, beta=0.0, feature="x1") -> Pool:
        X, _ = self.draw_instances(rng, n, beta, feature)
        s = norm.cdf(X[:, 0])
        if self.quantize:
            K = self.quantize
            s = (np.floor(s * K) + 0.5) / K
        wrong = rng.random(n) < g_fn(X[:, 0]) * h_fn(X[:, 1])
        region = (X[:, 1] >= 0.6).astype(int)
        return Pool(X=X, s=s, wrong=wrong, region=region)


def selftest(n=400_000, seed=7, n_sigma=5.0):
    """Quadrature vs Monte Carlo, the rung-1 exactness gate: R0, rho at a
    tilt, and the temper estimating curve must all agree within n_sigma
    MC standard errors at three thresholds."""
    rng = np.random.default_rng(seed)
    env = AnalyticEnv()
    report = {"ok": True}
    for b, gamma in [(0.0, 1.0), (0.8, 1.0), (0.8, 0.4)]:
        cal = env.case_table(rng, n, beta=0.0)
        w = np.exp(env.tilt_logweight(cal.X, gamma * b))
        for lam in (0.55, 0.75, 0.9):
            L = ((cal.s >= lam) & cal.wrong).astype(float)
            mc = w * L
            m_mc, se = float(mc.mean()), float(mc.std() / np.sqrt(n))
            m_q = Rw(lam, gamma * b)
            key = f"b={b},g={gamma},lam={lam}"
            diff = abs(m_mc - m_q)
            report[key] = {"quad": m_q, "mc": m_mc, "diff": diff,
                           "tol": n_sigma * se}
            if diff > n_sigma * se:
                report["ok"] = False
    return report
