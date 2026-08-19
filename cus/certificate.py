"""The WP2 observable certificate (Phase 3).

Deployment-visible inputs only: labelled held-out SOURCE data, the
ratio estimate w_hat, the calibration draw that set the thresholds, and
UNLABELLED target covariates. The bound is assembled in the own-rule
coordinate, where for ANY deployed commit function C(x) (here: the
literal Prop-2 rule x -> 1{s(x) >= lam_hat(x)}),

    excess = E_Q[K C] - alpha
           = E_P0[(w - w_hat) K C]  +  ( E_P0[w_hat K C] - alpha ).
             \____________________/     \_______________________/
                bounded by A_hat          measured directly (b_own)

The first term is bounded through the audit model m_hat ~ P(wrong | s,
dec, x) (fitted on held-out source; the conditional p is invariant
under the platform's enforced covariate shift):

    E_P0[(w - w_hat) p C] = { E_Q[m C] - E_P0[w_hat m C] }        (plug-in)
                          + { E_Q[(p-m) C] - E_P0[w_hat (p-m) C] } (localized
                                                                    calibration)

with the second brace bounded by a reliability analysis of m_hat
RESTRICTED to the commit region, binned by m_hat, gaps transferred to
the target through the unlabelled sample (p-invariance). Certificate:

    alpha_cert = alpha + a_plugin + cal_err_loc + b_own_ucb + z * se(a_plugin).

No kappa, no m, no lambda* enters the bound; those are explanatory
(wp2_theory.md sections 3-5). Known attack surfaces, deliberately left
for Phase 4: miscalibration mass hidden by within-bin cancellation or
placed outside the audited commit region; estimator errors invisible on
held-out source but loss-aligned under the target.

REVISION 1 (red-team round 1, R1-3): the certificate carries a
deployment-visible FRESHNESS PRECONDITION. Post-audit drift breached
the bound at +5.9 beta-units on tickets (and marginally at +3.6) while
the stale bound barely moved, so alpha_cert is valid only while
drift_monitor(audited target sample, arriving covariates) stays at or
below the tolerance set by the last surviving round-1 point; beyond
it the certificate must be recomputed against a fresh target sample.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from cus import crc


class Audit:
    """Held-out-source audit model m_hat(s, dec, X) per the B0 ledger:
    logistic on [s, dec, X, dec*X]."""

    def __init__(self, env, rng, n_audit: int = 60_000):
        pool = env.case_table(rng, n_audit)
        dec, _ = env.route(pool.X)
        self.clf = LogisticRegression(max_iter=2000)
        self.clf.fit(self._feats(pool.X, pool.s, dec), pool.wrong)

    @staticmethod
    def _feats(X, s, dec):
        d = dec.astype(float)[:, None]
        return np.hstack([s[:, None], d, X, d * X])

    def predict(self, env, pool):
        dec, _ = env.route(pool.X)
        return self.clf.predict_proba(self._feats(pool.X, pool.s, dec))[:, 1]


def certificate(env, audit, alpha, rng, feat, beta, what_fn,
                n_cal: int = 1000, n_src: int = 20_000, n_tgt: int = 20_000,
                n_lambda: int = 400, z: float = 1.645, n_bins: int = 10):
    """One deployment draw -> (alpha_cert, parts). what_fn maps a case
    table to weights (the deployment's estimator applied to X)."""
    lambdas = np.linspace(0, 1, n_lambda)
    cal = env.case_table(rng, n_cal)
    w_cal = what_fn(cal)
    losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)

    src = env.case_table(rng, n_src)                       # held-out labelled
    tgt = env.case_table(rng, n_tgt, beta=beta, feature=feat)  # unlabelled
    w_src, w_tgt = what_fn(src), what_fn(tgt)

    lam_src = crc.lhat_prop2(losses, lambdas, alpha, w_cal, w_src)
    lam_tgt = crc.lhat_prop2(losses, lambdas, alpha, w_cal, w_tgt)
    C_src = (src.s >= np.asarray(lam_src))
    C_tgt = (tgt.s >= np.asarray(lam_tgt))

    wn_src = w_src / w_src.mean()

    m_src = audit.predict(env, src)
    m_tgt = audit.predict(env, tgt)

    # plug-in aligned term
    t_term = m_tgt * C_tgt
    s_term = wn_src * m_src * C_src
    a_plugin = float(t_term.mean() - s_term.mean())
    se_a = float(np.hypot(t_term.std() / np.sqrt(n_tgt),
                          s_term.std() / np.sqrt(n_src)))

    # localized reliability of m_hat inside the commit region
    edges = np.quantile(m_src[C_src], np.linspace(0, 1, n_bins + 1)) \
        if C_src.sum() >= n_bins * 20 else np.linspace(0, 1, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    gap_ub = np.zeros(n_bins)
    for b in range(n_bins):
        m = C_src & (m_src >= edges[b]) & (m_src < edges[b + 1])
        nb = int(m.sum())
        if nb == 0:
            gap_ub[b] = 0.0
            continue
        gap = abs(float(src.wrong[m].mean()) - float(m_src[m].mean()))
        se = float(np.sqrt(max(src.wrong[m].mean()
                               * (1 - src.wrong[m].mean()), 0.25 / nb) / nb))
        gap_ub[b] = gap + z * se
    # transfer through both weightings
    def mass(vals, Cmask, wts):
        out = np.zeros(n_bins)
        for b in range(n_bins):
            m = Cmask & (vals >= edges[b]) & (vals < edges[b + 1])
            out[b] = float(wts[m].sum()) / len(vals)
        return out
    q_tgt = mass(m_tgt, C_tgt, np.ones(n_tgt))
    q_src = mass(m_src, C_src, wn_src)
    cal_err = float((q_tgt + q_src) @ gap_ub)

    # own crossing margin on held-out source
    bterm = wn_src * src.wrong * C_src
    b_own = float(bterm.mean()) - alpha
    b_own_ucb = b_own + z * float(bterm.std() / np.sqrt(n_src))

    bound = a_plugin + cal_err + b_own_ucb + z * se_a
    return {
        "alpha_cert": alpha + bound,
        "excess_bound": bound,
        "a_plugin": a_plugin,
        "cal_err_loc": cal_err,
        "b_own_ucb": b_own_ucb,
        "se_a": se_a,
    }


def drift_monitor(X_audited, X_arriving,
                  clip: tuple[float, float] = (0.01, 0.99)):
    """Deployment-visible freshness check (Revision 1): the estimated
    chi-square divergence of the arriving covariate mix from the audited
    target sample, by the same classifier-odds machinery the deployment
    already runs. Both inputs are covariates only; no labels.

    Returns chi2_hat = E_audited[r_n^2] - 1 with r_n the mean-one
    normalized odds ratio dQ_arriving/dQ_audited.
    """
    X = np.vstack([X_audited, X_arriving])
    y = np.concatenate([np.zeros(len(X_audited)), np.ones(len(X_arriving))])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    p = np.clip(clf.predict_proba(X_audited)[:, 1], clip[0], clip[1])
    r = p / (1.0 - p)
    rn = r / r.mean()
    return float((rn ** 2).mean() - 1.0)
