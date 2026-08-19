"""Controlled covariate shift with a likelihood ratio known by construction.

THE DESIGN. The AST redraw in the environment family gives you a POOL of
instances. It does not by itself give you a scalar shift magnitude, and it
does not hand you the true likelihood ratio in closed form. So separate the
two jobs:

    redraw  ->  supplies the covariate space and the labels
    tilt    ->  supplies the shift, with w(x) known exactly

Concretely: draw the calibration set uniformly from the pool, and draw the
evaluation set from the same pool with probability proportional to

    w(x) = exp(x . beta)

Because both sets come from one pool, w(x) IS the likelihood ratio
dP_eval/dP_cal, exactly, up to a normalising constant that the weighted
procedures do not need. Sweeping ||beta|| sweeps shift magnitude.

This is the design used in Tibshirani et al. (2019) Section 2 on the airfoil
data. The difference in your setting is that you are not stuck with one
found dataset: the pool is generated, so you can make it as large as you
like and the tilt is the only source of non-exchangeability.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def tilt_weights(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """True likelihood ratio w(x) = exp(x . beta), unnormalised."""
    z = X @ np.asarray(beta, dtype=float)
    z = z - z.max()          # numerical stability; a constant factor is free
    return np.exp(z)


def draw_shifted_indices(rng: np.random.Generator,
                         X: np.ndarray,
                         beta: np.ndarray,
                         m: int) -> np.ndarray:
    """Sample m pool indices with probability proportional to w(x).

    Sampling WITH replacement, as in Tibshirani et al.'s airfoil setup. With
    strong tilts this means the evaluation set contains duplicates, which is
    realistic (a deployment really does see the same case shape repeatedly)
    but does inflate apparent sample size. Log the number of distinct
    indices drawn if you want to be careful about it.
    """
    w = tilt_weights(X, beta)
    p = w / w.sum()
    return rng.choice(len(X), size=m, replace=True, p=p)


def estimate_weights(X_cal: np.ndarray,
                     X_eval: np.ndarray,
                     method: str = "logistic",
                     clip: tuple[float, float] = (0.01, 0.99),
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Probabilistic-classification estimate of the likelihood ratio.

    Fit a classifier to separate calibration (class 0) from evaluation
    (class 1) covariates, then take the odds ratio

        w_hat(x) = p_hat(x) / (1 - p_hat(x))

    which equals the true ratio up to the class-prior constant. This is the
    estimator in Tibshirani et al. (2019) Eq. (12).

    The CLIP is not cosmetic. Tibshirani et al. report that without it their
    random-forest probabilities hit exactly 1 in about 2% of repetitions,
    producing infinite weights. The clip bound is therefore a tuning knob
    that silently caps how much shift the estimator can express, and WP2
    should treat it as a parameter to sweep rather than a constant to set
    once and forget.

    Returns (w_hat_cal, w_hat_eval).
    """
    X = np.vstack([X_cal, X_eval])
    y = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_eval))])

    if method == "logistic":
        clf = LogisticRegression(max_iter=2000)
    elif method == "rf":
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                     random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"unknown method {method!r}")

    clf.fit(X, y)
    p = np.clip(clf.predict_proba(X)[:, 1], clip[0], clip[1])
    w = p / (1.0 - p)
    return w[:len(X_cal)], w[len(X_cal):]


def shift_diagnostics(w_true: np.ndarray) -> dict:
    """Distribution-level summaries of how far the tilt has actually moved.

    ||beta|| is the knob you turn, but it is parameterisation-specific and
    means nothing to a reader comparing your curves to someone else's. Report
    a knob-free measure on the x-axis as well. chi2 = E_P[w^2] - 1 with w
    normalised to mean 1 is the standard one, and it upper-bounds how badly
    importance weighting can behave.
    """
    w = np.asarray(w_true, dtype=float)
    wn = w / w.mean()
    return {
        "chi2_divergence": float((wn ** 2).mean() - 1.0),
        "w_max_over_mean": float(wn.max()),
        "ess_fraction": float(wn.sum() ** 2 / ((wn ** 2).sum() * len(wn))),
    }


def ratio_error(w_hat: np.ndarray, w_true: np.ndarray) -> dict:
    """Estimator quality, on the scale WP2 needs.

    Both vectors are normalised to mean 1 first, because likelihood ratios
    are only identified up to a constant and an unnormalised comparison would
    report error that the procedure never sees.
    """
    a = np.asarray(w_hat, dtype=float)
    b = np.asarray(w_true, dtype=float)
    a = a / a.mean()
    b = b / b.mean()
    return {
        "log_ratio_rmse": float(np.sqrt(np.mean((np.log(a) - np.log(b)) ** 2))),
        "max_log_ratio_err": float(np.max(np.abs(np.log(a) - np.log(b)))),
        "mean_abs_rel_err": float(np.mean(np.abs(a - b) / b)),
    }


# ---------------------------------------------------------------------------
# Exact tilting for the Gaussian placeholder.
#
# The review is right that finite-pool resampling gives an empirical
# approximation to the tilted population, not exact draws from it, so the
# "exact by construction" claim needs an exact mechanism. Rejection sampling
# is NOT that mechanism here: exp(b.x) is unbounded on Gaussian support, so
# no accept bound exists. The exact route is conjugacy. Tilting N(0, I) by
# exp(b.x) IS N(b, I), so drawing evaluation covariates from N(b, I) samples
# the tilted population exactly, with normalised ratio, chi2, and TV all in
# closed form. For real environments with bounded covariates, rejection
# sampling from fresh generator draws is the exact mechanism; pool
# resampling remains a labelled approximation with preregistered
# convergence diagnostics.
# ---------------------------------------------------------------------------

def gaussian_tilt_ratio(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact normalised ratio dQ_b/dP0 for P0 = N(0,I), Q_b = N(b,I).

    w(x) = exp(b.x - |b|^2 / 2), with E_P0[w] = 1 exactly.
    """
    b = np.asarray(b, dtype=float)
    return np.exp(X @ b - 0.5 * float(b @ b))


def gaussian_tilt_chi2(b: np.ndarray) -> float:
    """chi2(Q_b || P0) = exp(|b|^2) - 1, closed form."""
    b = np.asarray(b, dtype=float)
    return float(np.expm1(b @ b))


def gaussian_tilt_tv(b: np.ndarray) -> float:
    """TV(P0, Q_b) = 2 Phi(|b|/2) - 1, closed form."""
    from scipy.stats import norm
    b = np.asarray(b, dtype=float)
    return float(2.0 * norm.cdf(np.linalg.norm(b) / 2.0) - 1.0)


def fit_ratio(X_src: np.ndarray,
              X_tgt: np.ndarray,
              method: str = "logistic",
              clip: tuple[float, float] = (0.01, 0.99),
              seed: int = 0):
    """Fit the classifier-odds ratio estimator on DEDICATED covariate splits
    and return a callable w_hat(X).

    Replaces estimate_weights, which fitted on the same calibration and
    evaluation covariates it was later applied to. That double use lets
    estimator overfitting masquerade as robustness; the four-way split
    (induction / labelled calibration / ratio-fit covariates / fresh
    evaluation) keeps the estimator blind to both the calibration and the
    evaluation draws.
    """
    X = np.vstack([X_src, X_tgt])
    y = np.concatenate([np.zeros(len(X_src)), np.ones(len(X_tgt))])
    if method == "logistic":
        clf = LogisticRegression(max_iter=2000)
    elif method == "rf":
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                     random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"unknown method {method!r}")
    clf.fit(X, y)

    def w_hat(Xq: np.ndarray) -> np.ndarray:
        p = np.clip(clf.predict_proba(Xq)[:, 1], clip[0], clip[1])
        return p / (1.0 - p)

    return w_hat


def ratio_error_battery(w_hat: np.ndarray, w_true: np.ndarray) -> dict:
    """Estimator error in guarantee-relevant norms, both normalised to mean 1.

    L1(P0) is primary: for losses in [0, B],
        | E_P0[w_hat L] - E_P0[w L] |  <=  B * E_P0[ |w_hat - w| ],
    so mean absolute normalised-weight error is the quantity from which a
    WP2-style excess-risk bound can actually be assembled. Log-ratio RMSE is
    kept as a diagnostic, not a target.
    """
    a = np.asarray(w_hat, dtype=float); a = a / a.mean()
    b = np.asarray(w_true, dtype=float); b = b / b.mean()
    return {
        "w_l1": float(np.mean(np.abs(a - b))),
        "w_l2": float(np.sqrt(np.mean((a - b) ** 2))),
        "log_ratio_rmse": float(np.sqrt(np.mean((np.log(a) - np.log(b)) ** 2))),
        "w_hat_max": float(a.max()),
    }
