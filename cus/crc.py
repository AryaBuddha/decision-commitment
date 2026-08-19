"""Conformal risk control cores for selective commitment.

CONVENTION. Throughout this package, ``lam`` is a COMMITMENT THRESHOLD. The
system commits iff the evidence score ``s_i >= lam``, and defers otherwise.
Raising ``lam`` makes the system commit less often, so the commit-error loss

    L_i(lam) = 1{ s_i >= lam  AND  case i would be answered wrong }

is non-increasing in ``lam`` and bounded by B = 1. That is exactly the
monotonicity condition required by Theorem 1 of Angelopoulos et al. (2022),
so the standard conformal risk control algorithm applies unmodified.

WHAT IS AND IS NOT GUARANTEED HERE.

  * ``lhat_unweighted`` is Angelopoulos et al. (2022) Eq. (4). It is a
    theorem under exchangeability of calibration and evaluation data.

  * ``lhat_weighted`` is the natural analogue obtained by substituting the
    Tibshirani et al. (2019) weighted-exchangeability probabilities for the
    uniform ones. It reduces exactly to the unweighted procedure when all
    weights are equal. IT IS A CONJECTURE, NOT A THEOREM. Tibshirani et al.
    prove a weighted *quantile* result, which covers conformal prediction;
    the extension to risk control is not established. Measuring whether it
    holds empirically, with an oracle ratio and then with an estimated one,
    is the point of WP1.
"""

from __future__ import annotations

import numpy as np


def commit_error_losses(scores: np.ndarray,
                        wrong: np.ndarray,
                        lambdas: np.ndarray) -> np.ndarray:
    """Loss matrix L_i(lam) of shape (n_cases, n_lambdas).

    Parameters
    ----------
    scores : (n,) float. Evidence score per case; higher means more evidence.
    wrong  : (n,) bool.  True if committing on this case would be an error.
    lambdas: (m,) float, ASCENDING. Threshold grid.
    """
    lambdas = np.asarray(lambdas)
    if np.any(np.diff(lambdas) < 0):
        raise ValueError("lambdas must be ascending")
    committed = scores[:, None] >= lambdas[None, :]
    return (committed & wrong[:, None]).astype(float)


def _first_satisfying(bound: np.ndarray,
                      lambdas: np.ndarray,
                      alpha: float) -> float:
    """Smallest lam on the grid whose bound is <= alpha, else lambda_max.

    ``bound`` is non-increasing in lam because the loss is, so the boolean
    mask is monotone and argmax picks the first True.
    """
    ok = bound <= alpha
    if not ok.any():
        return float(lambdas[-1])
    return float(lambdas[int(np.argmax(ok))])


def lhat_unweighted(losses: np.ndarray,
                    lambdas: np.ndarray,
                    alpha: float,
                    B: float = 1.0) -> float:
    """Angelopoulos et al. (2022) Eq. (4).

        lam_hat = inf { lam :  (n/(n+1)) * Rhat_n(lam) + B/(n+1)  <=  alpha }
    """
    n = losses.shape[0]
    rhat = losses.mean(axis=0)
    bound = (n / (n + 1.0)) * rhat + B / (n + 1.0)
    return _first_satisfying(bound, lambdas, alpha)


def lhat_weighted(losses: np.ndarray,
                  lambdas: np.ndarray,
                  alpha: float,
                  w_cal: np.ndarray,
                  w_test: float,
                  B: float = 1.0) -> float:
    """Weighted analogue using Tibshirani et al. (2019) probabilities.

        p_i = w_i / (sum_j w_j + w_test),   p_test = w_test / (sum_j w_j + w_test)
        lam_hat = inf { lam :  sum_i p_i L_i(lam) + p_test * B  <=  alpha }

    ``w_test`` is a single scalar. For a batch evaluation there is a choice
    to make and it is NOT innocuous:

      * mean of the evaluation-set weights  -> average-case, matches how the
        realized risk is averaged, and is what WP1 uses by default;
      * max of the evaluation-set weights   -> conservative;
      * per-test-point, called in a loop     -> closest to the theory, and
        the version WP2 should analyse.

    Record which one you used. Reviewers will ask.
    """
    w_cal = np.asarray(w_cal, dtype=float)
    if np.any(w_cal < 0):
        raise ValueError("weights must be non-negative")
    denom = w_cal.sum() + w_test
    p_cal = w_cal / denom
    p_test = w_test / denom
    rhat = (p_cal[:, None] * losses).sum(axis=0)
    bound = rhat + p_test * B
    return _first_satisfying(bound, lambdas, alpha)


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size, (sum w)^2 / sum(w^2).

    Report this alongside every coverage number. Tibshirani et al. (2019)
    show that the extra dispersion in weighted conformal's coverage is fully
    explained by reduced effective sample size, not by any failure of the
    method. Without this column a WP1 shift-response curve cannot be read.
    """
    w = np.asarray(w, dtype=float)
    denom = float((w ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float(w.sum() ** 2 / denom)


def lhat_prop2(losses: np.ndarray,
               lambdas: np.ndarray,
               alpha: float,
               w_cal: np.ndarray,
               w_test: np.ndarray,
               B: float = 1.0) -> np.ndarray:
    """Angelopoulos et al. (2022), Section 4.1, Proposition 2. LITERAL.

        lam_hat(x) = inf{ lam : (sum_i w(X_i) L_i(lam) + w(x) B)
                                / (sum_i w(X_i) + w(x))  <=  alpha }

    with guarantee E[L_{n+1}(lam_hat(X_{n+1}))] <= alpha under covariate
    shift with the true ratio w. This is a THEOREM, and this function is the
    oracle arm. The earlier lhat_weighted (single global threshold with a
    scalar test weight) is NOT Proposition 2; it is retained below as
    lhat_weighted_global, an ablation measuring what the practical
    global-threshold shortcut loses relative to the literal procedure.

    Returns one lam_hat per test point, shape (m,).
    """
    w_cal = np.asarray(w_cal, dtype=float)
    w_test = np.atleast_1d(np.asarray(w_test, dtype=float))
    if np.any(w_cal < 0) or np.any(w_test < 0):
        raise ValueError("weights must be non-negative")
    S_L = (w_cal[:, None] * losses).sum(axis=0)      # (n_lambda,), non-increasing
    S_w = w_cal.sum()
    rhs = alpha * (S_w + w_test) - w_test * B        # (m,)
    ok = S_L[None, :] <= rhs[:, None]                # (m, n_lambda)
    any_ok = ok.any(axis=1)
    idx = ok.argmax(axis=1)
    return np.where(any_ok, np.asarray(lambdas)[idx], lambdas[-1])


# Renamed, kept as an ablation only. See lhat_prop2 docstring.
lhat_weighted_global = lhat_weighted
