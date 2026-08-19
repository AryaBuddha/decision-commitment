"""Outcome measures for a chosen commitment threshold.

WHAT IS BEING CONTROLLED, AND WHAT IS NOT. The quantity conformal risk
control bounds here is the MARGINAL commit-error rate,

    E[ 1{ commit AND wrong } ]  <=  alpha,

taken over all cases including the deferred ones. That is not the same as the
SELECTIVE risk a practitioner usually wants,

    E[ wrong | commit ],

which is a ratio of two random quantities and is therefore not an expectation
of a per-case monotone loss. Plain CRC does not control it. Angelopoulos's own
selective-classification example reaches for a binomial upper bound, i.e. the
Learn-then-Test machinery, rather than CRC, for exactly this reason.

Both are reported below. Control the marginal one in WP1 because it is the
one with a theorem attached, report the selective one as a descriptive
diagnostic, and treat selective risk under estimated shift as a WP2 extension.
Do not quietly present a marginal guarantee as if it were a selective one.

The deferral rate is the efficiency side of the trade. It plays the same role
that prediction-set size plays in ordinary conformal: a procedure that defers
on everything controls risk perfectly and is worthless.
"""

from __future__ import annotations

import numpy as np


def evaluate(scores: np.ndarray,
             wrong: np.ndarray,
             region: np.ndarray,
             lam,
             alpha: float) -> dict:
    lam = np.asarray(lam, dtype=float)      # scalar or per-test-point array
    committed = scores >= lam
    n = len(scores)
    n_commit = int(committed.sum())

    marginal_risk = float((committed & wrong).sum() / n)
    selective_risk = float((committed & wrong).sum() / n_commit) if n_commit else 0.0

    out = {
        "lambda_hat": float(np.mean(lam)),
        "alpha": float(alpha),
        "marginal_risk": marginal_risk,
        "selective_risk": selective_risk,
        "commit_rate": float(n_commit / n),
        "deferral_rate": float(1.0 - n_commit / n),
        "excess_marginal_risk": float(marginal_risk - alpha),
        "violated": bool(marginal_risk > alpha),
    }

    for r in np.unique(region):
        m = region == r
        nr = int(m.sum())
        if nr == 0:
            continue
        cr = committed & m
        out[f"marginal_risk_region{r}"] = float((cr & wrong).sum() / nr)
        out[f"deferral_rate_region{r}"] = float(1.0 - cr.sum() / nr)
        out[f"share_region{r}"] = float(nr / n)

    return out


def summarise(trials: list[dict]) -> dict:
    """Aggregate across repeated calibration/evaluation splits.

    The headline number is ``violation_rate``: the fraction of trials in which
    realized risk exceeded alpha. A procedure whose MEAN risk sits under alpha
    can still blow the bound half the time, and only the mean would be visible
    if you averaged and stopped.
    """
    keys = set().union(*(t.keys() for t in trials))
    out = {}
    for k in sorted(keys):
        vals = [t[k] for t in trials if k in t]
        if isinstance(vals[0], bool):
            out[k.replace("violated", "violation_rate")] = float(np.mean(vals))
            continue
        arr = np.asarray(vals, dtype=float)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_se"] = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        out[f"{k}_p05"] = float(np.percentile(arr, 5))
        out[f"{k}_p95"] = float(np.percentile(arr, 95))
    out["n_trials"] = len(trials)
    return out
