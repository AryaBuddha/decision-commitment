"""Placeholder environment, v2. Replaces synth.py.

WHY V1 WAS WRONG, AND WHY IT MATTERS BEYOND THE CODE.

In v1 the evidence score was a noisy read of the same difficulty variable
that drove correctness. Under shift, harder cases arrived AND their evidence
scores fell, so the commitment rule declined to commit on them all by itself.
Unweighted CRC therefore barely broke: at the largest shift it was
overcovering, not undercovering. A self-correcting environment cannot exhibit
the failure WP1 exists to measure.

The fix is the substantive modelling assumption of the whole project. In the
setting the proposal describes, a rule's evidence is a property of THE RULE
(how often witnessed, how consistently, under what conditions), not of the
case in front of it. So there are always case features that move correctness
and that the evidence score cannot see. Shift along those features is the
dangerous kind, because the system's own confidence signal does not register
that anything has changed. That is precisely why a distribution-level
guarantee is needed and why per-case confidence is not a substitute.

STRUCTURE.

    X[:, 0]   VISIBLE  driver. Affects correctness; evidence score sees it.
    X[:, 2]   HIDDEN   driver. Affects correctness; evidence score is blind.
    region    minority group defined on the hidden driver.

Tilt on dim 2 and unweighted CRC should fail. Tilt on dim 0 and it should
largely self-correct. Running both is a cheap and informative WP1 ablation:
it shows the guarantee's decay is a function of WHERE the shift falls
relative to the evidence signal, not of shift magnitude alone. No natural
dataset pair lets you separate those two.

Covariate shift is still respected: P(wrong | X, s) is a fixed function and
the tilt only moves the marginal of X.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pool:
    X: np.ndarray        # (N, d) float
    s: np.ndarray        # (N,)   float, evidence score in (0, 1)
    wrong: np.ndarray    # (N,)   bool
    region: np.ndarray   # (N,)   int

    def __len__(self) -> int:
        return len(self.s)

    def take(self, idx: np.ndarray) -> "Pool":
        return Pool(self.X[idx], self.s[idx], self.wrong[idx], self.region[idx])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


VISIBLE_DIM = 0
HIDDEN_DIM = 2


def realize(X: np.ndarray,
            rng: np.random.Generator,
            w_visible: float = 0.85,
            w_hidden: float = 1.15,
            evidence_noise: float = 0.50,
            minority_cut: float = 1.2816,   # top ~10% of a standard normal
            minority_penalty: float = 1.40,
            intercept: float = -1.35) -> Pool:
    """Apply the FIXED conditional (s, wrong, region | X) to any covariates.

    Exact covariate shift by construction: source and target draws differ
    only in where X comes from; this function is the shared conditional.
    """
    n = len(X)
    visible = X[:, VISIBLE_DIM]
    hidden = X[:, HIDDEN_DIM]
    difficulty = w_visible * visible + w_hidden * hidden

    # Evidence sees ONLY the visible driver, plus noise.
    s = _sigmoid(-1.20 * visible + evidence_noise * rng.standard_normal(n))

    region = (hidden > minority_cut).astype(int)

    logit_wrong = (intercept
                   + 1.30 * difficulty
                   - 2.20 * (s - 0.5)
                   + minority_penalty * region)
    wrong = rng.random(n) < _sigmoid(logit_wrong)

    return Pool(X=X, s=s, wrong=wrong, region=region)


def draw_cases(rng: np.random.Generator,
               n: int,
               d: int = 5,
               mean: np.ndarray | None = None,
               **params) -> Pool:
    """Fresh i.i.d. cases. mean=None gives source P0 = N(0, I); mean=b gives
    the EXACT tilted target Q_b = N(b, I) (conjugacy of the exponential
    tilt). No pool, no resampling, no approximation."""
    X = rng.standard_normal((n, d))
    if mean is not None:
        X = X + np.asarray(mean, dtype=float)
    return realize(X, rng, **params)


def make_pool(rng: np.random.Generator, n: int = 40_000, d: int = 5,
              **params) -> Pool:
    """Backward-compatible source pool; prefer draw_cases."""
    return draw_cases(rng, n, d, mean=None, **params)
