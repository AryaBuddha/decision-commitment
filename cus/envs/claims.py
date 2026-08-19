"""Environment 1: claims triage. The first real environment (Phase 1).

WHAT MAKES THIS "REAL" WHERE synth2 IS A PLACEHOLDER. synth2 wires
P(wrong | X, s) by hand. Here every case-table column is produced by an
actual pipeline: instances come from a seeded generator with latent
structure, decision rules are INDUCED from noisy expert demonstrations by
a standard learner, the evidence score is a property of the induced rule
(Laplace-shrunk consistency on induction data, frozen formula), the routed
decision is the single rule that fires, and wrong is computed by
mechanically recomputing gold from the stored instance (manifest plus
latent). Nothing about commit-error structure is scripted; it emerges from
what the induction missed.

THE FROZEN CONTRACT (registrations/env_claims.json carries the hash):

  sampling unit   one routed decision per generated instance; instances are
                  i.i.d. generator draws, so units are exchangeable.
  covariates      the 13 numeric manifest features in FEATURES, in order.
  evidence score  s(rule) = (n_majority + 1) / (n_leaf + 2) on the
                  induction sample. A property of the rule that fired,
                  constant within a rule, monotone in support and
                  consistency. Never recomputed after induction.
  regions         region 1 iff provider_flag_rate >= REGION_CUT.
  gold            gold(X, u) below, deterministic in the instance. The
                  latent u is part of the instance record, so gold is
                  mechanically recomputable, but u is NOT a covariate:
                  rules and ratio estimators see the manifest only.
  shift           rung 2: exponential tilts exp(beta * phi(x)) on BOUNDED
                  manifest features (severity, inconsistency,
                  doc_completeness), sampled exactly by rejection from
                  fresh generator draws. The unnormalised ratio is exact by
                  construction; chi2 is estimated by Monte Carlo with a
                  reported standard error (no closed form on this rung).

Induction data are drawn once from a dedicated seed at Environment build
and never touched again (split 1 of the four-way protocol). Calibration,
ratio-fit, and evaluation splits are the caller's responsibility, as in
the placeholder experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from cus.synth2 import Pool


FEATURES = [
    "amount_log",        # 0
    "item_count",        # 1
    "days_to_file",      # 2
    "prior_claims",      # 3
    "provider_tenure_m", # 4
    "provider_flag_rate",# 5  bounded [0, 1]
    "doc_completeness",  # 6  bounded [0, 1]
    "severity",          # 7  bounded [0, 1]
    "inconsistency",     # 8  bounded [0, 1]
    "weekend",           # 9
    "customer_tenure_m", # 10
    "adjuster_load",     # 11 bounded [0, 1]
    "auto_flag",         # 12
]
BOUNDED = {"provider_flag_rate": 5, "doc_completeness": 6, "severity": 7,
           "inconsistency": 8, "adjuster_load": 11}
REGION_CUT = 0.45
SEGMENT_P = (0.70, 0.22, 0.08)      # routine, complex, adversarial


def generate(rng: np.random.Generator, n: int):
    """Fresh i.i.d. instances: manifest X (n, 13) and latent u (n,)."""
    z = rng.choice(3, size=n, p=SEGMENT_P)
    X = np.empty((n, len(FEATURES)))

    def by(vals):
        return np.asarray(vals)[z]

    X[:, 0] = rng.normal(by([3.0, 4.2, 4.8]), 0.7)
    X[:, 1] = rng.poisson(by([2.0, 5.0, 4.0])) + 1.0
    X[:, 2] = rng.exponential(by([5.0, 12.0, 3.0]))
    X[:, 3] = rng.poisson(by([0.4, 0.8, 2.2])).astype(float)
    X[:, 4] = rng.gamma(2.0, by([24.0, 30.0, 9.0]))
    X[:, 5] = rng.beta(by([1.0, 1.5, 4.0]), by([12.0, 8.0, 4.0]))
    X[:, 6] = rng.beta(by([8.0, 5.0, 2.5]), by([2.0, 3.0, 4.0]))
    X[:, 7] = rng.beta(by([2.0, 4.0, 5.0]), by([6.0, 3.0, 2.5]))
    X[:, 8] = rng.beta(by([1.5, 2.0, 6.0]), by([10.0, 6.0, 3.0]))
    X[:, 9] = (rng.random(n) < 0.28).astype(float)
    X[:, 10] = rng.gamma(2.0, by([30.0, 22.0, 10.0]))
    X[:, 11] = rng.beta(3.0, 3.0, size=n)
    X[:, 12] = (rng.random(n) < 0.10 + 0.50 * X[:, 8]).astype(float)

    u = rng.standard_normal(n)
    return X, u


def gold(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Executable ground-truth disposition: 1 = INVESTIGATE, 0 = APPROVE.

    Deterministic in the instance (X, u). The smooth interaction terms and
    the hinge are deliberately tree-unfriendly: what the induced rule list
    cannot represent becomes within-rule, evidence-blind risk structure,
    which is the phenomenon the platform studies. u carries case merit not
    recorded in the manifest (irreducible noise GIVEN the manifest, zero
    noise given the instance)."""
    risk = (1.8 * X[:, 7]
            + 1.6 * X[:, 8]
            + 0.9 * X[:, 5]
            + 0.35 * np.minimum(X[:, 3], 4.0)
            + 0.5 * np.maximum(X[:, 0] - 3.5, 0.0)
            + 1.2 * X[:, 7] * X[:, 8]
            - 1.1 * X[:, 6]
            - 0.3 * np.tanh(X[:, 10] / 24.0)
            + 0.95 * u
            - 1.45)
    return (risk > 0.0).astype(bool)


@dataclass
class ClaimsEnv:
    tree: DecisionTreeClassifier
    leaf_label: dict
    leaf_score: dict
    env_seed: int
    n_demo: int
    expert_noise: float

    @classmethod
    def induce(cls, env_seed: int = 20260818, n_demo: int = 6000,
               expert_noise: float = 0.06, max_depth: int = 7,
               min_samples_leaf: int = 40) -> "ClaimsEnv":
        """Build the environment: draw demonstrations once, learn the rule
        list, freeze rule-level evidence scores. Induction data are never
        exposed to callers."""
        rng = np.random.default_rng([env_seed, 0])
        X, u = generate(rng, n_demo)
        y = gold(X, u)
        flip = rng.random(n_demo) < expert_noise
        y_expert = np.where(flip, ~y, y)

        tree = DecisionTreeClassifier(max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=0)
        tree.fit(X, y_expert)

        leaves = tree.apply(X)
        leaf_label, leaf_score = {}, {}
        for leaf in np.unique(leaves):
            m = leaves == leaf
            n_leaf = int(m.sum())
            n_pos = int(y_expert[m].sum())
            label = n_pos * 2 >= n_leaf
            n_maj = n_pos if label else n_leaf - n_pos
            leaf_label[int(leaf)] = bool(label)
            leaf_score[int(leaf)] = (n_maj + 1.0) / (n_leaf + 2.0)
        return cls(tree, leaf_label, leaf_score, env_seed, n_demo, expert_noise)

    # -- routing ----------------------------------------------------------
    def route(self, X: np.ndarray):
        """One routed decision per instance: the leaf that fires."""
        leaves = self.tree.apply(X)
        dec = np.array([self.leaf_label[int(l)] for l in leaves])
        s = np.array([self.leaf_score[int(l)] for l in leaves])
        return dec, s

    # -- rung-2 shift -----------------------------------------------------
    @staticmethod
    def tilt_logweight(X: np.ndarray, beta: float, feature: str) -> np.ndarray:
        """Exact unnormalised log-ratio beta * phi(x) for a bounded feature.
        Proposition 2 is scale-invariant, so the oracle arm needs no
        normaliser; chi2 and error batteries normalise empirically."""
        return beta * X[:, BOUNDED[feature]]

    def draw_instances(self, rng: np.random.Generator, n: int,
                       beta: float = 0.0, feature: str = "severity"):
        """Source draws (beta = 0) or exact rung-2 rejection draws from the
        tilted target exp(beta * phi(x)) * P0."""
        if beta == 0.0:
            return generate(rng, n)
        j = BOUNDED[feature]
        bound = np.exp(max(beta, 0.0))
        Xs, us = [], []
        got = 0
        while got < n:
            batch = max(int((n - got) * 2.5) + 64, 512)
            X, u = generate(rng, batch)
            acc = np.exp(beta * X[:, j]) / bound
            keep = rng.random(batch) < acc
            Xs.append(X[keep]); us.append(u[keep])
            got += int(keep.sum())
        return np.vstack(Xs)[:n], np.concatenate(us)[:n]

    def case_table(self, rng: np.random.Generator, n: int,
                   beta: float = 0.0, feature: str = "severity") -> Pool:
        """The data contract: one row per routed decision, columns
        (X, s, wrong, region). Loader-compatible with the placeholder."""
        X, u = self.draw_instances(rng, n, beta, feature)
        dec, s = self.route(X)
        wrong = dec != gold(X, u)
        region = (X[:, BOUNDED["provider_flag_rate"]] >= REGION_CUT).astype(int)
        return Pool(X=X, s=s, wrong=wrong, region=region)

    def tilt_chi2_mc(self, rng: np.random.Generator, beta: float,
                     feature: str, n: int = 400_000):
        """chi2(Q || P0) with the empirically normalised ratio, by Monte
        Carlo on fresh source draws; returns (estimate, standard error)."""
        X, _ = generate(rng, n)
        w = np.exp(self.tilt_logweight(X, beta, feature))
        wn = w / w.mean()
        v = wn ** 2
        return float(v.mean() - 1.0), float(v.std() / np.sqrt(n))
