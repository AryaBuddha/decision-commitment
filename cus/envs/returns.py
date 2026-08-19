"""Environment 7 for WP2 Phase 5: returns-abuse triage. THE HOLDOUT.

Designed 2026-08-19, AFTER the certificate freeze
(registrations/wp2_certificate.json), per the WP2 handover: the
prospective test environment must not have existed when the certificate
was built, and spike (environment 6) is spent. Structure deliberately
distinct from all six predecessors: TWO gold-heavy drift candidates,
one visible (desc_vagueness, tree-splittable), one blind with a WEAK
logged proxy (serial_rate; flag_hist reads it at coefficient 0.35),
plus a gold interaction between them that the certificate's audit
model class cannot represent. Reconstruction strength sits between
tickets (none) and fraud (strong) by design.

Contract identical in form to the family (one routed decision per
generated instance, induced tree rules on the induction view,
Laplace-consistency evidence scores, gold mechanically recomputable,
rung-2 rejection tilts on bounded manifest features).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from cus.synth2 import Pool

N_FEATURES = 13
BOUNDED = {"photo_quality": 4, "desc_vagueness": 5, "serial_rate": 6,
           "item_condition_claim": 7, "category_risk": 8}
INDUCTION_HIDDEN = [6]
REGION = (8, 0.55)


def generate(rng, n):
    z = rng.choice(3, size=n, p=(0.68, 0.24, 0.08))
    X = np.empty((n, N_FEATURES))
    by = lambda v: np.asarray(v)[z]                                # noqa: E731
    X[:, 0] = rng.normal(by([3.4, 4.3, 4.9]), 0.7)     # price_log
    X[:, 1] = rng.poisson(by([0.5, 1.2, 3.0])).astype(float)  # prior_returns
    X[:, 2] = rng.gamma(2.0, by([20.0, 26.0, 8.0]))    # account_age_m
    X[:, 3] = rng.gamma(2.0, by([2.0, 4.0, 6.0]))      # order_freq
    X[:, 4] = rng.beta(by([6.0, 4.0, 2.0]), by([2.5, 3.0, 4.0]))  # photo_quality
    X[:, 5] = rng.beta(by([2.0, 3.0, 4.5]), by([7.0, 5.0, 3.0]))  # desc_vagueness
    X[:, 6] = rng.beta(by([1.6, 2.4, 5.0]), by([9.0, 6.0, 3.0]))  # serial_rate
    X[:, 7] = rng.beta(2.5, 4.0, size=n)               # item_condition_claim
    X[:, 8] = rng.beta(by([2.0, 2.6, 4.0]), by([5.0, 4.0, 3.0]))  # category_risk
    X[:, 9] = (rng.random(n) < 0.26).astype(float)     # weekend
    X[:, 10] = (rng.random(n) < 0.12).astype(float)    # gift_flag
    X[:, 11] = (rng.random(n) < 0.18).astype(float)    # expedited
    X[:, 12] = (rng.random(n) < 0.06 + 0.35 * X[:, 6]).astype(float)  # flag_hist
    return X, rng.standard_normal(n)


def gold(X, u):
    risk = (1.9 * X[:, 5]
            + 1.5 * X[:, 6]
            + 1.2 * X[:, 5] * X[:, 6]
            + 0.45 * np.minimum(X[:, 1], 3.0)
            + 0.55 * np.maximum(X[:, 0] - 4.0, 0.0)
            - 1.1 * X[:, 4]
            - 0.30 * np.tanh(X[:, 2] / 18.0)
            + 0.40 * X[:, 7]
            + 0.80 * u
            - 1.82)
    return (risk > 0.0).astype(bool)


@dataclass
class ReturnsEnv:
    tree: DecisionTreeClassifier = field(default=None)
    leaf_label: dict = field(default_factory=dict)
    leaf_score: dict = field(default_factory=dict)
    env_seed: int = 20260821
    n_demo: int = 6000
    expert_noise: float = 0.07

    spec_name = "returns"

    @classmethod
    def induce(cls, env_seed: int = 20260821, n_demo: int = 6000,
               expert_noise: float = 0.07, max_depth: int = 6,
               min_samples_leaf: int = 50) -> "ReturnsEnv":
        env = cls(env_seed=env_seed, n_demo=n_demo, expert_noise=expert_noise)
        rng = np.random.default_rng([env_seed, 7, 0])
        X, u = generate(rng, n_demo)
        y = gold(X, u)
        flip = rng.random(n_demo) < expert_noise
        y_expert = np.where(flip, ~y, y)
        view = env.view(X)
        tree = DecisionTreeClassifier(max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=0)
        tree.fit(view, y_expert)
        env.tree = tree
        leaves = tree.apply(view)
        for leaf in np.unique(leaves):
            m = leaves == leaf
            n_leaf = int(m.sum())
            n_pos = int(y_expert[m].sum())
            label = n_pos * 2 >= n_leaf
            n_maj = n_pos if label else n_leaf - n_pos
            env.leaf_label[int(leaf)] = bool(label)
            env.leaf_score[int(leaf)] = (n_maj + 1.0) / (n_leaf + 2.0)
        return env

    def view(self, X):
        keep = [j for j in range(N_FEATURES) if j not in INDUCTION_HIDDEN]
        return X[:, keep]

    def route(self, X):
        leaves = self.tree.apply(self.view(X))
        dec = np.array([self.leaf_label[int(l)] for l in leaves])
        s = np.array([self.leaf_score[int(l)] for l in leaves])
        return dec, s

    @staticmethod
    def tilt_logweight(X, beta, feature):
        return beta * X[:, BOUNDED[feature]]

    def draw_instances(self, rng, n, beta=0.0, feature="serial_rate"):
        if beta == 0.0:
            return generate(rng, n)
        j = BOUNDED[feature]
        bound = np.exp(max(beta, 0.0))
        Xs, us = [], []
        got = 0
        while got < n:
            batch = max(int((n - got) * 3.0) + 64, 512)
            X, u = generate(rng, batch)
            acc = np.exp(beta * X[:, j]) / bound
            keep = rng.random(batch) < acc
            Xs.append(X[keep]); us.append(u[keep])
            got += int(keep.sum())
        return np.vstack(Xs)[:n], np.concatenate(us)[:n]

    def case_table(self, rng, n, beta=0.0, feature="serial_rate") -> Pool:
        X, u = self.draw_instances(rng, n, beta, feature)
        dec, s = self.route(X)
        wrong = dec != gold(X, u)
        col, cut = REGION
        region = (X[:, col] >= cut).astype(int)
        return Pool(X=X, s=s, wrong=wrong, region=region)
