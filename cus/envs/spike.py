"""Environment 6 for Block C: adversarial kappa ("spike").

Designed AFTER the Block A A3 verdicts, per NEXT_BLOCK.md, to place
kappa_pred far from the 0.85 to 1.25 range the first five environments
occupy, by concentrating wrong-mass in a narrow score band just above
the alpha = 0.10 operating threshold and making that band's error rate
ride on a BLIND, independent feature. The visible driver v sets both the
gold cut and the tree's splits, so leaves near the cut are impure (a
score band just above the CRC threshold) and leaves far from it are
pure; the blind driver b multiplies the error rate only inside the
near-cut bump, so tilting b pours loss mass into the band and the local
slope of E[w L] at lambda* runs far ahead of its tempered counterparts.

This is the ADVERSARIAL-kappa world, explicitly NOT the WP2 certificate
holdout, which must be a different environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from cus.synth2 import Pool

N_FEATURES = 8
BOUNDED = {"v": 0, "b": 1, "r": 2}
INDUCTION_HIDDEN = [1]
REGION = (2, 0.45)


def generate(rng, n):
    X = np.empty((n, N_FEATURES))
    X[:, 0] = rng.beta(2.5, 2.5, n)          # v: visible driver
    X[:, 1] = rng.beta(2.0, 4.0, n)          # b: blind driver, independent
    X[:, 2] = rng.beta(2.0, 5.0, n)          # r: region feature, independent
    X[:, 3] = rng.standard_normal(n)
    X[:, 4] = rng.poisson(2.0, n).astype(float)
    X[:, 5] = rng.random(n)
    X[:, 6] = (rng.random(n) < 0.3).astype(float)
    X[:, 7] = rng.gamma(2.0, 10.0, n)
    return X, rng.standard_normal(n)


def gold(X, u):
    v, b = X[:, 0], X[:, 1]
    bump = np.exp(-((v - 0.60) / 0.13) ** 2)
    b_gate = 1.0 / (1.0 + np.exp(-14.0 * (b - 0.62)))
    risk = 5.0 * (v - 0.60) + 3.2 * bump * (b_gate - 0.22) + 0.50 * u
    return (risk > 0.0).astype(bool)


@dataclass
class SpikeEnv:
    tree: DecisionTreeClassifier = field(default=None)
    leaf_label: dict = field(default_factory=dict)
    leaf_score: dict = field(default_factory=dict)
    env_seed: int = 20260819
    n_demo: int = 6000
    expert_noise: float = 0.03

    spec_name = "spike"

    @classmethod
    def induce(cls, env_seed: int = 20260819, n_demo: int = 6000,
               expert_noise: float = 0.03, max_depth: int = 7,
               min_samples_leaf: int = 40) -> "SpikeEnv":
        env = cls(env_seed=env_seed, n_demo=n_demo, expert_noise=expert_noise)
        rng = np.random.default_rng([env_seed, 6, 0])
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

    def draw_instances(self, rng, n, beta=0.0, feature="b"):
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

    def case_table(self, rng, n, beta=0.0, feature="b") -> Pool:
        X, u = self.draw_instances(rng, n, beta, feature)
        dec, s = self.route(X)
        wrong = dec != gold(X, u)
        col, cut = REGION
        region = (X[:, col] >= cut).astype(int)
        return Pool(X=X, s=s, wrong=wrong, region=region)
