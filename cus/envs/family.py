"""Environments 2-5: the generated family (tickets, fraud, moderation,
compliance), sharing one induction/routing/contract engine.

Each environment is claims-shaped in its contract (see cus/envs/claims.py:
one routed decision per generated instance, rules induced from noisy
demonstration logs on an INDUCTION VIEW, rule-level Laplace-consistency
evidence scores, gold(X, u) deterministic and mechanically recomputable,
rung-2 rejection tilts on bounded manifest features) but differs in the
one dimension the cross-environment hypothesis needs varied: how much of
the drift driver's gold effect the rules can see or defend against.

  tickets      frustration: blind, gold-heavy, INDEPENDENT of every logged
               feature. No reconstruction, no proxy defense.
  fraud        device_novelty: blind, gold-heavy, strongly correlated
               logged features. Reconstruction defends.
  moderation   toxicity: VISIBLE, gold-dominant, finely split by the
               rules. Score-side deferral defense at full strength.
               (sarcasm is the blind gate-3 candidate.)
  compliance   redline_density: blind, moderate gold weight, weak
               correlates, NOISY experts and coarse rules. Weak defense
               and a plateau-heavy loss curve.

Design intents were registered in
registrations/cross_environment_hypothesis.json BEFORE this file existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import zlib

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from cus.synth2 import Pool


# ---------------------------------------------------------------------------
# Generators and gold policies. u is the latent case merit: stored with the
# instance (gold is mechanically recomputable), never a covariate.
# ---------------------------------------------------------------------------

def _seg(rng, n, p):
    return rng.choice(len(p), size=n, p=p)


def gen_tickets(rng, n):
    z = _seg(rng, n, (0.72, 0.20, 0.08))
    X = np.empty((n, 12))
    by = lambda v: np.asarray(v)[z]                                 # noqa: E731
    X[:, 0] = rng.normal(by([4.0, 4.8, 4.4]), 0.6)
    X[:, 1] = rng.poisson(by([0.4, 1.5, 0.9])).astype(float)
    X[:, 2] = rng.gamma(2.0, by([14.0, 20.0, 40.0]))
    X[:, 3] = rng.poisson(by([0.8, 1.6, 2.5])).astype(float)
    X[:, 4] = rng.integers(0, 5, n).astype(float)
    X[:, 5] = rng.beta(by([5.0, 2.5, 4.0]), by([2.5, 4.0, 3.0]))
    X[:, 6] = rng.beta(by([2.0, 5.0, 3.0]), by([6.0, 2.5, 4.0]))
    X[:, 7] = rng.beta(2.2, 4.5, size=n)          # frustration: INDEPENDENT
    X[:, 8] = rng.poisson(by([1.0, 3.0, 2.0])).astype(float)
    X[:, 9] = (rng.random(n) < 0.27).astype(float)
    X[:, 10] = np.where(rng.random(n) < 0.9, z, rng.integers(0, 3, n)).astype(float)
    X[:, 11] = rng.beta(3.0, 3.0, size=n)
    return X, rng.standard_normal(n)


def gold_tickets(X, u):
    risk = (2.2 * X[:, 7] + 1.3 * X[:, 6] + 0.7 * (1.0 - X[:, 5])
            + 0.30 * np.minimum(X[:, 3], 4.0) + 0.72 * u - 2.38)
    return (risk > 0.0).astype(bool)


def gen_fraud(rng, n):
    z = _seg(rng, n, (0.85, 0.10, 0.05))
    X = np.empty((n, 13))
    by = lambda v: np.asarray(v)[z]                                 # noqa: E731
    X[:, 0] = rng.normal(by([3.2, 3.6, 4.4]), 0.8)
    X[:, 1] = rng.beta(by([1.5, 3.0, 5.0]), by([8.0, 5.0, 3.0]))
    X[:, 2] = (rng.random(n) < by([0.05, 0.25, 0.55])).astype(float)
    X[:, 3] = rng.poisson(by([0.3, 2.5, 4.0])).astype(float)
    X[:, 4] = rng.poisson(by([2.0, 8.0, 12.0])).astype(float)
    X[:, 5] = rng.beta(by([1.5, 4.0, 6.0]), by([8.0, 4.0, 2.5]))    # novelty
    X[:, 6] = rng.gamma(2.0, by([30.0, 10.0, 4.0]))
    X[:, 7] = rng.poisson(by([1.2, 3.0, 5.0])).astype(float) + 1.0
    X[:, 8] = (rng.random(n) < by([0.15, 0.35, 0.50])).astype(float)
    X[:, 9] = (rng.random(n) < by([0.05, 0.20, 0.50])).astype(float)
    X[:, 10] = rng.normal(by([3.0, 3.2, 3.8]), 0.5)
    X[:, 11] = rng.poisson(by([0.1, 0.5, 1.5])).astype(float)
    X[:, 12] = rng.beta(by([2.0, 4.0, 5.0]), by([5.0, 3.0, 2.5]))
    return X, rng.standard_normal(n)


def gold_fraud(X, u):
    risk = (2.3 * X[:, 5] + 1.0 * X[:, 1] + 0.6 * X[:, 2]
            + 0.10 * np.minimum(X[:, 4], 15.0) + 0.5 * X[:, 12]
            + 0.85 * u - 2.05)
    return (risk > 0.0).astype(bool)


def gen_moderation(rng, n):
    z = _seg(rng, n, (0.78, 0.16, 0.06))
    X = np.empty((n, 12))
    by = lambda v: np.asarray(v)[z]                                 # noqa: E731
    X[:, 0] = rng.beta(by([1.6, 4.5, 6.0]), by([8.0, 3.0, 2.0]))    # toxicity
    X[:, 1] = rng.poisson(by([0.2, 1.5, 5.0])).astype(float)
    X[:, 2] = rng.gamma(2.0, by([24.0, 12.0, 5.0]))
    X[:, 3] = rng.normal(by([3.0, 3.5, 2.5]), 1.0)
    X[:, 4] = rng.poisson(by([0.1, 0.8, 2.0])).astype(float)
    X[:, 5] = (rng.random(n) < 0.30).astype(float)
    X[:, 6] = rng.beta(3.0, 3.0, size=n)
    X[:, 7] = rng.beta(by([2.0, 3.0, 2.5]), by([6.0, 4.0, 5.0]))    # sarcasm
    X[:, 8] = rng.beta(by([1.5, 2.5, 4.0]), by([6.0, 4.0, 2.5]))    # virality
    X[:, 9] = rng.beta(6.0, 2.0, size=n)
    X[:, 10] = rng.random(n)
    X[:, 11] = rng.poisson(2.0, n).astype(float)
    return X, rng.standard_normal(n)


def gold_moderation(X, u):
    risk = (3.1 * X[:, 0] + 1.0 * X[:, 7] + 0.6 * X[:, 8] * X[:, 0]
            + 0.75 * u - 1.95)
    return (risk > 0.0).astype(bool)


def gen_compliance(rng, n):
    z = _seg(rng, n, (0.70, 0.22, 0.08))
    X = np.empty((n, 11))
    by = lambda v: np.asarray(v)[z]                                 # noqa: E731
    X[:, 0] = rng.normal(by([7.5, 8.3, 8.8]), 0.5)
    X[:, 1] = rng.poisson(by([1.5, 2.5, 4.0])).astype(float) + 2.0
    X[:, 2] = rng.beta(by([1.5, 2.5, 4.0]), by([7.0, 5.0, 3.0]))
    X[:, 3] = rng.poisson(by([0.5, 2.0, 4.0])).astype(float)
    X[:, 4] = rng.exponential(by([20.0, 45.0, 90.0]))
    X[:, 5] = rng.beta(by([1.8, 2.6, 3.2]), by([7.0, 5.5, 4.5]))    # redline
    X[:, 6] = rng.poisson(by([2.0, 3.0, 4.0])).astype(float)
    X[:, 7] = rng.beta(by([7.0, 4.0, 2.0]), by([2.0, 3.0, 5.0]))    # template
    X[:, 8] = rng.beta(2.5, 3.5, size=n)
    X[:, 9] = rng.poisson(by([1.0, 3.0, 6.0])).astype(float)
    X[:, 10] = rng.normal(by([11.0, 12.2, 13.0]), 0.9)
    return X, rng.standard_normal(n)


def gold_compliance(X, u):
    risk = (2.1 * X[:, 5] + 1.0 * (1.0 - X[:, 7]) + 0.8 * X[:, 2]
            + 1.0 * u - 2.08)
    return (risk > 0.0).astype(bool)


# ---------------------------------------------------------------------------
# Specs. Frozen per environment in registrations/env_<name>.json; the
# cross-environment design intents were registered before this file.
# ---------------------------------------------------------------------------

@dataclass
class EnvSpec:
    name: str
    generate: callable
    gold: callable
    n_features: int
    bounded: dict            # feature name -> column
    induction_hidden: list   # columns removed from the demonstration log
    tilt_features: list      # registered bounded tilt candidates (names)
    primary_tilt: str
    region: tuple            # (column, cut): region 1 iff X[:, col] >= cut
    env_seed: int = 20260819
    n_demo: int = 6000
    expert_noise: float = 0.06
    max_depth: int = 7
    min_samples_leaf: int = 40


SPECS = {
    "tickets": EnvSpec(
        name="tickets", generate=gen_tickets, gold=gold_tickets,
        n_features=12,
        bounded={"sentiment": 5, "complexity": 6, "frustration": 7,
                 "automation_score": 11},
        induction_hidden=[7],
        tilt_features=["frustration", "sentiment"],
        primary_tilt="frustration",
        region=(6, 0.62)),
    "fraud": EnvSpec(
        name="fraud", generate=gen_fraud, gold=gold_fraud,
        n_features=13,
        bounded={"merchant_risk": 1, "device_novelty": 5, "txn_entropy": 12},
        induction_hidden=[5],
        tilt_features=["device_novelty", "merchant_risk"],
        primary_tilt="device_novelty",
        region=(1, 0.55)),
    "moderation": EnvSpec(
        name="moderation", generate=gen_moderation, gold=gold_moderation,
        n_features=12,
        bounded={"toxicity": 0, "context_score": 6, "sarcasm": 7,
                 "virality": 8, "language_conf": 9},
        induction_hidden=[7],
        tilt_features=["toxicity", "sarcasm"],
        primary_tilt="toxicity",
        max_depth=8,
        region=(8, 0.60)),
    "compliance": EnvSpec(
        name="compliance", generate=gen_compliance, gold=gold_compliance,
        n_features=11,
        bounded={"jurisdiction_risk": 2, "redline_density": 5,
                 "template_match": 7, "urgency": 8},
        induction_hidden=[5],
        tilt_features=["redline_density", "jurisdiction_risk"],
        primary_tilt="redline_density",
        expert_noise=0.12, max_depth=5, min_samples_leaf=48,
        region=(2, 0.50)),
}


@dataclass
class GenEnv:
    spec: EnvSpec
    tree: DecisionTreeClassifier = field(default=None)
    leaf_label: dict = field(default_factory=dict)
    leaf_score: dict = field(default_factory=dict)

    @classmethod
    def induce(cls, name: str) -> "GenEnv":
        spec = SPECS[name]
        env = cls(spec)
        rng = np.random.default_rng([spec.env_seed,
                                     zlib.crc32(name.encode()), 0])
        X, u = spec.generate(rng, spec.n_demo)
        y = spec.gold(X, u)
        flip = rng.random(spec.n_demo) < spec.expert_noise
        y_expert = np.where(flip, ~y, y)
        view = env.view(X)
        tree = DecisionTreeClassifier(max_depth=spec.max_depth,
                                      min_samples_leaf=spec.min_samples_leaf,
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

    def view(self, X: np.ndarray) -> np.ndarray:
        keep = [j for j in range(self.spec.n_features)
                if j not in self.spec.induction_hidden]
        return X[:, keep]

    def route(self, X: np.ndarray):
        leaves = self.tree.apply(self.view(X))
        dec = np.array([self.leaf_label[int(l)] for l in leaves])
        s = np.array([self.leaf_score[int(l)] for l in leaves])
        return dec, s

    def tilt_logweight(self, X, beta, feature):
        return beta * X[:, self.spec.bounded[feature]]

    def draw_instances(self, rng, n, beta=0.0, feature=None):
        spec = self.spec
        if beta == 0.0:
            return spec.generate(rng, n)
        j = spec.bounded[feature or spec.primary_tilt]
        bound = np.exp(max(beta, 0.0))
        Xs, us = [], []
        got = 0
        while got < n:
            batch = max(int((n - got) * 3.0) + 64, 512)
            X, u = spec.generate(rng, batch)
            acc = np.exp(beta * X[:, j]) / bound
            keep = rng.random(batch) < acc
            Xs.append(X[keep]); us.append(u[keep])
            got += int(keep.sum())
        return np.vstack(Xs)[:n], np.concatenate(us)[:n]

    def case_table(self, rng, n, beta=0.0, feature=None) -> Pool:
        spec = self.spec
        X, u = self.draw_instances(rng, n, beta, feature)
        dec, s = self.route(X)
        wrong = dec != spec.gold(X, u)
        col, cut = spec.region
        region = (X[:, col] >= cut).astype(int)
        return Pool(X=X, s=s, wrong=wrong, region=region)
