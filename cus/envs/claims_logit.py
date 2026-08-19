"""Environment variant for Block B3: claims with a probabilistic scorer.

Same generator, gold, induction view, region, and tilt machinery as
cus/envs/claims.py; ONLY the induction family changes. Rules are replaced
by a logistic scorer fitted to the same noisy demonstration log on the
same induction view. The routed decision is 1{p >= 0.5} and the evidence
score is s = max(p, 1 - p): per-case, near-continuous, a deliberate
departure from the rule-level plateaued scores of the tree family. This
is the probe of whether the programme's findings are tree-family
artifacts (critique defect D3) and the discreteness test's continuous
endpoint (D4).

Contract note, frozen here: the evidence-score clause of the claims
contract ('a property of the rule that fired') is REPLACED for this
variant by 'the scorer's confidence in the routed decision'; everything
else in registrations/env_claims.json carries over. The variant is gated
from scratch; its freeze is registrations/env_claims_logit.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from cus.synth2 import Pool
from cus.envs import claims as base
from cus.envs.claims import BOUNDED, INDUCTION_VIEW, REGION_CUT, generate, gold


@dataclass
class ClaimsLogitEnv:
    clf: LogisticRegression = field(default=None)
    env_seed: int = 20260818
    n_demo: int = 6000
    expert_noise: float = 0.06

    spec_name = "claims_logit"

    @classmethod
    def induce(cls, env_seed: int = 20260818, n_demo: int = 6000,
               expert_noise: float = 0.06) -> "ClaimsLogitEnv":
        env = cls(env_seed=env_seed, n_demo=n_demo, expert_noise=expert_noise)
        rng = np.random.default_rng([env_seed, 1])   # distinct stream from the
        # tree family's [env_seed, 0]: a fresh demonstration log, same process
        X, u = generate(rng, n_demo)
        y = gold(X, u)
        flip = rng.random(n_demo) < expert_noise
        y_expert = np.where(flip, ~y, y)
        env.clf = LogisticRegression(max_iter=2000)
        env.clf.fit(X[:, INDUCTION_VIEW], y_expert)
        return env

    def route(self, X: np.ndarray):
        p = self.clf.predict_proba(X[:, INDUCTION_VIEW])[:, 1]
        dec = p >= 0.5
        s = np.maximum(p, 1.0 - p)
        return dec, s

    tilt_logweight = staticmethod(base.ClaimsEnv.tilt_logweight)

    def draw_instances(self, rng, n, beta=0.0, feature="severity"):
        return base.ClaimsEnv.draw_instances(self, rng, n, beta, feature)

    def case_table(self, rng, n, beta=0.0, feature="severity") -> Pool:
        X, u = self.draw_instances(rng, n, beta, feature)
        dec, s = self.route(X)
        wrong = dec != gold(X, u)
        region = (X[:, BOUNDED["provider_flag_rate"]] >= REGION_CUT).astype(int)
        return Pool(X=X, s=s, wrong=wrong, region=region)
