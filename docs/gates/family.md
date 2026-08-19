# Environments 2-5: validation gate report

All four environments passed all five gates on their first formal run
(2026-08-19), under contracts frozen before the runs and design intents
registered in cross_environment_hypothesis.json before the family module
was written. The decision-conditional G3 form (env claims amendment 2)
was used from the start; the claims iteration paid for this generation's
first-pass success.

| Env | Rules | G3 best lift (feature) | Region share (beta 0 to max) | Score cells | Verdict |
|-----|-------|------------------------|------------------------------|-------------|---------|
| tickets | 63 | +0.0861 (frustration, blind, independent) | 0.154 to 0.151 (flat: tilt independent of region) | 50 | ALL PASS |
| fraud | 31 | +0.0931 (device_novelty, blind, correlated) | 0.054 to 0.268 | 30 | ALL PASS |
| moderation | 64 | +0.0726 (toxicity, visible); sarcasm +0.0263 (blind candidate) | 0.062 to 0.239 | 57 | ALL PASS |
| compliance | 22 | +0.0781 (redline_density, blind, weak correlates) | 0.101 to 0.215 | 21 | ALL PASS |

APE/CFR audits were computed and filed in
cross_environment_hypothesis.json immediately after these gates and
before any sweep. Audit-forecast ordering at the matched top level
(chi2 ~ 3.47): tickets +0.089 > compliance +0.037 > moderation +0.016 >
fraud -0.005, which already disagrees with the registered design-intent
ordering H0 on the fraud/moderation pair; H0 is left as registered and
will be judged against realized sweeps. The audits also forecast
NON-MONOTONE decay curves for the two defended environments (fraud,
moderation): a falsifiable shape prediction carried into the sweep
registrations.
