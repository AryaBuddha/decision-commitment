# Commitment under Distribution Shift: how conformal risk control fails, and what predicts the failure

Working draft, started 2026-08-19. Numbers cite config hashes; every
claim's evidence tier is marked. Placeholder-tier numbers appear only in
the narrative of how the platform was built, never as findings.
TODO slots mark numbers the remaining registered runs must supply.

## Abstract (sketch)

Systems that induce decision rules from observed work must decide, per
case, whether to commit to the rule's output or defer. Conformal risk
control (CRC) certifies the marginal commit-error rate at a level alpha,
under an exchangeability assumption that deployment shift breaks; the
known repair (Angelopoulos et al. 2022, Prop. 2) assumes the likelihood
ratio is known, and its authors leave the estimated-ratio case open. We
study that open case on a controlled-violation platform: generated
environments with rules genuinely induced from noisy demonstrations,
mechanically recomputed gold, and exactly known, dialled shift. Three
results. (1) The published robustness bound explains none of the observed
decay: realized unweighted risk rises smoothly (0.098 to 0.123 across
chi2 0 to 3.5 on our first environment) while the Prop. 3 envelope is
vacuous by two to three orders of magnitude. (2) Excess risk under
estimated-ratio Prop. 2 is one-dimensional in the SIGNED ALIGNED weight
error a = E[(w - w_hat) L(lambda*)]: across 58 cells spanning four
realistic estimator-failure axes and two synthetic families, all 58 fall
on one monotone curve through the origin with slope ~0.86 (Spearman
0.956). Error magnitude (L1) does not predict damage; its signed
projection onto the loss does. (3) The damage a given shift does is
governed by the gold-relevance of the shifted feature net of the rule
system's self-correction, not by shift size: two shifts of identical
chi2 differ threefold in realized excess. We operationalize this as a
deployment-computable audit statistic (APE: source labels plus unlabelled
target covariates only) and preregistered its calibration and ranking
performance across four further environments built after the
registration. TODO: cross-environment verdicts.

## 1. The problem and the open question

[Anchor: Angelopoulos et al. (2022), Section 4.1 prove Proposition 2 for
a KNOWN covariate-shift ratio and close the section by leaving weighted
conformal risk control with an ESTIMATED shift explicitly to future
research. That sentence is the paper's question.]

- Setting: commit/defer per case; evidence score s is a property of the
  induced RULE (support and consistency), not the case; there are always
  case features that move correctness that the evidence cannot see.
- Endpoints, kept structurally separate: marginal error mass (the
  theorem-aligned primary), commitment rate (utility), selective error
  (descriptive only, no CRC theorem).
- Contribution list (each with its hash):
  1. Evidence-tier verification of Prop. 2 itself, on two exact sampling
     mechanisms (ab56864e6cfb3400, 56704982681d6960: oracle consistent in
     7 + 6 + 58 cells; no published empirical evaluation existed).
  2. The vacuity gap of the known robustness bound (ab56864e6cfb3400).
  3. The aligned-error collapse: the coordinate in which the open
     estimated-ratio problem is one-dimensional (56704982681d6960).
  4. Tilt location: gold-relevance net of self-correction, not
     visibility, decides damage at matched chi2 (3879b17b6fa4ae6f).
  5. APE, a preregistered, deployment-computable decay predictor. TODO:
     verdicts from four environments built after the prediction.
  6. The platform and its evidence discipline (registrations with
     enforced config hashes; the amendment and miss trail as data).

## 2. Methodology: a controlled-violation study

- Take a procedure proven correct under assumption A; violate A by a
  known, dialled amount; measure realized against certified performance;
  certify what survives.
- The exactness ladder (rung 1 conjugate analytic tilt; rung 2 rejection
  sampling on bounded tilt features, used for all evidence-tier runs
  here; rung 3 labelled approximation, unused so far). Every figure
  states its rung.
- Environments: seeded generators with latent case merit; rules induced
  by a standard learner from noisy demonstration logs on an INDUCTION
  VIEW (attributes the log never recorded are the evidence-blind
  directions); gold(X, u) deterministic and mechanically recomputable;
  case table contract: one routed decision per generated instance.
- Five validation gates, with the audit trail published including
  failures (docs/gates/): the two claims G3 failures produced the
  decision-conditional audit form (a blind feature's effect on wrongness
  has opposite signs across routed decisions and cancels in unsigned
  audits) and the proxy-leak lesson (a logged correlate of an unlogged
  driver reproduces self-correction).
- Registration discipline: predictions and named falsifiers committed
  before runs, config hash enforced by the runner, verdicts filed
  prediction-by-prediction with misses verbatim, artifacts append-only.

## 3. Environment 1 (claims triage): the guarantee under shift

### 3.1 Shift response (ab56864e6cfb3400, 500 trials/level, Bonferroni)

- Unweighted CRC: strictly monotone decay 0.0975 -> 0.1231 across MC
  chi2 0 -> 3.54; adjusted VIOLATION from the first nonzero level.
- Oracle Prop. 2: consistent at every level (0.0935-0.0975); plateau
  conservatism (46 discrete rule scores) worth ~0.002-0.006 of margin.
- Estimated arm: equivalent to oracle at every level (correctly
  specified logistic; Q1 deliberately unexercised in this run).
- Prop. 3 envelope, MC TV: 62.6 at the first nonzero level, 440 at the
  top, against realized 0.1231. The bound explains nothing of the decay.
- The decay is an order milder than the placeholder at matched chi2
  (+0.023 vs +0.174): realistic induction partially self-corrects
  through rule-visible correlates. Registered miss: the minority-region
  failure did not transfer under the chosen region cut (risk BELOW
  aggregate; region relevance depends on what rules see).

### 3.2 The collapse (56704982681d6960, 58 cells)

- Figure: docs/wp1_claims_collapse.png. Spearman 0.956; residual pass
  58/58; through-origin slope 0.862 (preregistered window 0.8-1.5).
- Realistic failures live in |a| <= 0.02: dropping the never-logged
  feature from the ratio fit outright still leaves a third of the signal
  reconstructable from correlates (registered sub-clause miss, the
  useful kind: correlated logging partially protects deployments).
- First-order theory (wp2_notes): excess ~= kappa * a with kappa the
  ratio of local loss-curve slopes under the two weightings; measured
  slope 0.86-1.19 across environments and tiers.

### 3.3 Tilt location (3879b17b6fa4ae6f)

- At matched chi2, the visible gold-heavy tilt outdamages the blind one
  at every level (z up to 14.9); the visible tilt triggers a real
  deferral response (0.090 -> 0.168) that loses anyway.
- Practitioner sentence: divergence monitors measure how much the mix
  moved, not what it will cost; two identical-chi2 shifts differed
  threefold.

## 4. The cross-environment study (registered before construction)

- Hypothesis file: registrations/cross_environment_hypothesis.json.
  Ladder: H0 design-intent ordering (registered before the environments
  existed) -> APE forecasts (filed after gates, before sweeps) -> realized
  verdicts.
- APE: audit-predicted excess at the frozen threshold, from source
  labels plus unlabelled target covariates only. Forecast ordering at
  matched top chi2: tickets +0.089 > compliance +0.037 > moderation
  +0.016 > fraud -0.005, with non-monotone decay curves forecast for the
  two defended environments. The forecast already contradicts H0 on the
  fraud/moderation pair; both were left standing to be judged.
- Realized (all four sweeps, 500 trials/level): H1 CONFIRMED (max
  |realized excess - APE| 0.0063 across 28 preregistered level-cells);
  H2 CONFIRMED (exact rank agreement, all pairs >> 2 SE); H3 CONFIRMED
  (commit-rate forecasts within 0.02); H0 MISSED on the fraud/moderation
  pair exactly as the pre-sweep audits forecast (reconstruction beats
  fine-split visibility as a defense). Fraud's non-monotone decay ending
  back inside alpha, and moderation's mid-sweep peak, were both called
  before the sweeps ran. Figure: docs/wp1_ape_forecast.png.
- Boundary findings: forecast residuals err toward caution on defended
  environments; the default ratio clip binds at extreme bounded tilts,
  breaking oracle-equivalence (not absolute control) for correctly
  specified estimators in the two blind-driver environments.
- TODO: collapse batteries on environments 2-5; tilt-location where an
  environment has both feature types.

## 5. What WP2 must now supply

- Job 1: excess = kappa * a + sign-asymmetric second-order remainder,
  with the measured boundary (|a| ~ 0.04 on the placeholder; unreached
  by realistic mechanisms on claims).
- Job 2: bound a from observables (wp2_notes section 6): a is the audit
  model's signed miscalibration mass inside the loss region; APE is the
  plug-in version and the cross-environment study is its empirical
  audit. The certificate shape: excess <= kappa_max * CalErr_loc + C2.

## 6. Limitations and threats

- All environments are generated; realism is structural (induced rules,
  recomputed gold, unlogged attributes), not textural. The case for
  transfer rests on mechanism, not provenance.
- Single rule-induction family (trees) so far; evidence scores are
  rule-level by design, which produces plateaued loss curves; kappa's
  behavior on smoother score families is untested.
- chi2 matched at the top level only across environments; APE is tested
  at seven levels per environment but only two environments carry
  detailed misspecification batteries so far.
- The audit model family (logistic with decision interactions) is a
  choice; H1 misses localize exactly where it is too weak.
