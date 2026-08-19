# Findings

One paragraph per confirmed result, with the config hash that produced it.
Everything below is PILOT TIER: placeholder environment (synth2), calibrating
expectations and pinning definitions, never paper evidence. An entry is
promoted only when a gated real environment reproduces it.

Reproduction status: the registered pilot ef4e524e83fd3cd4 and the full
misspecification sweep fa8459eb3cb50722 have each been re-run from seed on a
second machine (2026-08-18) and reproduce to the fourth decimal, including
the single non-deprivation violation cell, the conservative d=100 cell, and
the clip-ordering inversion.

## F1. Signed aligned weight error, not error magnitude, breaks estimated-shift risk control (fa8459eb3cb50722; PROMOTED to evidence tier by F11; refined by F10, 2026-08-19)

The misspecification sweep degraded the ratio estimator along four axes at
two fixed shift levels. One curve does not map L1(P0) weight error to excess
marginal risk: at matched L1 near 0.4, feature deprivation shows excess risk
of +0.04 to +0.08 while dimension inflation sits at -0.002 to -0.005. The
estimator has two ways to be wrong and they are not symmetric. Blindness
(under-weighting the shifted, risk-relevant, evidence-blind dimension:
deprivation, and ratio fits starved to 50 points at beta 1.25) is
anti-conservative and genuinely breaks control, reaching 0.274 against a
certified 0.10 at full deprivation. Confusion (weight error sprayed across
nuisance directions: dimension inflation to d=100, model-class mismatch)
is conservative; risk stays controlled and the price is paid in deferral
and effective sample size (ESS 157 vs oracle 231 at d=100, beta 1.25).
The quantity WP2 should bound is therefore not a norm of the weight error
but its signed alignment with the loss on the shifted direction, roughly
E[(w - w_hat) L]. Caveat, filed in the registration: the mismatch axis is
not at matched chi2 (0.21/0.52 vs 0.76/3.77 on the linear axes), so
cross-axis level comparisons carry an annotation. REFINEMENT (F10,
evidence tier): "blindness is what is dangerous" was the placeholder's
version of this finding and it does not survive contact with a real
environment unqualified. Gold-relevance net of self-correction decides
the damage; visibility only determines whether a score-side defense is
mounted, and the defense can lose. The signed-aligned-error statement
itself is the part that generalized (F11).

## F2. Proposition 2 verified empirically, on both sampling rungs (ef4e524e83fd3cd4, fa8459eb3cb50722)

The literal per-test-covariate Proposition 2 procedure with the true ratio
held at every swept shift level (chi2 up to 8.49, where unweighted CRC
realizes 0.315 against 0.10) and in all 30 misspecification cells,
including the rung-2 cells where the target is drawn by exact rejection
sampling from a bounded tanh tilt. No published empirical evaluation of
Proposition 2 existed; this is the platform's verification of it, pilot
tier. The unit test (exact reduction to unweighted CRC at w == 1) and the
importance-weighted-moment exactness check gate every run. PROMOTED:
oracle consistent in all 7 wp1c levels, all 6 tilt-location cells, and
all 58 collapse cells on the claims environment.

## F3. A risk-relevant region CAN fail silently at zero shift (ef4e524e83fd3cd4; scope narrowed 2026-08-19)

An existence result, placeholder tier: region 1 (defined on the hidden
driver) marginal risk sat near 0.42 at zero shift while the aggregate was
exactly controlled at 0.10, so marginal guarantees do not automatically
transfer to subgroups and shift aggravates rather than causes the
failure. SCOPE NOTE: this did NOT transfer to the claims environment
under the provider-flag-rate region cut; the first evidence-tier run
(F9, P5 miss) found region-1 risk BELOW aggregate at every level, because
the induced rules already treat that region cautiously. Whether a region
fails silently depends on the region definition's relation to what the
rules can see, which is itself a finding. The group-conditional extension
remains motivated by the existence result, but the claims region hunt is
a registered two-step: exploratory identification on a dedicated design
split, then a fresh registered confirmatory run on new draws. Nothing
from ab56864e6cfb3400 may be reused to pick the cut.

## F4. B*L1 is a valid but expensive envelope (fa8459eb3cb50722; PROMOTED: 58/58 on claims at factor up to 32, see 56704982681d6960)

In all 30 misspecification cells, mean excess risk minus 1.645 SE stays
below mean L1(P0) error, so the bound |E[w_hat L] - E[w L]| <= B E|w_hat - w|
is a valid empirical envelope. It is loose by a factor of 5 or more
everywhere, including the worst (full deprivation) cells, because L1 charges
for conservative error directions that never translate into excess risk.
A certificate priced at B*L1 would be safe and nearly useless, which is the
gap WP2's aligned-error functional should close.

## F5. The global-threshold shortcut loses little, and loses it at large shift (ef4e524e83fd3cd4)

The deployment-realistic single-threshold variant with the true ratio (not
Proposition 2) tracked the literal procedure at every level, running
slightly less conservative at large shift (0.098 vs 0.093 at chi2 8.49).
Worth re-measuring on real environments before anyone ships it.

## F6. The unweighted decay curve is real and the known robustness bound does not explain it (ef4e524e83fd3cd4)

Unweighted CRC is in VIOLATION from the smallest swept shift (0.128 at
chi2 = 0.06) with a smooth monotone decay to 0.315 at chi2 = 8.49, while
the Proposition 3 TV envelope is vacuous (exceeds 1) from the first nonzero
tilt at n_cal = 1000. The observed decay is far milder than the published
bound allows, which is the opening WP2 sharpens.

## F7. The collapse: excess risk is one-dimensional in signed aligned error (fd2279c8f7dc2df6; PROMOTED to evidence tier by F11)

Manipulating aligned error directly with synthetic estimators (tempering
w^gamma from the unweighted endpoint through the oracle into
over-correction, and directional w*exp(delta*x_2) in both signs) and
pooling with the 30 recomputed misspecification cells: excess marginal risk
vs a = E[(w - w_hat) L(lambda*)] collapses onto one monotone curve through
the origin. Spearman 0.970, 56 of 58 cells within the preregistered
residual tolerance, six distinct mechanisms and both shift levels on the
same curve; realistic and synthetic cells at matched aligned error agree to
the third decimal (deprivation rho=0 vs temper gamma=0 at beta 1.25: excess
0.174 vs 0.175). Tempering is strictly monotone through zero at gamma=1 and
the matched-effective-tilt coincidence check passed. The two residual
failures are symmetric and informative: at the far anti-conservative end
(aligned error near 0.053) the two shift levels separate and the curve
steepens, a second-order regime a first-order theory in a will not capture.
WP2's target: bound the signed aligned functional, not a weight-error norm.
The misspec recompute reproduced every archived cell mean with zero drift,
so the aligned-error diagnostics attach to the archived run without
touching it.

## F8. The gates work as designed, and the failures were the valuable part (gates_claims_5459d3b5a7b3c1a1)

Environment 1 (claims triage: generated instances, rules induced from
noisy demonstration logs, rule-level Laplace-consistency evidence, gold
mechanically recomputed) failed gate 3 twice before passing. First
failure: with every manifest feature visible to induction, depth-7 rules
absorbed nearly all covariate dependence of wrongness. Second failure
found two general lessons: a proxy feature (auto_flag reading the blind
feature at coefficient 0.5) let the rule system partially reconstruct the
unlogged attribute, reproducing synth v1's self-correction disease
through a realistic mechanism; and the unsigned blindness audit cancels
sign-opposed effects, because a blind feature raises wrongness in
APPROVE-routed cases and lowers it in INVESTIGATE-routed ones (+0.0049
unsigned vs +0.0627 decision-conditional on identical data). The audit is
now decision-conditional. Full trail in docs/gates/claims.md.

## F9. EVIDENCE TIER: the placeholder story survives contact with a real environment (ab56864e6cfb3400)

First real run, 500 trials per level, Bonferroni-adjusted verdicts.
Unweighted CRC decays strictly monotonically from 0.0975 to 0.1231 across
chi2 0 to 3.54 (adjusted VIOLATION from beta 1); oracle Proposition 2 is
consistent at every level (0.0935 to 0.0975); the well-specified
estimated arm is equivalent to the oracle at every level; ESS stays above
232 of 1000 under the bounded tilt; the Proposition 3 envelope is vacuous
by two to three orders of magnitude. Two env-specific results: plateau
conservatism is real (oracle sits at or below alpha - 0.002 everywhere),
and the decay is an order milder than the placeholder at matched chi2
(+0.023 vs +0.174) because rule-visible correlates partially self-correct;
that attenuation is a finding about real rule-induction systems, not a
platform weakness. One verbatim miss: region-1 (high provider flag rate)
risk sits BELOW aggregate at every level; the induced rules already
handle that region cautiously, so the group-conditional extension needs a
region defined on the blind feature instead, under a fresh registration.

## F10. EVIDENCE TIER: tilt location is governed by gold-relevance net of self-correction, not visibility alone (3879b17b6fa4ae6f)

At matched chi2 on the claims environment, the tilt on the rule-visible,
gold-heavy feature (severity) decays unweighted CRC harder than the tilt
on the rule-blind feature (inconsistency) at every level (+0.0059,
+0.0153, +0.0234; z up to 14.9), inverting the placeholder's ordering,
and the inversion was registered as the prediction before the run from a
disclosed 30-trial probe. The mechanism resolved into two separately
visible effects: the visible tilt triggers a genuine score-side response
(unweighted deferral rises to 0.168 vs 0.090 at zero shift; the
registered prediction that deferral would stay flat missed, verbatim)
but the response loses to the gold-side risk mass the tilt moves; the
blind tilt barely moves deferral at all and breaks CRC through pure
calibration invalidity. chi2 matches covariate-space divergence, not
risk relevance: two shifts of identical chi2 differ threefold in damage
depending on where they fall. The oracle pays for this honestly, deferring
0.306 vs 0.209 at the top level. Practitioner reading: knowing HOW MUCH
the case mix drifted (any divergence monitor gives that) says little
about guarantee damage without knowing WHERE the drift falls relative to
what the rules can see and what drives correctness.

## F11. EVIDENCE TIER: the collapse holds on a real environment, and the paper has its central figure (56704982681d6960)

Fifty-eight cells on the gated claims environment (four realistic
degradation axes and two synthetic families at two shift levels): excess
marginal risk vs signed aligned error collapses onto one monotone curve
with Spearman 0.956, all 58 cells within the preregistered residual
tolerance, and through-origin slope 0.862, inside the registered [0.8,
1.5] window for the kappa ~ 1 first-order identity. The oracle is
consistent in all 58 cells. Two evidence-tier refinements over the
placeholder. First, realistic failures on a real manifest live entirely
in the near-linear regime (|a| <= 0.02): even dropping the blind feature
from the ratio fit outright leaves a logistic classifier able to
partially reconstruct the tilt from correlated, rule-visible features,
cutting aligned error by a third relative to total blindness (the P2
sub-clause miss, filed verbatim). Correlated logging partially protects
deployments; the placeholder's orthogonal-Gaussian geometry was the
worst case. Second, the temper gamma = 0 endpoint reproduces the wp1c
unweighted arm across registrations (0.6 and 0.9 combined SE), tying the
two evidence-tier experiments together. The quantity WP2 must bound is
confirmed on real data: signed aligned error, with the first-order
theory covering the entire realistically reachable range on this
environment.

## F12. EVIDENCE TIER: a deployment-computable number predicts guarantee decay across five environments (cross_environment_hypothesis)

APE, the audit-predicted excess at the frozen threshold, uses only what
deployments have (labelled source data plus unlabelled target
covariates) and was registered, recipe and thresholds included, BEFORE
environments 2-5 existed. Built, gated, audited, then swept, the four
environments realized decay within 0.0063 of the forecast at every one
of 28 preregistered level-cells and in the exact forecast order
(tickets +0.0875 > compliance +0.0378 > moderation +0.0098 > fraud
-0.0103 at matched top chi2), with claims slotting into the same
ordering descriptively. The audits called two qualitative shapes before
the sweeps ran: fraud's decay is non-monotone and ends back inside the
certified level (full defense by correlate reconstruction), and
moderation's peaks mid-sweep. The registered design-intent ordering H0
missed on exactly the pair the audits flagged: reconstruction (fraud)
turned out to be a stronger defense than fine-split visibility
(moderation), and the statistic tracked reality while intuition did not.
Two boundary findings for WP2: forecast residuals err toward caution on
the defended environments, and the default ratio clip binds at extreme
bounded tilts, driving even a correctly specified estimator out of
oracle-equivalence in the two blind-driver environments (tickets,
compliance) while absolute control survived on plateau conservatism.
Figure: docs/wp1_ape_forecast.png.

## F13. EVIDENCE TIER: the collapse holds in five environments and kappa is universal within noise (four wp1mf runs + 56704982681d6960)

Within-environment collapse in all five environments (Spearman 0.903 to
0.961; 289 of 290 evidence-tier cells inside the residual tolerance),
with through-origin slopes 0.74 to 0.96 that no pair separates beyond 2
combined SE. The registered prediction that slopes would differ (kappa
tracking loss-curve geometry) MISSED on its good branch: across
tree-induced environments kappa is empirically a constant near 0.85, and
WP2's first-order theorem can carry one constant instead of an
environment functional. The registered pooled-rank claim also missed
(Spearman 0.843 vs 0.9) for an instructive reason: rank statistics on
cells whose |a| sits inside the trial noise floor are uninformative by
construction; the curve is monotone, the near-origin ranks are noise.
Figure: docs/wp1_all_env_collapse.png.

## F14. EVIDENCE TIER: partial reconstruction lands anywhere on the curve; the coordinate predicts, the narrative does not (wp1mf batteries)

The reconstruction spectrum registered from the claims experience was
half wrong, with a sign inversion as the best datum: on the
self-correcting fraud environment, NO correction is already safe (the
gamma = 0 endpoint has NEGATIVE aligned error, because the loss mass
sits where the true ratio is low), and partially reconstructed weights
FLIP aligned error positive, re-importing risk the score composition
had neutralized (excess +0.0078 vs -0.0027). Compliance's weak-correlate
reconstruction likewise overshot the no-correction endpoint; tickets'
independent driver reproduced no-correction to the fourth decimal;
moderation's reconstruction genuinely defended. Every such cell sits on
its environment's collapse curve. The claims-era sentence 'correlated
logging partially protects you' is requalified: correlated logging
moves you ALONG the curve, and which direction depends on where the
loss mass sits relative to what the correlates rebuild. Only the signed
aligned error predicts the landing point.

## F15. EVIDENCE TIER: the tilt-location crossover completes the F10 law (bf521b1f263f8cc8)

On moderation, at matched chi2, the gold-heavy visible tilt does more
damage at small shift (0.1126 vs 0.1098) and the blind tilt does more at
large shift (0.1318 vs 0.1103, z near 20), because the visible feature's
defense saturates the damage while blindness never defends: sarcasm's
decay is monotone, toxicity's bends back. Combined with claims (defense
mounted and lost) the law now has both branches measured: damage equals
gold-relevance NET of self-correction, and the defense can be silent
(moderation routes correctly, deferral moves at most +0.016) or loud and
failing (claims deferral +0.078). One registered constant was 0.002 too
tight (sarcasm top-cell deferral drift +0.032 vs the 0.03 ceiling),
filed verbatim.

## F16. EVIDENCE TIER: the calibration budget cannot buy away deployment-window noise (a473b915381bcb1e)

The budget table's registered square-root dispersion scaling MISSED:
the 90% spread of realized risk shrinks with calibration budget at a
measured exponent near 0.32, not 0.5, because per-window binomial noise
on the 1000-case evaluation sample floors the dispersion and no
calibration budget removes it. Mean control holds at every budget-shift
cell (oracle consistent 9/9; estimated equivalent even at n_cal = 250);
what n_cal buys at large shift is conservatism margin and about a 2.4x
dispersion reduction against a floor. Deployment reading: past roughly
n_cal = 1000, spend the next label budget on monitoring windows, not on
calibration.

## F13 correction (2026-08-19, Block A). The slope-constancy claim is retracted

F13's statement that kappa is "universal within noise" was
failure-to-reject dressed as equivalence, and the through-origin fits
behind it were misspecified. TOST at the preregistered margin 0.25
establishes equivalence for 0 of 10 environment pairs, and after the
affine correction (F17) the slopes are genuinely unequal (1.005 to
1.734). The within-environment collapse results in F13 stand (the
isotonic yardstick absorbed the intercept); the constancy sentence does
not. Retraction filed as an amendment in
cross_environment_hypothesis.json.

## F17. EVIDENCE TIER: the collapse relation is affine, its intercept is the plateau conservatism, and the corrected functional is the paired difference (blockA_e58d1bc4b8627b0d)

Three archived-data recomputations converge on one correction. The
sign-split slope anatomy (A2) inverted both its registered predictions:
positive branches sit below pooled fits and negative branches are
steeper, the signature of a missing negative intercept, and the fitted
intercepts match each environment's measured oracle conservatism to the
third decimal in five of five environments. The registered per-cell
kappa test (A3) failed on the contaminated y-variable exactly as the
affine model predicts, and the corrected form, paired difference
(estimated minus oracle) regressed through the origin on aligned error,
collapses at R2 0.939 to 0.987 per environment. This resolves critique
defect D4 entirely: the below-1 pooled slopes and the pilot-vs-evidence
slope tension were fit-model artifacts. It also sharpens D3 beyond the
critique: the de-contaminated slopes are genuinely unequal across
environments (fraud 1.005 to compliance 1.734), the local-slope-ratio
kappa_pred explains less of that variation than the environment label
does (R2 0.22 vs 0.46), and the coarsest-score environment has the
steepest slope, naming plateau discreteness as the missing term (Block
B2's target). Separately, the unification test (A1) put 38 of 42
archived unweighted cells on the existing curves, including tilt
directions the batteries never tilted, and localized where the
coordinate degrades: not by tilt direction but by threshold distance,
since a(lambda*) saturates once the evaluated arm's threshold sits far
from the oracle's (compliance sweep: a flat at 0.017 while excess climbs
to 0.038). One coordinate, one affine law, two measured boundaries:
threshold distance and plateau coarseness.
