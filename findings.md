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

## F1. Signed aligned weight error, not error magnitude, breaks estimated-shift risk control (fa8459eb3cb50722)

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
cross-axis level comparisons carry an annotation.

## F2. Proposition 2 verified empirically, on both sampling rungs (ef4e524e83fd3cd4, fa8459eb3cb50722)

The literal per-test-covariate Proposition 2 procedure with the true ratio
held at every swept shift level (chi2 up to 8.49, where unweighted CRC
realizes 0.315 against 0.10) and in all 30 misspecification cells,
including the rung-2 cells where the target is drawn by exact rejection
sampling from a bounded tanh tilt. No published empirical evaluation of
Proposition 2 existed; this is the platform's verification of it, pilot
tier. The unit test (exact reduction to unweighted CRC at w == 1) and the
importance-weighted-moment exactness check gate every run.

## F3. The minority-region failure needs no shift (ef4e524e83fd3cd4)

Region 1 (minority, defined on the hidden driver) marginal risk sits near
0.42 at zero shift while the aggregate is exactly controlled at 0.10.
Marginal guarantees do not transfer to subgroups; shift aggravates an
existing failure rather than causing it. This motivates the
group-conditional calibration extension.

## F4. B*L1 is a valid but expensive envelope (fa8459eb3cb50722)

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

## F7. The collapse: excess risk is one-dimensional in signed aligned error (fd2279c8f7dc2df6)

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
