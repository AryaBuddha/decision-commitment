# NEXT BLOCK: unification, slope anatomy, and knob justification

Response to external critique of 2026-08-19. Scope: EXPERIMENTATION ONLY.
Do not create, extend, or commit any paper prose. paper/draft.md is frozen
as-is; PAPER_DRAFT.md is NOT to be committed. findings.md remains the only
narrative artifact and is updated per verdict as usual.

The critique, restated as testable defects:
(D1) Experimental constants lack justification (alpha, clip, depths, shift
     levels, tolerance and window choices).
(D2) The headline results are presented as three separate findings and
     their claimed common structure has never been tested.
(D3) The slope-constancy claim is a failure-to-reject dressed as
     equivalence, and rests on five environments from one recipe.
(D4) Evidence-tier through-origin slopes (0.74 to 0.96) sit below both the
     pilot near-origin slopes (1.11 to 1.19) and the informal theory
     (kappa near 1), with no reconciliation on file.

Standing rules apply: registrations with predictions and falsifiers before
full runs, config-hash enforcement, verdicts with misses verbatim, RUNLOG
lines, artifacts append-only. Blocks are ordered; A costs almost nothing
and reshapes everything after it, so run it first.

---

## Block A: recomputation from archived artifacts (no new sweeps)

### A1. The unification test (answers D2)

Claim under test: every arm is the estimated arm. The unweighted procedure
is w_hat == 1, so its excess should satisfy the same relation
excess ~ kappa * a with a = E_P0[(w - 1) L(lambda*)]. If true, the decay
curves, the tilt-location cells, and the collapse are one finding.

Do: for every archived evidence-tier unweighted-arm cell (7 shift-response
levels x 5 environments; all tilt-location cells on claims AND moderation,
both tilt features, all matched-chi2 levels), compute a with w_hat == 1
using the archived draws and the oracle-threshold machinery from the
collapse batteries. Plot these cells ON each environment's existing
collapse curve (same residual tolerance, same isotonic fit, fit NOT refit;
the archived curve is the yardstick).

Critical sub-test: the tilt-location cells use tilt features the collapse
batteries never tilted (severity on claims; toxicity on moderation). If
those cells land on the curve, the coordinate unifies across tilt
DIRECTION, which is new and is exactly the connection D2 says is missing.

Register before computing: predicted on-curve fraction (pick a number and
defend it), and the named falsifier: if tilt-location cells sit
systematically off-curve, the collapse is per-tilt-direction and the
aligned-error claim must be narrowed accordingly. Both outcomes are
reportable; only an untested claim is not.

Secondary: APE connects through the same functional. For the four
cross-env sweeps, compare APE per level against kappa_env * a_unw(level)
computed in A1. Descriptive, no threshold; the point is to show (or
refute) that headline 4 is a plug-in of headline 2.

### A2. Slope anatomy: does "below 1" dissolve under a sign split? (answers half of D4)

From the archived 290 collapse cells, fit separate through-origin slopes
for a > 0 and a < 0 per environment, and near-origin local slopes
(|a| <= 0.005, 0.01) per sign. Register the prediction before looking:
positive-branch slopes are closer to 1 than the pooled fits, negative
branches sit near the pilot's conservative-side compression (~0.76), and
the pooled 0.74-0.96 range is substantially an averaging artifact of
fitting one line through a sign-asymmetric curve. Falsifier: if the
positive branch alone still sits well below 1, discreteness attenuation
(Block B2) carries the explanation, not fit pooling.

### A3. Per-cell kappa: from constant to computable (answers D3 and D4 jointly)

kappa has a formula: the ratio of local loss-curve slopes at lambda*
under the two weightings. Estimate both slopes per cell from the archived
calibration draws (finite differences across the threshold grid, window
preregistered), yielding kappa_pred per cell. Test: realized excess / a
against kappa_pred, per cell, across all five environments. Register: the
regression of measured on predicted kappa has slope in a window
containing 1 (choose and defend the window), and kappa_pred explains the
between-environment slope differences better than the environment label
does (compare R^2, criterion preregistered). If this holds, the constancy
claim is RETIRED and replaced by: kappa is computable per deployment from
calibration data alone. That is the claim that generalizes, because it no
longer asserts anything about unseen environments.

Statistics fix, non-optional: wherever any current document states slopes
are "statistically indistinguishable," restate as equivalence or retract.
Preregister an equivalence margin for slope pairs, run TOST, and report
power at the achieved SEs. Failure-to-reject is not sameness; this is our
own rule and F13's verdict currently violates it. File the correction as
an amendment to the wp1mf registration, not a silent edit.

---

## Block B: the knob ledger and sensitivity sweeps (answers D1)

### B0. design_decisions.md

One row per constant: value, origin (theory / convention / convenience),
what conclusion it could plausibly move, and which experiment (existing or
below) tests it. Constants that must appear: alpha = 0.10; clip
(0.01, 0.99); tree depth 7 and min leaf 40; demo noise 0.06; n_cal, n_fit,
n_eval = 1000; two shift levels per collapse battery; bounded tanh/linear
tilt features; residual tolerance; slope window [0.8, 1.5]; z = 1.645 and
delta = 0.005; Bonferroni scope. "Convenience, untested" is an acceptable
entry only if the row says what it would take to test it.

### B1. alpha sweep (one environment, claims)

alpha in {0.05, 0.20} beside the archived 0.10, shift-response arms plus a
reduced collapse battery (temper family only is enough). kappa_pred from
A3 machinery FIRST, then register the predicted slopes at each alpha, then
run. This converts an arbitrary constant into a tested prediction of the
identity.

### B2. Score-granularity sweep (claims variants)

Re-induce claims rules at depth {4, 7, 10} and min-leaf {80, 40, 20},
producing distinct-score counts from tens to hundreds. Gate each variant
(G5 especially). Temper-family collapse per variant. Register: pooled and
positive-branch slopes move toward the A3-predicted kappa as score
granularity rises; plateau conservatism (oracle margin below alpha)
shrinks. This is the discreteness-attenuation test for D4's other half.

### B3. A second induction family (the real generalization probe for D3)

One environment (claims), rules induced by a probabilistic scorer
(logistic model or small forest with calibrated leaf probabilities), so s
is near-continuous and per-case rather than plateaued rule-level.
Re-gate from scratch; G3 and G5 will behave differently and that is part
of the result. Then shift-response and temper-family collapse. Register
slope predictions from A3 machinery before running. Whatever happens,
the tree-only caveat either falls or becomes a measured boundary.

### B4. Clip as a swept parameter

The cross-env boundary finding says the clip binds at extreme tilts and
breaks oracle-equivalence. Promote it from footnote to swept knob on the
two blind-driver environments: clip in {(0.01, 0.99), (0.005, 0.995),
(0.002, 0.998)} at the top two tilt levels. Register where equivalence is
recovered and what it costs in ESS.

---

## Block C: the adversarial-kappa environment (the generalization claim, done right)

Only after A3 verdicts. Design environment 6 with loss-curve geometry
near threshold deliberately chosen so kappa_pred lands far from 0.85
(target: below 0.5 or above 1.5; achievable by concentrating or starving
wrong-mass just above the operating threshold in the generator design).
Freeze the design, compute kappa_pred from calibration draws, REGISTER the
predicted collapse slope with a window, then gate and run the battery.

Outcome logic, registered: prediction hits -> constancy was five similar
worlds and the correct claim is "kappa computable, slope predicted";
prediction misses -> the identity's kappa is incomplete and the miss
localizes what it omits. There is no outcome in which this experiment is
wasted, and there is no version of the old constancy claim that survives
it unchanged, which is the point.

---

## Sequencing and cost

A1-A3 are recomputation plus statistics: hours, mostly archived data, and
they must come first because B1-B3 and all of C register predictions
GENERATED by the A3 machinery. B0 is an afternoon. B1, B2, B4 are cheap
sweeps; B3 is a day (new gates). C is last and is the block's only new
environment.

## Explicitly out of scope

Paper prose of any kind. New headline claims. Extending APE. The sixth
holdout-certificate environment (that is WP2's test and must not be
burned here; environment 6 above is an ADVERSARIAL-kappa world, not the
certificate holdout, and the two must be different environments).
