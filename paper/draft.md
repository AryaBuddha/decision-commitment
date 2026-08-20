# Commitment under Distribution Shift: an exact coordinate for estimated-ratio risk control, and a certificate that survived its own red team

Working draft, rewritten 2026-08-19 after the WP2 verdicts and the two
post-completion corrections. Prose was frozen until every claim below
existed first as a registered prediction with a filed verdict; numbers
cite config hashes; misses are quoted, not paraphrased. Placeholder-tier
numbers appear only in the narrative of how the platform was built.

## Abstract

Systems that induce decision rules from observed work must decide, per
case, whether to commit to the rule's output or defer to a slower
process. Conformal risk control (CRC) certifies the marginal
commit-error rate at a level alpha under exchangeability, which
deployment shift breaks; the known repair (Angelopoulos et al. 2022,
Proposition 2) assumes the covariate-shift likelihood ratio is KNOWN,
and its authors leave the estimated-ratio case explicitly open. We study
that open case on a controlled-violation platform: seven generated
environments whose decision rules are genuinely induced from noisy
demonstrations, whose ground truth is mechanically recomputable, and
whose shift is exact and dialled. Four results. (1) The failure has a
one-dimensional coordinate: excess risk under estimated-ratio
Proposition 2 collapses onto the signed aligned weight error
a = E[(w - w_hat) L] in every environment (five worlds, 289 of 290
cells within preregistered tolerance), while error magnitude does not
predict damage and the published robustness envelope is vacuous by two
to three orders of magnitude. (2) The collapse relation is affine with
an intercept owned by score discreteness, and its residual
amplification m(alpha), left open by our own prior work, is solved: a
finite-sample charge asymmetry (derived in closed form), crossing noise
(dies at root n), and a discrete-crossing plateau arithmetic that
survives infinite calibration and reproduces the amplification's
environment ordering; in the coordinate of the deployed rule's OWN
threshold the law is exact and the amplification does not exist. (3)
That exact coordinate yields an observable certificate: alpha_cert =
alpha + a_plugin + CalErr_loc + b_own + margins, computable from
labelled source data and unlabelled target covariates only, with no
oracle reference and no worst-case translation constant. Validated over
674 archived evidence-tier cells, it covers every dangerous cell;
two registered red-team rounds then broke it once (post-audit drift),
which became a monitored validity precondition, and exposed one silent
load-bearing default (reliability binning), which was pinned. (4) On a
holdout environment built AFTER the certificate froze, with the
envelope registered from deployment-visible data before any sweep ran,
the certificate covered 12 of 12 deliberately degraded deployments
while six of them genuinely failed their certified level (realized
excess up to +0.036), with the envelope approached (0.37 of the bound)
and not crossed. The certificate's honest character is stated with its
coverage: it protects by measured margin, roughly a doubling of alpha
at level 0.10, and the slack is attributed term by term.

## 1. The problem and the open question

Angelopoulos et al. (2022), Section 4.1, prove Proposition 2: weighted
CRC with the TRUE covariate-shift ratio w(x) controls risk under shift.
The section closes by leaving the estimated-ratio case to future
research. That sentence is this paper's question, and it matters
because deployments never have w; they have a classifier-odds estimate
w_hat fitted from finite samples, with a view of the covariates that
may omit exactly the attribute that drifted.

Setting and endpoints, kept structurally separate throughout: the
evidence score s is a property of the induced RULE that fired (support
and consistency on the induction log), not of the case; the certified
endpoint is the marginal error mass P(commit AND wrong), the quantity
with a theorem attached; commitment rate is the utility; selective
error P(wrong | commit) is reported descriptively and never certified
(it has no CRC theorem; certifying it needs Learn-then-Test machinery
we do not use here).

## 2. Methodology: a controlled-violation study

Every environment is generated, so violations of exchangeability are
exact and known, and every claim about an estimator faces ground truth
that the experimenter cannot fudge: instances come from seeded
generators with latent structure; rules are induced by standard
learners from noisy expert demonstrations on an INDUCTION VIEW that
omits designated attributes; gold is deterministic in the stored
instance and mechanically recomputed; shift is an exponential tilt on
BOUNDED manifest features, sampled exactly by rejection (rung 2), with
divergence reported per level. Environments pass five validation gates
before any experiment may cite them (solvable gold; exact tilts against
importance-weighted moments; decision-conditional blind-feature lift;
region mass under every tilt; enough distinct evidence plateaus).

The protocol is registration-first: every experiment's runner hashes
its config and refuses to run unless the hash matches a committed
registration containing directional predictions and named falsifiers;
verdicts are written back prediction by prediction with misses quoted
verbatim; artifacts are append-only; every design constant carries a
ledger row stating its origin and which experiment tests it. The
record includes five registered risky predictions that failed and are
reported as such below; two of those failures forced revisions of the
paper's own instrument. The platform's own history includes two gate
failures, three retractions by amendment, and one instrument bug found
by a red-team round; all are in the audit trail, none were cleaned up.

Environments (all evidence-tier claims come from these): claims triage
(blind driver with a proxy; the first real environment), tickets
(independent blind driver, no reconstruction), fraud (strong
correlates, reconstruction defends), moderation (visible dominant
driver), compliance (weak correlates, noisy experts, coarse rules),
spike (adversarial loss-curve geometry, built to break the slope
formula), and returns (the holdout, built after the certificate
froze). A smooth-scored variant (claims-logit) and a quadrature-exact
analytic world support the mechanism work.

## 3. The failure map

### 3.1 The decay is real and the known bound explains none of it

On claims (ab56864e6cfb3400, 500 trials/level, Bonferroni), unweighted
CRC is in violation from beta >= 1, decaying monotonically from 0.098
to 0.123 across chi2 0 to 3.5, while oracle Proposition 2 is consistent
at every level; the Proposition 3 total-variation envelope exceeds 1
from the first nonzero tilt at n_cal = 1000: vacuous by two to three
orders of magnitude. Across the four later environments the decay
ranges from +0.19 (tickets) to a full defense (fraud, non-monotone and
back inside the certified level), at MATCHED chi2: how much the mix
drifted says little; where the drift falls relative to what the rules
see and what drives correctness decides the damage (evidence-tier
tilt-location experiments 3879b17b6fa4ae6f and bf521b1f263f8cc8: at
matched chi2, damage differs threefold, the visible tilt loses to
gold-relevance on claims, and the visible-blind ordering CROSSES with
shift size on moderation).

### 3.2 One coordinate: the signed aligned error

Excess marginal risk under estimated-ratio Proposition 2 collapses on
a = E_P0[(w - w_hat) L(lambda*)], the estimator's signed error
projected onto the loss at the oracle threshold. Evidence tier: 58
cells on claims (four realistic degradation axes, two synthetic
families; Spearman 0.956, 58/58 within preregistered tolerance,
56704982681d6960), then 4 x 58 cells on tickets, fraud, moderation,
compliance (0.903 to 0.961; 289/290; wp1mf runs). Error magnitude does
not predict: at matched L1 near 0.4, deprivation realizes +0.04 to
+0.08 excess while dimension inflation realizes slightly negative
excess. The B*L1 envelope is valid (58/58) and useless (5x to 32x
loose): only the signed, loss-aligned component of estimator error
moves the threshold.

Two registered corrections to our own first reading of the collapse,
both by archived-data recomputation (blockA_e58d1bc4b8627b0d): the
relation is AFFINE, its intercept matching each environment's oracle
plateau conservatism to the third decimal in five of five
environments; and the through-origin slope constancy we initially
claimed was a failure-to-reject dressed as equivalence (TOST 0 of 10
pairs; the claim is retracted by amendment). The corrected functional,
the paired difference (estimated minus oracle) against a, collapses at
R2 0.94 to 0.99 per environment with genuinely unequal slopes.

### 3.3 A deployment-computable forecast

APE, the audit-predicted excess (an audit model on held-out labelled
source data, integrated against unlabelled target covariates at the
frozen threshold) was registered, recipe and thresholds included,
BEFORE environments 2 through 5 existed, and then realized decay
within 0.0063 at every one of 28 preregistered level-cells with the
forecast ordering exact (cross_environment_hypothesis). The registered
design-intent ordering, by contrast, missed on the pair the audits
flagged: the statistic tracked reality while intuition did not.

## 4. The law, confirmed fresh and then solved

### 4.1 Fresh-seed confirmation with the extremes registered

Because the affine law, the per-cell slope predictor kappa_pred, and
the amplification m(alpha) were discovered partly through exploratory
analysis, WP2 re-derived all of them on fresh seeds under registration
(wp2p0_ec15383b39b52206: four environments, alpha down to 0.02;
wp2p0b_16996ec167284e46: n_cal 100 to 10000). Confirmed: intercept
anchors within 0.0004 of archived in all four environments; oracle
safety in 384 of 384 cells; every registered m window hit; per-cell
kappa_pred structure at R2 0.52 to 0.87 wherever the guards leave
enough cells. Corrected by the registered extremes: m is NOT a
universal monotone function of alpha (tickets: 1.07 at alpha 0.10,
1.84 at 0.05, 1.26 at 0.02), family invariance of m holds at alpha
0.10 (spread 0.158) and fails at 0.05 (spread 0.69), and at alpha 0.02
the aligned error scales below an absolute measurability guard in
three of four environments: the law's extreme-alpha domain is bounded
by measurability, not by violation. The budget grid found m falling
steeply with calibration budget (2.5 to 1.1) but PLATEAUING above 1 at
an alpha-dependent limit, with threshold noise itself scaling at root
n: a budget-independent residual demanding a mechanism.

### 4.2 The amplification solved

A quadrature-exact analytic world decomposes m term by term
(wp2sw_532eecd6e078f48b). The finite-sample pseudo-loss charge
B/(n+1) is a DERIVED amplifier: Proposition 2 charges each arm
(1 + chi2_arm)(B - alpha)/n_cal of effective level, arms differ in
chi2, and the asymmetry amplifies the paired difference (1.70 at alpha
0.02, n_cal 250), dying by n_cal 10000. Crossing noise dies at root n
and partially CANCELS the charge at small budgets. Per-covariate
threshold averaging contributes nothing (gap at most 0.001; the H3
candidate is dead), curvature is structurally absent in that world yet
the gated plateau exists (H4 dead independently by a registered
split: 8 of 8 blocks, with small-a halves sitting FARTHER from 1), and
the smooth world has NO plateau at all. The last structural difference
standing was score discreteness, and the quantized world confirmed its
registered risky prediction (wp2qsw_0a8f6613460472fd): the two arms
cross on DIFFERENT score plateaus, excess = a(lam_e) + overshoot with
a(lam_e)/a(lam*) a ratio of loss-curve values at the two crossings, an
arithmetic that survives infinite calibration, amplifies OR compresses
depending on where alpha lands (2.16 at coarse quantization and alpha
0.05; 0.78 at alpha 0.10), and thereby explains our own earlier
granularity non-monotonicity. On the gated tier the identity-derived
per-cell m reproduces measured m at slopes 0.97 to 1.12 (R2 0.79 to
0.96 wherever plateau spread exists) and reproduces the alpha-0.05
environment ordering including the tickets anomaly
(wp2mc_4f15d0310ff68a3f). Honest boundary, filed in the verdicts: the
derived arithmetic is exact only in the large-budget limit; at
intermediate budgets crossing noise softens it from below, so m is
bounded between regimes, not pointwise derived.

### 4.3 The exact coordinate

The reframing everything downstream uses: for ANY deployed commit
function C(x), algebraically,

    excess = E_Q[K C] - alpha
           = E_P0[(w - w_hat) K C] + (E_P0[w_hat K C] - alpha),

the aligned error AT THE DEPLOYED RULE plus the rule's own crossing
margin. Measured with both terms on independent source draws, all 15
recomputed off-curve cells from the earlier boundary reconstruct
excess within 0.0015 at pooled slope 1.013 +/- 0.005
(wp2p0c_a801f76c2d55fb4d), and the identity holds within 2 SE in 122
of 128 tempered-arm cells across four environments (wp2mc). In this
coordinate there is no kappa, no m, and no second-order regime: the
lambda*-referenced law is the composition of an exact identity with a
threshold translation, and the amplification literature above is the
anatomy of that translation. Theory (wp2_theory.md, every result
numerically verified, 58/58 checks): the intercept lemma (crossing
margin bounded by one plateau's loss mass; proved), the first-order
law with an explicit proved remainder bound Lip (1 + |kappa|)/c^2 a^2,
the charge formula (proved), and the plateau arithmetic (derived;
quantitative in the large-budget limit). The pilot-era claim that the
remainder's sign is universally asymmetric is DOWNGRADED: a rank-2
world compresses where the pilot steepened; only the magnitude bound
is general.

## 5. The certificate

### 5.1 Form

Deployments cannot compute a (it contains w). The certificate bounds
it observably, in the exact coordinate, from labelled held-out source
data, the deployment's own w_hat and calibration draw, and unlabelled
target covariates:

    alpha_cert = alpha + max(0, a_plugin + CalErr_loc + b_own_ucb + z se),

where a_plugin integrates an audit model m_hat(s, dec, x) (held-out
source fit; the conditional p(x) is invariant under covariate shift)
against the deployed commit region under both weightings, CalErr_loc
bounds the audit's miscalibration RESTRICTED to the commit region
(reliability bins, measured gap plus a binomial confidence charge,
masses transferred through the unlabelled target sample), and
b_own_ucb is the measured crossing margin's upper confidence bound.
The certified excess is floored at zero: a certificate exists to bound
risk from above, and certifying risk strictly below alpha is an
over-claim (this floor is Revision 3; it was adopted after the v2
validation exposed 26 safe-side technical misses), and the
crossing-margin credit inside the bound is likewise never spent
(Revision 3.1, adopted through a registered draw-tail follow-up whose
numbers are in Section 5.2). Validity carries a
deployment-visible FRESHNESS PRECONDITION (Revision 1, below). No
worst-case translation constant appears anywhere: the m_max kappa_pred
shape our own plan once envisioned was rejected at the freeze because
the coordinate makes it unnecessary.

### 5.2 In-sample validation, red team, and slack attribution

Validated over all 674 archived evidence-tier estimated-arm cells at
30 deployment draws each, final form v3 (wp2cert_7f8658cdfc4cac42):
673 of 674 at the registered 27-of-30 rule, with the single miss among the three cells the registration named in advance; its registered follow-up chain (a draw-tail diagnosis whose own prediction missed, then an object-matched per-draw test that failed at 84 of 100) produced Revision 3.1, the crossing-margin credit is never spent, under which the exact per-draw recompute covers 674 of 674 with zero dangerous-cell median failures. The safety statistic: zero of the 219 dangerous cells
(archived excess >= 0.005) have a median bound below their excess, in
either the v2 or v3 runs. Price: mean conservatism +0.060 under v3 and +0.065 under v3.1; median looseness on dangerous cells 5.1x at the pinned conservative binning (15 of 219 within 2x). The registered
anti-vacuity clause of the v2 run MISSED and is reported verbatim:
median looseness on dangerous cells 4.1x against the hoped-for 2x,
better than the B*L1 envelope's 5x to 32x but plainly an instrument
that covers by margin. The registered slack attribution
(wp2_slack_attribution) says where the margin lives: CalErr_loc owns 87.4% of the mean slack on dangerous cells; within it the binomial confidence charge owns 58.9% and the measured audit-miscalibration gap most of the rest; the four statistical terms account for 100.0% of the slack, so no structural worst-case constant remains to remove; and the tightening lever is measured, not conjectured: quadrupling the held-out audit samples cuts the worst-cell bounds by 0.030.

Two registered red-team rounds attacked the frozen form, each attack
with a predicted breach size. Survived: an audit-blindness times
blind-estimator attack aimed at the model class's unrepresentable
interaction (breach -0.035, inside the predicted window); per-draw
coverage on the highest-variance estimator cells (95.5% and 91%
against the mismatched cell-mean reading). BREACHED, as predicted at
its registered edge: post-audit drift, at +5.9 beta-units of
deployment drift against a stale audit (+0.014 median breach), with
the stale bound barely moving. The breach became Revision 1: a drift
monitor on arriving covariates (the same classifier machinery the
deployment already runs), tolerance 0.10, verified in round 2 to
separate surviving from breaching drifts by 9.2x and to flag the
breaching deployment in 30 of 30 draws: the silent failure is now a
detected one. Round 2 also hunted the pipeline's own silent defaults:
the audit model's regularization default is benign (at most 0.0092),
but the reliability BIN COUNT is load-bearing (within-bin cancellation
hides up to 0.031 of miscalibration), so the final recipe pins the
conservative end (Revision 2). The honesty note from round 1 stands
and the paper adopts it as the certificate's character: survival
inside the freshness window rests substantially on CalErr_loc margin,
not on drift tracking.

## 6. The prospective tests

Environment 7 (returns) was designed after the certificate froze,
passed all five gates on its frozen design, and deliberately stresses
the audited weak points: a gold interaction outside the audit model's
class and a blind driver with only a weak logged proxy. The protocol
is two-stage: stage A computes the certificate envelope from
deployment-visible data only and the numbers are registered as the
predictions; stage B sweeps realized risk on fresh streams.

First run (wp2p5_7644a21fb970d7bd), the well-specified deployment:
12 of 12 cells covered at mean price +0.056, and the registered risky
danger-location clause MISSED verbatim: the deployed estimator tracked
the oracle within 0.003 everywhere (the weak proxy's reconstruction
defended more than the ledgered regularization shrinkage hurt), so
this run validates the workflow, not the protection.

The stressed rerun is the protection test (wp2p5s_8e357241960a63b7):
the realistic degradation axes instantiated on the same holdout, with
a disclosed probe first filing that fit-based degradations are heavily
defended on this world (the manifest's segment structure reconstructs
the blind driver), so the honest stress is NO CORRECTION, which is
also the most common deployment reality. Result, registered envelope
against 300-trial sweeps: 12 of 12 covered while the six no-correction
deployments genuinely failed their certified level (excess +0.014 to
+0.036, up to 30 SE above zero); the envelope was approached (maximum
approach ratio 0.37) and not crossed; minimum price +0.048; oracle
consistent everywhere. Figures: docs/wp2_prospective_envelope.png and
docs/wp2_prospective_stress.png.

## 7. Limitations and threats

Generated worlds, one authorial hand. Every environment was built by
the same process that built the methods, and gold is recomputable
because the worlds are synthetic. The holdout protocol (built after
the freeze, envelope before sweep) answers the objection's sharpest
form but not its general form; external replication on found data is
the real test, and APE plus the certificate are specified so a
replicator needs only labelled source data and unlabelled target
covariates.

The certificate covers by margin. Its median looseness on dangerous
cells is 4.1x, the stressed capstone's closest approach is 0.37 of the
bound, and the slack attribution shows the margin is roughly half a statistical confidence charge that shrinks with audit size (quadrupling the held-out samples bought 0.030 of bound on the worst cells) and roughly a third measured audit miscalibration that only a better audit model removes.
A deployment at alpha 0.10 should expect alpha_cert near 0.15 to 0.21
under stress. We report this as the price of observability at these
audit sizes, not as a virtue.

Boundaries measured and carried: validity requires the freshness
precondition (drift monitor at tolerance 0.10); the law's
extreme-alpha domain is measurability-bounded (alpha 0.02 at n_cal
1000 is measurable only where aligned errors are large); evaluation
windows floor realized-risk dispersion (exponent 0.32, not 0.5), so
past n_cal near 1000 the next label buys more as monitoring than as
calibration; selective error is never certified here.

What did not survive our own tests, kept in the record: slope
constancy (retracted by amendment), the clip-binding diagnosis
(replaced by the regularization default, itself found by a registered
sweep), the universal remainder sign (downgraded), the danger-location
model on the holdout (missed verbatim), and the pilot's
blindness-is-dangerous reading (requalified to gold-relevance net of
self-correction).

## 8. Reproducibility

Everything is code: environments, gates, registrations, runners,
analysis. Each experiment's config hash is enforced against its
committed registration before it will run, `git log registrations/`
proves predictions preceded runs, artifacts are append-only and cited
by hash, and RUNLOG.md carries one line per run including every pilot
and every miss. The unit-test gate (exact reduction of Proposition 2
to unweighted CRC at w = 1; exactness of rung-2 rejection sampling)
runs before every experiment. The two figures the paper closes on were
produced by registered runs whose predictions were the figures'
contents.
