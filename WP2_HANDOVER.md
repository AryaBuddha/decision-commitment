# WP2_HANDOVER.md - From the empirical law to a proven, deployable certificate

You are continuing the decision-commitment research project. WP1 is complete:
read findings.md (F1-F18), design_decisions.md, and wp2_notes.md before any
work. CLAUDE.md's invariants remain in force (hash-enforced registrations,
predictions and falsifiers before runs, three-outcome decision rules, TOST
for equivalence, misses verbatim, append-only artifacts, RUNLOG lines, no em
dashes in prose). PAPER PROSE STAYS FROZEN until Phase 5 verdicts are filed;
PAPER_DRAFT.md contains retracted claims and must never be committed.

## The state of knowledge (what WP2 inherits)

The measured law, currently EXPLORATORY in its full form:

    excess(cell) = m(alpha) * kappa_pred(cell) * a(cell) + b(env)

where a = E_P0[(w - w_hat) L(lambda*)] is the signed aligned weight error,
kappa_pred is the local loss-curve slope ratio at lambda* (per-cell
computable from calibration draws; predicts measured kappa at slope 1.243,
R2 0.780 within the adversarial spike world), b is the oracle plateau
conservatism (matches measured plateau margins to the third decimal in 5/5
environments; collapses to -0.0015 on continuous scores), and m(alpha) is
an UNEXPLAINED amplification: ~1.19 at alpha 0.10, ~1.6-1.77 at 0.05,
invariant across induction families and environments at fixed alpha.
The coordinate is local: a evaluated at lambda* saturates when the
operating threshold sits far from lambda* (A1 boundary, 38/42 in-range).
Known instrument hazards: sklearn logistic regularization (C=1.0) shrinks
extreme log-odds and broke oracle-equivalence at extreme tilts (B4);
evaluation-window binomial noise floors dispersion (F16).

## Operating principles for this phase (non-negotiable)

1. RISKY PREDICTIONS MANDATORY. Every registration must contain at least
   one prediction whose failure would force revising the law, marked
   "risky" in the file. Registrations made only of safe bets are returned.
2. EXTREMES IN EVERY GRID. Every swept parameter includes at least one
   value outside the previously tested comfortable range (alpha down to
   0.02; n_cal down to 100 and up to 10000; tilts to the rejection-sampling
   feasibility edge). If an extreme cell breaks something, that is a
   result, not a nuisance.
3. ANOMALIES GET CHASED. Any residual structure in any verdict (systematic
   sign, per-environment pattern, level dependence) gets a registered
   follow-up before it is narrated. "Probably second-order" is a
   registration, not a sentence.
4. FALSIFICATION IS SCHEDULED WORK. Phase 4 exists to break the
   certificate. Treat it with the same effort as the phases that built it.

## Human checkpoints (stop and surface to the user)

After Phase 0 verdicts; whenever theory and empirics contradict; before
freezing the certificate (Phase 3 exit); before building environment 7
(Phase 5 entry); after Phase 5 verdicts.

---

## Phase 0: the confirmatory synthesis battery (the WP1 -> WP2 bridge)

The affine law, per-cell kappa, and m(alpha) were all DISCOVERED through
inverted predictions and exploratory analysis. By this project's own rules
they are not results until they survive a registration written in advance
and fresh data.

Register, then run on FRESH SEEDS (new instance draws, new trial seeds):

- Environments: claims, claims-logit, spike, tickets (one blind-driver,
  one continuous-score, one adversarial-kappa, one high-damage world).
- Alpha grid: {0.02, 0.05, 0.10, 0.15}. 0.02 is the mandatory extreme:
  wrong-mass near threshold is scarce there and the law has never seen it.
- n_cal grid crossed at claims: {100, 250, 1000, 4000, 10000}.
- Cells: temper family (dense gamma grid) plus two realistic axes per env.
- Quantitative registered predictions: b(env) equals the independently
  measured plateau margin within a stated tolerance; kappa_pred regression
  slope window (state it from Block C, widen honestly for fresh data);
  m(0.10) and m(0.05) windows from B1/B3.
- RISKY prediction (mechanism-committal): m depends on n_cal, decreasing
  toward a limit as n_cal grows, because the leading candidate mechanism is
  empirical-crossing noise against a convex loss curve (see Phase 1 H2).
  If m is flat in n_cal, H2 is dead on arrival and Phase 1 starts knowing it.
- Fold in two cheap closures: A1 locality (recompute off-curve cells with a
  evaluated at each arm's own operating threshold; registered prediction
  they land on-curve) and the ledgered B4 follow-up (C in {1, 10, 100,
  unregularized} at the affected cells, design-audit prediction first).

Exit criterion: the law's confirmed form (possibly corrected) is stated in
one equation with fresh-data verdicts attached. Everything downstream uses
that form, not the exploratory one.

## Phase 1: the m(alpha) mechanism hunt (the curiosity core)

Four candidate mechanisms. Each gets a DISCRIMINATING registered
prediction: an observable where the mechanisms disagree.

- H1 grid-and-pseudo-loss conservatism: the inf over a finite lambda grid
  plus the B/(n+1) term biases thresholds. Discriminator: refine the
  lambda grid 10x and vary the pseudo-loss handling; H1 predicts m moves,
  others predict indifference.
- H2 empirical-crossing noise + curvature: lam_hat solves a NOISY
  estimating equation; threshold noise against convex rho(lambda) yields
  Jensen amplification, growing as alpha shrinks (fewer loss events near
  threshold) and as n_cal shrinks. Discriminators: m(n_cal) slope from
  Phase 0; and an oracle-procedure run with n_cal = 100000, where H2
  predicts m -> 1.
- H3 per-covariate threshold averaging: Prop 2's lambda_hat(x) varies with
  w(x); averaging excess over a threshold distribution against convex rho
  amplifies. Discriminator: compare literal Prop 2 against the global
  shortcut arm (single threshold) at matched cells; H3 predicts the
  shortcut has smaller m.
- H4 sign-asymmetric second order already visible at pilot |a| ~ 0.04:
  m is not a constant at all but the local secant of a curved relation.
  Discriminator: m fitted on |a| <= 0.005 cells only; H4 predicts it falls
  toward 1, others predict stability.

Build the ANALYTIC WORLD for this phase: revive the rung-1 placeholder
(Gaussian conjugate tilts) where rho(lambda) is computable in closed form,
so m can be decomposed exactly, term by term. Pilot tier by definition, and
that is fine: mechanism identification there, confirmation on gated
environments after. Exit: one mechanism (or a stated combination) survives
its discriminators on both tiers, and m is either DERIVED (formula in
n_cal, alpha, local loss geometry) or BOUNDED with a mechanism-backed
constant. If all four die, say so loudly; that is the most interesting
outcome and goes straight to the human checkpoint.

## Phase 2: the theorem (Job 1)

Write wp2_theory.md with three graded results, each numerically verified
against the analytic world before being believed:

- Lemma (intercept): for discrete score support, the CRC/Prop-2 threshold
  overshoots conservatively by at most the loss-mass of one score plateau;
  b in [-(plateau mass), 0], tight when thresholds land mid-plateau.
  This formalizes F17 and should be fully provable.
- Proposition (first order): excess = kappa * a + R under a local
  regularity condition (rho' bounded away from 0, Lipschitz near
  lambda*), kappa the local slope ratio, with the Phase-1 mechanism
  incorporated (m derived, or the statement given in expectation over the
  empirical estimating equation so amplification appears explicitly).
- Remainder characterization: R sign-asymmetric (convex dangerous side,
  saturating safe side), with the pilot-measured boundary |a| ~ 0.04 as
  the empirical anchor.

Honesty rule: mark each result proved / proved-under-condition /
conjectured-with-evidence. No result enters the certificate at a rigor
level above its marking.

## Phase 3: the observable certificate (Job 2)

a is oracle-referenced; deployments cannot compute it. Construct the
surrogate and the certificate:

- A_hat: held-out, LOSS-REGION-LOCALIZED calibration bound on the ratio
  classifier (its signed odds miscalibration integrated against the loss
  indicator near and above the operating threshold), per wp2_notes
  section 6. APE is the plug-in cousin; A_hat must be an upper bound
  with a stated confidence, not a point estimate.
- Certificate: alpha_cert = alpha + m_max(alpha, n_cal) * kappa_pred *
  A_hat + b_bound + margin(dispersion floor per F16).
- Registered in-sample validation: the certificate must cover ALL archived
  evidence-tier cells (state the count), and its price must be reported
  honestly: mean conservatism and added deferral versus the oracle.
  RISKY prediction: state in advance the fraction of cells where the
  certificate is within 2x of realized excess (a certificate that covers
  by being vacuous fails the point; F4 is the cautionary tale).

Exit: certificate frozen by commit. Human checkpoint before freezing.

## Phase 4: red team (break your own certificate)

At least two registered attack rounds. Attack surface, in order of
promise: (i) miscalibration mass placed just OUTSIDE the audited loss
region but inside it under the target law (attacks the localization);
(ii) estimator errors crafted to be calibration-invisible on held-out
source data while maximally loss-aligned under the target (attacks the
surrogate); (iii) score families or shifts that move lambda* between
audit time and deployment (attacks the locality boundary from A1);
(iv) the B4 lesson generalized: hunt for other silent library defaults in
the certificate pipeline. Each round: register the attack and its
predicted breach size, run it, then either file the certificate's
survival with quantified slack or REVISE the certificate and re-attack.
A certificate that has never been attacked is a hope, not a result.

## Phase 5: the frozen-certificate prospective test (the capstone)

Only after Phase 4. Build environment 7 as the HOLDOUT (spike/env 6 is
spent and must not be reused): design it after the certificate is frozen,
gate it, compute the certificate's inputs from its deployment-visible
data only, register the certificate's predicted safe operating envelope,
then sweep. Report coverage and price, whatever they are. This single
experiment answers both the one-authorial-hand objection and the
does-it-actually-deploy question, and it is the paper's closing figure
when prose eventually resumes.

## Phase 6 (optional, only if ahead of schedule)

Group-conditional two-step on claims; selective-error certification via
Learn-then-Test at fixed threshold; gradual-drift design for the
fixed-weights literature. Each is a separate registration; none blocks
the capstone.

## Definition of done for WP2

1. Phase 0 fresh-data verdicts on the law's final form.
2. m(alpha) mechanism identified or all candidates falsified, on both
   tiers, with the analytic-world decomposition on file.
3. wp2_theory.md with graded, numerically verified results.
4. A frozen certificate that survived two red-team rounds, with in-sample
   coverage and price quantified.
5. Prospective coverage verdict on a holdout environment built after the
   freeze.
6. Registrations, verdicts, misses verbatim, RUNLOG current, ledger rows
   for every new knob.
