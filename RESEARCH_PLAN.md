# Research Plan v2: Commitment under Distribution Shift

**Revision note.** This version responds to an external review of v1. Three v1 claims were wrong and are corrected throughout: the oracle-weighted risk-control procedure is a published theorem, not a conjecture (Angelopoulos et al. 2022, Section 4.1, Proposition 2); finite-pool tilted resampling is an approximation, not exact, and has been replaced with an exact sampling mechanism; and the v1 decision rule mistook failure-to-detect-violation for evidence of control. The novelty claim about the oracle arm is also corrected. All amendments are recorded in the registration file, not silently overwritten.

---

## Part 1. The research in plain terms

A system watches experts work, extracts decision rules from what it saw, and then applies those rules on its own. For each new case it makes one choice: **commit** (apply the rule and record the answer) or **defer** (send the case to a slower, more careful process). The safety question lives in that choice.

Two principled ways exist to set the commit threshold with a guarantee attached, and each is blocked in the deployment you care about.

The first is **conformal risk control** (CRC). Hold out cases with known answers (the *calibration set*), measure how error falls as the evidence threshold λ rises, and pick the λ that certifies expected error at or below a chosen level α. Deployment breaks the required interchangeability between calibration and future cases. The repair is known: reweight calibration cases by the **likelihood ratio** w(x), how much more common a case like x is in deployment than in calibration.

**The theorem status, stated correctly this time.** Angelopoulos et al. (2022), Section 4.1, Proposition 2 proves that the reweighted procedure with a *test-covariate-dependent* threshold,

    λ̂(x) = inf{ λ : ( Σᵢ w(Xᵢ)Lᵢ(λ) + w(x)·B ) / ( Σᵢ w(Xᵢ) + w(x) ) ≤ α },

controls expected target-domain risk at α for any bounded monotone loss, when w is the true ratio. They state that this generalizes weighted conformal prediction to any monotone risk, and they close the section by leaving *weighted conformal risk control with an estimated covariate shift* explicitly for future research. That sentence is Question 1's citation anchor: the field's own authors name your problem as the open one. WP1's oracle arm is therefore a **verification of a published theorem plus a test of the theorem-to-code path**, and WP2 begins exactly where the published theory stops.

The second literature is **off-policy evaluation**, blocked by deterministic logging (Question 2, WP3). Unchanged from v1.

**Methodology: a controlled-violation study.** Take a procedure provably correct under an assumption. Deploy it where the assumption is violated by a known, dialled amount. Measure realized against certified performance as the dial turns. Then certify what survives.

**The novelty claim, corrected.** v1 claimed the oracle-ratio arm "exists nowhere in the empirical literature." False: Tibshirani et al.'s airfoil experiment constructs an exponential tilt, uses the known oracle ratio, and compares oracle against estimated weighted conformal prediction. The defensible claim is narrower and stronger: this platform extends constructed-shift experiments from coverage to **risk control** (testing Proposition 2 itself, which has no published empirical evaluation), to **executable decision rules with mechanically recomputed gold**, sweeping both shift **magnitude and direction** (evidence-blind vs evidence-visible), deliberately varying **estimator error measured in guarantee-relevant norms**, and reading out **marginal, selective, and region-conditional** commitment risk. No single prior study does more than one of these.

### Small glossary

**λ.** Commitment threshold; commit when evidence score s ≥ λ. Under Proposition 2 the threshold is per-test-covariate, λ̂(x).

**α.** The certified expected-error budget, e.g. 0.10.

**Three endpoints, kept structurally separate.** *Marginal error mass* P(commit AND wrong) is the primary, theorem-aligned endpoint; it is what CRC bounds. *Commitment rate* P(commit) is the utility endpoint (its complement, deferral, is the price of safety). *Selective error* P(wrong | commit) is the product-safety endpoint practitioners usually mean; it is a ratio, not an expectation of a per-case monotone loss, so CRC does not certify it. WP1 reports it descriptively; certifying it requires a separate Learn-then-Test binomial procedure and is scoped as an extension, honestly labelled, because reporting it uncertified does not fully answer the practical safety question.

**Excess marginal risk.** Realized marginal error mass minus α. (Renamed from v1's "coverage gap": in selective prediction "coverage" can also mean commitment rate, and the ambiguity was doing damage.)

**χ² divergence.** Knob-free x-axis for shift magnitude; in the placeholder it is exp(|b|²) − 1 in closed form.

**Effective sample size (ESS).** (Σw)²/Σw², how many equally weighted cases a weighted calibration set is worth. A useful *diagnostic* of the price of weighting, reported next to every risk number. (v1 overstated Tibshirani et al.'s finding as ESS "fully explaining" weighted-method dispersion; that was one matched-subsample observation in one experiment, not a theorem.)

**Preregistration.** Predictions, decision rules, and config hash committed before running; the runner refuses a mismatched config; misses recorded verbatim.

---

## Part 2. Programme architecture

```
Phase 0  Pipeline scaffold, placeholder environment            [DONE]
Phase 0.5 Review corrections: Prop 2 literal, exact sampling,
          four-way splits, decision rules                      [DONE]
Phase 1  Real environment extraction + validation gates
Phase 2  WP1 sweeps and ablations
Phase 3  WP2 (certification under estimated ratios), WP3, WP4
```

Findings carried forward from the corrected pilot (placeholder environment, not evidence, but calibrating):

1. Proposition 2, implemented literally and unit-tested (it must reduce exactly to unweighted CRC at w ≡ 1; it does), holds at every shift level swept, including χ² ≈ 8.5 where unweighted CRC realizes 0.31 against a certified 0.10.
2. The **minority-region failure needs no shift**: region-conditional marginal risk sat at ~0.42 at χ² = 0 while the aggregate was exactly controlled. Marginal guarantees do not transfer to subgroups; shift is an aggravator, not the cause.
3. The **global-threshold shortcut** (one self-normalized weighted threshold instead of per-x λ̂(x)) tracked the literal procedure closely in this sweep, running slightly less conservative at large shift. Since deployed systems will prefer a single threshold, quantifying what the shortcut loses is a practitioner-relevant ablation and it stays in the design, clearly labelled as not-Proposition-2.
4. The correctly specified ratio estimator still tracks the oracle almost exactly (normalized-weight L1 error ~0.08 to 0.10 across the sweep), so **Question 1 has no bite until the estimator is degraded deliberately**. The misspecification sweep is the experiment, not an ablation.

---

## Part 3. Phase 1: setting up the real environment

### 3.1 The data contract and the sampling unit

The calibration pipeline consumes a **case table**: covariates X, evidence score s, commit-would-be-wrong flag, region label. One loader function swaps the placeholder for a real environment.

**The sampling unit is frozen as: one routed decision per generated instance.** Instances are i.i.d. draws of the generator, so units are exchangeable by construction. If an instance can trigger several rules, either the system's actual routing selects the single rule that fires (one decision per instance), or rule families are run as separate experiments. Multiple rows per instance, or rows sharing a rule treated as independent, would silently break the finite-sample guarantee; the review is right that this must be a frozen definition, not an emergent property of the extractor.

**Four genuinely separate data sources per experiment,** so that no reuse can masquerade as robustness:

1. **Induction data**: demonstrations from which rules are learned and evidence scores computed. Never touched again.
2. **Labelled source calibration data**: fresh instances, used only to compute the loss curve and thresholds.
3. **Ratio-fit covariates**: fresh source covariates and fresh (unlabelled) target covariates, used only to fit ŵ.
4. **Labelled target evaluation data**: fresh, used only to measure realized risk.

Cross-fitting is a reasonable secondary condition later; clean splitting is primary. The v1 runner fitted the ratio on the same covariates it calibrated and evaluated on; that path has been deleted.

**Frozen before any real run:** the covariate list (10 to 20 numeric features from instance manifests), the evidence-score formula (a fixed monotone function of rule-level support and consistency; a property of the rule, not the case), and region definitions.

### 3.2 Creating shift: the exactness ladder

v1 claimed pool-splitting-plus-tilted-resampling made the likelihood ratio exact "by construction." The review is right that it does not: resampling a finite pool produces a self-normalized empirical approximation to the tilted population, with dependence through the shared pool. Since exactness is the platform's flagship claim, the mechanism must actually be exact wherever possible. Three rungs, in order of preference:

**Rung 1, analytic tilting (exact).** When the tilt is conjugate to the generator, sample the tilted population directly. The placeholder does this now: tilting N(0, I) by exp(b·x) *is* N(b, I), so target draws come from N(b, I), the normalized ratio exp(b·x − |b|²/2) is exact with E[w] = 1, and χ² and TV are closed-form (validated against Monte Carlo in the code). Note the review's suggested universal fix, rejection sampling, is impossible on this rung: exp(b·x) is unbounded on Gaussian support, so no acceptance bound exists. Conjugacy is the exact mechanism here.

**Rung 2, rejection sampling from fresh generator draws (exact).** For real environments, define the tilt on *bounded* functions of the covariates (clipped or sigmoid features, or inherently bounded counts). Then exp(β·φ(x)) has a known upper bound, rejection sampling from fresh generator output is exact, and bounding the tilt has the side benefit of bounding w itself, which protects ESS. This is the default for real environments, and it is why the covariate-list freeze should prefer bounded features.

**Rung 3, pool resampling (labelled approximation).** Where generation is expensive enough that fresh target draws are impractical, keep the pool design, but call it what it is, note the Tibshirani et al. precedent for it, and preregister convergence diagnostics: pool size sensitivity, normalizer error, evaluation-set collision rate, maximum normalized weight, and target-moment error against the analytic tilt.

Whichever rung, report χ² using the **normalized** ratio w(x) = exp(β·φ(x)) / E[exp(β·φ(X))]; normalization cancels in λ̂(x) but not in divergence or estimator-error numbers.

Resampling vs regeneration is unchanged from v1: regeneration risks concept shift and stays a separate, labelled, harder condition.

### 3.3 The five validation gates

Unchanged in substance: (1) oracle solvability ≥ 0.98 on the actual case sample; (2) covariate-shift fidelity, now mostly an extractor-bug check since rungs 1 and 2 make it hold by construction; (3) **evidence-blindness audit**, fit wrong ~ s vs wrong ~ (s, X), require material lift on the dimensions to be tilted, else that environment cannot exhibit the failure and is redirected or excluded with the audit published; (4) region mass, ≥ 50 expected minority evaluation cases per draw at every tilt level; (5) evidence-score spread across ≥ 20 λ-grid cells.

---

## Part 4. Phase 2: the WP1 run plan

### 4.1 Arms

All arms run on identical splits within a trial (paired design; arm differences are estimated on paired differences).

1. **Unweighted CRC** (Eq. 4). The cost-of-ignoring-shift baseline. Overlay the paper's own Proposition 3 total-variation envelope, E[L] ≤ α + B·Σᵢ TV(Zᵢ, Zₙ₊₁), computable in closed form on rung 1. At n_cal = 1000 this envelope is vacuous (it exceeds 1 at the first nonzero tilt while realized risk is ~0.13), and that gap is itself a finding: the published robustness bound does not begin to explain the observed, much milder decay, which motivates WP2-style sharper analysis.
2. **Oracle, Proposition 2 literal.** Per-x threshold, true ratio, unit-tested reduction to arm 1 at w ≡ 1. A theorem verification and an implementation check. If it ever appears to fail, the presumption is a bug in implementation, sampling, dependence structure, or assumptions, in that order; a genuine counterexample would contradict a published theorem and demands that standard of evidence. (v1's framing of oracle failure as "the best publishable outcome" is retracted.)
3. **Estimated, Proposition 2 with ŵ** from the dedicated ratio-fit splits. The subject of Q1.
4. **Global-threshold shortcut with the true ratio.** Not Proposition 2; the deployment-realistic single-λ variant. Measures what the shortcut loses. (Pilot answer so far: little, and slightly anti-conservative at large shift; worth confirming on real environments.)

**Removed from WP1:** the Barber-style fixed decaying-weights arm. The review is right that index-decaying weights have no principled role when all calibration points are identically distributed source draws; the design they answer is *gradual drift within the calibration sequence*. That experiment is worth running, but as a separately designed extension with P₁, …, Pₙ approaching the target over time and weights fixed before outcomes, not as an arm bolted onto an abrupt-shift sweep.

### 4.2 Sweeps

**Primary: shift magnitude**, ~8 levels, χ² from 0 to ~10, tilt on gate-3-verified evidence-blind dimensions. Outputs per arm: marginal error mass, commitment rate, ESS, region-conditional risk, all vs χ². Proposal Deliverable 1.

**Ablation A: tilt location** (evidence-blind vs evidence-visible dimensions at matched χ²). Shows decay depends on where shift falls relative to the evidence signal.

**Ablation B: estimator misspecification. The load-bearing sweep.** Degrade ŵ along four axes (feature deprivation, dimension inflation, ratio-fit data starvation, model-class mismatch including nonlinear tilts and clip settings). For each degraded estimator record the **error battery**, all on normalized weights: L1(P₀) and L2(P₀) error (primary, because |E[ŵL] − E[wL]| ≤ B·E|ŵ − w| for losses in [0, B], the inequality from which a WP2 excess-risk bound can actually be assembled), log-ratio RMSE under source and target, clipping bias, maximum and upper-quantile weights, ESS, error localized near the commitment boundary, and the alignment between weight error and loss. Deliverable: **excess marginal risk vs L1(P₀) error**, pooled across axes, with the one-curve-or-many verdict, plus whether B·L1 is a valid empirical envelope. Either answer hands WP2 its target.

**Ablation C: calibration budget**, n_cal ∈ {250, 1000, 4000}, the certifiable-shift-vs-budget table.

**Extension: group-conditional calibration**, λ per region, deferral cost measured. The impossibility motivating group-level rather than fully conditional targets is Barber, Candès, Ramdas, Tibshirani, *The limits of distribution-free conditional predictive inference* (Information and Inference, 2021), cited in full here because v1's shorthand was mistaken by the reviewer for the beyond-exchangeability paper; the claim itself stands, stated precisely: exact distribution-free conditional coverage is impossible for nonatomic covariates without further assumptions.

**Extension, separately labelled: selective-error certification** via a Learn-then-Test binomial procedure at fixed λ, and the gradual-drift experiment for Barber-style weighting.

### 4.3 Decision rules (corrected)

Per sweep point, on mean realized marginal error mass across trials, with one-sided z = 1.645:

1. **Violation** if mean − z·SE > α.
2. **Consistent with control** if mean + z·SE ≤ α + δ_control, with δ_control = 0.005 preregistered. Failure to establish violation is never reported as evidence of control.
3. **Inconclusive** otherwise.

Confirmatory claims on real environments apply Bonferroni across the shift levels within a figure (simultaneous bands are an acceptable substitute if preregistered); exploratory panels are unadjusted and labelled. "Estimated tracks oracle" is an **equivalence claim**: the 90% CI of the paired difference must lie inside [−δ_oracle, +δ_oracle], δ_oracle = 0.005. Trial counts: 200 for pilots, 500 to 1000 for confirmatory points, sized so that SE ≈ 0.001 to 0.002 makes the equivalence margins decidable; at ~3 ms per trial this costs seconds.

### 4.4 Preregistration workflow

Unchanged: register (question, directional predictions, falsifiers, decision rules, config hash), commit, pilot on the placeholder at reduced trials, freeze, run real, verdicts prediction-by-prediction with misses verbatim, amendments append-only. The registration now carries five amendments from Phase 0 and the review response; the trail stays.

---

## Part 5. How to run the runs

**Measured cost.** The corrected four-arm sweep (7 levels × 4 arms × 200 trials, per-x Proposition 2 thresholds and per-trial classifier fits included) runs in well under a minute on one core. The full WP1 programme remains minutes to hours on one machine. Constraints are discipline and provenance, not hardware.

**Two-tier artifacts** unchanged: write-once generation outputs with manifests; freely regenerated analysis keyed by config hash. **Run hygiene** unchanged: one command per experiment, RUNLOG.md, immutable results, seeds in config.

**Execution order, revised to the reviewer's sequencing with the completed items marked:**

1. Freeze the sampling unit, the three endpoints, the commitment-rate definition, and the exact loss. **[done in this document and the registration]**
2. Implement Proposition 2 literally; unit-test the reduction to unweighted CRC at w ≡ 1. **[done, passing]**
3. Replace finite-pool tilted sampling with an exact mechanism; validate closed forms against Monte Carlo. **[done on rung 1; rung 2 lands with the real loader]**
4. Build independent ratio-fit and evaluation splits. **[done]**
5. Run the misspecification machinery on the placeholder, recording the full error-norm battery. **[next]**
6. Gate and run **one** real environment end to end, verifying the entire theorem-to-code path at real-data scale.
7. Preregister the five-environment sweep only after step 6 verifies cleanly.
8. Add group-conditional calibration, the selective-error certification, and the gradual-drift experiment as separately labelled extensions.

## Part 6. Risks

**Apparent oracle failure.** Presumed bug (implementation, sampling, dependence, assumptions), checked against the unit test and audits before any other interpretation; a surviving failure would contradict a published theorem and is treated with that seriousness.

**Gate 3 fails everywhere.** The report becomes a characterisation of when the danger regime exists, plus environment redesign so evidence is genuinely rule-level.

**Misspecification barely moves risk.** Reported as robustness beyond theory, with measured error ranges; WP2's bound targets tighten accordingly.

**ESS collapse.** The certifiable-shift frontier at fixed budget is a finding (Ablation C), not a nuisance.

**Approximation drift.** Rung 3 results quietly cited as if exact. Guard: every figure states its sampling rung.

**Endpoint conflation.** Marginal vs selective, standing risk; the metrics module computes both and every claim names its endpoint.

## Part 7. Definition of done for WP1

1. Five environments gated, reports archived, exclusions reasoned.
2. Shift-response curves per arm vs χ², with the Proposition 3 envelope overlaid and its looseness quantified. Sampling rung stated on the figure.
3. Excess marginal risk vs L1(P₀) estimator error across four degradation axes, with the one-curve verdict and the B·L1 envelope check. The WP2 target.
4. Tilt-location pair of curves.
5. Budget table and group-conditional deferral-cost table.
6. Global-shortcut-vs-Proposition-2 comparison on real environments.
7. Registrations with verdicts, misses verbatim, amendment trail intact.
8. One closing paragraph naming, in measured quantities, the guarantee WP2 must now supply, anchored to the open problem Angelopoulos et al. state at the end of Section 4.1.
