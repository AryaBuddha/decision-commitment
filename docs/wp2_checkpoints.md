# WP2 human checkpoints

The handover names five stop-and-surface points: after Phase 0 verdicts;
whenever theory and empirics contradict; before freezing the certificate
(Phase 3 exit); before building environment 7 (Phase 5 entry); after
Phase 5 verdicts. The operator instructed this session to run WP2 to
completion in one pass, so each checkpoint is written here as a decision
packet at the moment it would have paused, with the decision taken and
its reversibility stated. Every packet cites registrations and artifacts;
nothing here is evidence, it is navigation.

## Checkpoint log

### Checkpoint 1: after Phase 0 verdicts (2026-08-19)

State. Three registered evidence-tier experiments ran on fresh seeds and
carry filed verdicts: the confirmatory battery (wp2p0_ec15383b39b52206),
the budget grid (wp2p0b_16996ec167284e46), and the two closures
(wp2p0c_a801f76c2d55fb4d). The law's confirmed form, with fresh-data
verdicts attached:

    excess = m(alpha; env, n_cal) * kappa_pred * a  +  b(env)     [lambda* coordinate]
    excess = a_own + b_own                                        [own-threshold coordinate, EXACT]

What survived confirmation intact: the b anchors (4/4 within 0.0004);
oracle safety (384/384 cells, down to alpha 0.02); every registered m
window (m(0.10) in [1.07, 1.23] across four environments, spread 0.158
under the 0.25 margin; claims m(0.05) = 1.449); per-cell kappa_pred
structure in both wide-kappa worlds at every measurable alpha; the
collapse at rank rho 0.80 to 0.996 everywhere testable; and the
own-threshold exactness (slope 1.013 +/- 0.005, no amplification). The
B4 regularization diagnosis survived its risky test.

What the confirmations CORRECTED: (1) m is not universal-monotone in
alpha: tickets rises to 1.838 at 0.05 then falls to 1.261 at 0.02, and
the 0.05 family spread is 0.69, so family invariance is an alpha-0.10
phenomenon. (2) m falls with budget but plateaus ABOVE 1 at an
alpha-dependent limit; H2 owns only the budget-dependent part. (3) At
alpha 0.02 the absolute |a| guard starves measurement in three of four
environments (|a| scales with alpha): the law's extreme-alpha domain is
bounded by measurability, not by violation.

The mechanism hunt (Phase 1) then reframed the problem twice, both by
registered experiment: the smooth analytic switchboard proved H1's
B-term half is a large derivable small-budget amplifier, H2's noise term
dies with budget (and partially cancels the B-term at small budgets),
H3's per-covariate gap is zero, H4 was already structurally absent, and
the smooth world HAS NO PLATEAU. The registered H4 split on fresh gated
data killed H4 independently (8/8). Score discreteness became the only
candidate standing, with a closed-form mechanism (the two arms cross on
different plateaus; m carries G(lam_e)/G(lam*)) registered for the
quantized-world derivation test (wp2_quantized_switchboard).

Decision taken (operator authorized end-to-end execution): proceed to
the quantized derivation test; if it confirms, Phase 2 writes m as a
DERIVED quantity and Phase 3 builds the certificate in the own-threshold
coordinate where no m appears (F19), using the plateau arithmetic only
as a cross-check. Reversible: no prose has been written, no certificate
frozen; every verdict and miss is in the registrations.

### Checkpoint 2: theory and empirics in tension (2026-08-19)

Tension found while drafting wp2_theory.md: the WP1 pilot characterized
the second-order remainder as universally sign-asymmetric ("convex
dangerous side, saturating safe side"). The rank-2 quadrature world is
a counterexample: its population secant sits BELOW kappa on the far
anti-conservative side (0.85 at the gamma = 0 cell, alpha 0.05), i.e.
compression where the pilot saw steepening. Resolution taken: the
remainder bound |R| <= Lip (1 + |kappa|) / c^2 * a^2 is proved and
verified (wp2thk 16/16); the SIGN claim is downgraded in wp2_theory.md
to an instance-level observation, and the fresh battery's P0-7 misses
(negative residuals at far cells on spike/tickets) side with the
compression instances. No registered claim depended on the universal
sign; the downgrade is a documentation correction, not a retraction.

### Checkpoint 3: before freezing the certificate (2026-08-19)

Decision packet. The certificate form was frozen by the commit carrying
registrations/wp2_certificate.json BEFORE its validation ran, in the
own-rule coordinate (no kappa, no m, no lambda* in the bound), on the
strength of: Identity 1 (algebraic), closure L (slope 1.013 +/- 0.005,
15/15 reconstruction), and P-MC3 (identity within 2 SE in 122/128
tempered-arm cells). The alternative (the handover's m_max * kappa_pred
* A_hat shape) was rejected because m is now understood as a
lambda*-coordinate artifact with derivable geometry (Phase 1), and a
bound that never references lambda* cannot inherit that artifact. Known
weaknesses deliberately left in for Phase 4 to attack: source-empty
audit bins contribute zero to CalErr_loc while carrying target mass;
within-bin cancellation; the audit model's own regularization default;
post-audit mix drift. Reversible: Phase 4's registered loop may revise
and re-attack.

### Checkpoint 4: before building environment 7 (2026-08-19)

Decision packet, written at build time. Preconditions verified: the
certificate was frozen by commit (checkpoint 3) and then survived or
was revised through two registered red-team rounds (one drift breach
converted to a monitored validity window; binning pinned at the
conservative end; final form v2 in wp2_redteam2's phase4_exit).
Environment 7 (returns) was designed only after the freeze commit, with
three disclosed design iterations, and passed all five gates on the
frozen design. Its structure deliberately stresses the certificate's
audited weak points: a gold interaction outside the audit model's
class, and a blind driver with only a weak proxy. The prospective
protocol is two-stage: stage A computes the certificate envelope from
deployment-visible data only and the envelope numbers are registered
as predictions; stage B sweeps realized risk. Reversible: nothing about
stage B can retroactively change stage A's registered numbers.

### Checkpoint 5: after Phase 5 verdicts (2026-08-19)

WP2 is complete against its definition of done.

1. Phase 0 fresh-data verdicts: filed (wp2_phase0_battery, _budget,
   _closures). The law's confirmed form: excess = a_own + b_own exactly
   at the arm's own threshold; in lambda* coordinates excess =
   m(alpha; env, n_cal) * kappa_pred * a + b(env), with every anchor
   and window hit and the structural misses (m non-monotone in alpha on
   tickets; family invariance an alpha-0.10 phenomenon) resolved by the
   Phase 1 mechanism.
2. The m(alpha) mechanism: identified as a stated combination on both
   tiers (B/(n+1) charge asymmetry, derived; crossing noise, dies at
   root-n and softens; discrete-crossing plateau arithmetic, owns the
   budget-independent plateau and the environment ordering). H3 and H4
   dead by registered discriminators. The analytic-world decomposition
   is on file (wp2sw, wp2qsw artifacts).
3. wp2_theory.md: graded results, all numerically verified
   (wp2_theory_checks: 24/24, 16/16, 18/18).
4. The certificate: frozen, validated over 674 archived cells (zero
   dangerous-cell median failures; mean conservatism +0.0464; the
   registered anti-vacuity clause missed at 4.1x median looseness,
   filed verbatim), survived two registered red-team rounds with one
   breach converted to a monitored freshness window and one silent
   default pinned (final form v2).
5. The capstone: holdout environment built after the freeze, envelope
   registered from deployment-visible data before the sweep, 12/12
   covered at mean price +0.0562, the danger-location risky clause
   missed verbatim.
6. Registrations, verdicts, misses verbatim, RUNLOG current, ledger
   rows for every new knob.

Paper prose remained untouched throughout and may now resume per the
handover; reconciling paper/draft.md with F17 through F23 is the next
block's first task. Open threads left deliberately on file: the
finite-budget composition of noise-softening and plateau arithmetic
(P-Q2's refinement); the properly-posed FD-instrument split; the
safe-side bound dips (no risk content, disclosed anatomy); Phase 6
extensions (group-conditional two-step, Learn-then-Test selective
certification, gradual drift).

### Checkpoint 6: the two corrections before prose (2026-08-19)

Operator review after checkpoint 5 found two gaps, both accepted.

First, essential: the capstone had covered a deployment whose excess
was negative everywhere, validating the workflow but not the
protection. Fixed by the stressed capstone (wp2_prospective_stress):
the WP1 degradation axes instantiated on the same holdout, envelope
registered before the sweep, and the result is the claim the paper
actually needs: 12/12 covered while six no-correction deployments
genuinely failed their certified level, envelope approached (0.37) and
not crossed. A disclosed probe filed on the way: the fit-based
degradations are heavily defended on this holdout (F14 prospectively),
so no-correction is the honest stress.

Second, cheap: attack the looseness instead of accepting it.
Certificate v3 floors the certified excess at zero (never certify
below alpha; the definitional fix that converts the 26 safe-side
technical misses into trivial coverage), the 674-cell validation is
recomputed under the final form with per-draw values stored (the P-C4
instrumentation lesson), and a registered slack-attribution analysis
decomposes the bound term by term, including the structural question
(the own-coordinate certificate carries no worst-case translation
constant; the F19 dividend was banked at the Phase 3 freeze) and a
sample-size lever probe. Verdicts below in the registrations.

The freeze on paper prose lifts after these verdicts are filed.
