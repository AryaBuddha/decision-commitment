# Environment 1 (claims triage): validation gate report

Frozen contract: registrations/env_claims.json. Three gate runs, all
archived; the two failures are published per the gate 3 rule.

## Run 1: gates_claims_36bc9577ed644af7, FAILED (G3)

All manifest features visible to induction. Best unsigned AUC lift of
wrong ~ (s, x_j) over wrong ~ s was +0.0072 (severity), threshold 0.02.
Depth-7 rules absorbed nearly all covariate dependence of wrongness; the
environment could not exhibit the evidence-blind failure. Amendment 1:
induction view excludes inconsistency (unlogged in demonstration records).

## Run 2: gates_claims_2ce3e00aea6b6b19, FAILED (G3)

Unsigned lift for the now-blind inconsistency was only +0.0049, and an
offline design check showed unweighted CRC SELF-CORRECTING under the
inconsistency tilt (risk falling from 0.104 to 0.091 at beta 5). Two
causes, both instructive. First, auto_flag leaked inconsistency into the
induction view, routing high-inconsistency mass into low-purity, low-s
leaves that defer by themselves: synth v1's disease through a realistic
proxy mechanism. Second, the unsigned audit cancels sign-opposed effects:
inconsistency raises wrongness in APPROVE-routed cases (corr +0.29) and
lowers it in INVESTIGATE-routed ones (corr -0.45); on identical data the
unsigned lift was +0.0049 and the decision-conditional lift +0.0627.
Amendment 2: proxies decoupled, gold strengthened on the blind feature,
G3 moved to the decision-conditional form (threshold unchanged).

## Run 3: gates_claims_5459d3b5a7b3c1a1, ALL PASS (2026-08-18)

| Gate | Result | Detail |
|------|--------|--------|
| G1 solvability | PASS | gold deterministic 1.0000, well defined 1.0000, n = 50000 |
| G2 rung-2 fidelity | PASS | all three tilt features at beta = 6; worst moment diff 0.13 of tolerance |
| G3 evidence blindness | PASS | decision-conditional lift: inconsistency +0.0859, severity +0.0372, doc_completeness +0.0253; s-only AUC 0.7604, full-X 0.8016 |
| G4 region mass | PASS | region-1 share 0.0531 at beta 0 rising to 0.2049 at beta 6 (floor 0.05) |
| G5 evidence spread | PASS | 46 distinct lambda-grid cells (floor 20) |

The environment's character, measured offline post-redesign: unweighted
CRC risk rises 0.095 to 0.128 across beta 0 to 6 (chi2 0 to 3.47), a
milder decay than the placeholder because proxy self-correction is still
partially operative. That is the environment being honest, not weak:
inconsistency is blind to the rules, severity is rule-visible, and the
tilt-location ablation gets both from one environment.
