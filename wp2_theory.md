# WP2 theory: the exact decomposition, the first-order law, and the amplification

Started 2026-08-19, after the Phase 0 verdicts and during the Phase 1
mechanism hunt. Every result carries a grade: PROVED,
PROVED-UNDER-CONDITION, or CONJECTURED-WITH-EVIDENCE, and no result may
enter the certificate at a rigor level above its grade. Numerical
verification against the analytic world (registered:
wp2_theory_checks) accompanies each result; a result is not believed
here until its check passes.

## 0. Setting and notation

Cases (s_i, K_i) with evidence score s_i in [0, 1] and commit-error
indicator K_i in {0, 1}; loss L_i(lam) = 1{s_i >= lam} K_i,
non-increasing and right-continuous-from-the-left in lam (the commit
rule is inclusive), bounded by B = 1. For a weight function v >= 0 with
E_P0[v] = 1 define the weighted risk curve

    R_v(lam) = E_P0[ v(X) L(lam) ],

non-increasing, left-continuous, with downward jumps only at score
atoms; J_v(lam) = R_v(lam) - R_v(lam+) is the v-weighted loss mass of
the score atom at lam. The target risk is rho(lam) = R_w(lam) with w
the true (mean-one) likelihood ratio; a candidate weighting is w_hat.
The signed aligned error at threshold lam is

    a(lam) = E_P0[ (w - w_hat) L(lam) ] = R_w(lam) - R_what(lam).

## 1. The exact decomposition (own-threshold coordinate)

**Identity 1 (PROVED, one line).** For every threshold Lam,

    rho(Lam) - alpha = a(Lam) + [ R_what(Lam) - alpha ].

Proof: add and subtract R_what(Lam). QED.

This is F19's coordinate: at the deployed arm's OWN operating threshold
the excess is the aligned error there plus the arm's own crossing
margin, exactly, with no slope, no remainder, and no amplification.
Fresh-seed measurement (wp2p0c part L): reconstruction within 0.0015 in
all 15 cells, pooled slope 1.013 +/- 0.005. The second term is
computable from deployment-visible data (source sample plus w_hat); the
first is not (it contains w), and bounding it observably is the Phase 3
certificate's entire job.

## 2. The intercept lemma (plateau conservatism)

**Lemma 2 (PROVED).** Let Grid = {lam_1 < ... < lam_G} contain 1, and
let the procedure pick Lam = min{ lam in Grid : R(lam) <= alpha_eff }
for any non-increasing curve R and effective level alpha_eff <= alpha
(the B/(n+1) charge makes alpha_eff < alpha; see Lemma 4). If the set is
nonempty then, writing lam_prev for the predecessor of Lam in Grid
(where defined),

    R(Lam) <= alpha_eff <= alpha,
    alpha_eff - R(Lam) <= R(lam_prev) - R(Lam),

i.e. the crossing margin b := R(Lam) - alpha satisfies

    -( DeltaR + (alpha - alpha_eff) ) <= b <= 0,

where DeltaR = R(lam_prev) - R(Lam) is the one-grid-step drop of R at
the crossing: the loss mass of the score plateaus swept in one grid
step, plus nothing else. The bound is attained (b -> -DeltaR) when
alpha_eff sits just above R(lam_prev), i.e. the crossing lands at the
top of a plateau; b -> 0 when it lands at the bottom.

Proof. Minimality of Lam gives R(lam_prev) > alpha_eff >= R(Lam);
subtract. QED.

This formalizes F17/F18's intercept: |b| is bounded by one plateau's
loss mass (plus the grid-quantization mass, which is zero whenever the
grid is finer than the score support, as ledgered). Fresh-seed anchors
reproduce archived b(env) within 0.0004 (P0-1), the granularity
ordering (B2), the continuous-score collapse of b to the bare
finite-sample charge (B3: -0.0015), and the coarse-band value (C:
-0.013).

## 3. The first-order law (lambda-star coordinate)

**Proposition 3 (PROVED-UNDER-CONDITION).** Suppose R_w and R_what are
continuously differentiable on the closed interval I between lam_w and
lam_what (their population crossings at level alpha), with

    R_what'(lam) <= -c < 0  on I,      |R_v'(x) - R_v'(y)| <= Lip |x - y|
                                       on I for v in {w, w_hat}.

Then, with a = a(lam_w) and kappa = R_w'(lam_w) / R_what'(lam_w):

  (i) exact mean-value form:  rho(lam_what) - alpha = [R_w'(xi2) / R_what'(xi1)] * a
      for intermediate points xi1, xi2 in I;
  (ii) first-order law with explicit remainder:

      | rho(lam_what) - alpha - kappa * a |  <=  Lip * (1 + |kappa|) / c^2 * a^2.

Proof. R_what(lam_what) = alpha and R_w(lam_w) = alpha (continuous
crossings). Mean value theorem on R_what over I:
alpha - R_what(lam_w) = R_what'(xi1) (lam_what - lam_w), and the left
side is a, so lam_what - lam_w = a / R_what'(xi1). Mean value on R_w:
rho(lam_what) - alpha = R_w'(xi2) (lam_what - lam_w), giving (i). For
(ii), |xi - lam_w| <= |lam_what - lam_w| <= |a| / c, and

    | R_w'(xi2)/R_what'(xi1) - kappa |
      <= |R_w'(xi2) - R_w'(lam_w)| / c
         + |kappa| |R_what'(xi1) - R_what'(lam_w)| / c
      <= Lip (1 + |kappa|) |a| / c^2 .

Multiply by |a|. QED.

Remarks. (a) kappa is exactly the local slope ratio the A3/Block-C
machinery estimates per cell; PC-2 measured its per-cell predictive
slope at 1.243, R2 0.780, in a world built to break it. (b) In a
factorized world (R_v(lam) = G(lam) H_v) the mean-value ratio is
constant and the remainder vanishes identically: the secant equals the
tangent at every magnitude (verified to 3e-11, P-SW1). (c) The
remainder's SIGN is not universal: the pilot worlds steepen on the
anti-conservative side while the rank-2 quadrature world compresses
(population secant 0.85 at its far cell). The sign is a property of the
local curvature of R_w between the crossings; only the magnitude bound
in (ii) is general. The WP1 pilot's "convex dangerous side" sentence is
hereby DOWNGRADED to an instance-level observation. Grade of (c):
CONJECTURED-WITH-EVIDENCE against further structure; the bound (ii)
itself is proved.

## 4. The finite-sample charge asymmetry (H1b, derived)

**Lemma 4 (PROVED, elementary).** The Prop-2 bound at test weight
w_test selects against the effective level

    alpha_eff(w_test) = alpha - (w_test / (S_w + w_test)) (B - alpha_bar),

in the sense that the selection rule S_L(lam) + w_test B <= alpha (S_w
+ w_test) is equivalent to R_hat(lam) <= alpha - (w_test/(S_w + w_test))
(B - alpha) with R_hat = S_L / S_w the weighted empirical risk. With
mean-one weights S_w ~ n and averaging over test points, w_test in
expectation is E_Q[w_hat] = 1 + chi2(w_hat), so

    alpha - alpha_eff  ~=  (1 + chi2_arm)(B - alpha) / n_cal .

Two arms with different chi2 therefore run at DIFFERENT effective
levels, and the paired difference inherits a deterministic,
budget-dependent, alpha-dependent amplification: the m_grid term of the
switchboard (1.70 at alpha 0.02, n_cal 250; within 0.02 of 1 by n_cal
10000, P-SW2 verdict). For the oracle arm 1 + chi2 is exactly E_P0[w^2];
for a tempered arm it is strictly smaller, so the oracle is charged
MORE, which is conservative for the oracle and anti-conservative for
the paired difference read as an amplification.

## 5. The discrete-crossing amplification (the m mechanism)

Setting of Lemma 2 with two arms. Write Lam_o, Lam_e for the two
population grid crossings and G(lam) for the common loss-tail factor in
a factorized world (general statement below). Then, combining
Identities 1 and Lemma 2:

    excess_e - excess_o = a(Lam_e) + b_e - b_o,
    a(Lam_e) = a(lam*) * [ R-tail ratio at the two crossings ],

and on plateaued curves the tail ratio G(Lam_e)/G(Lam_o) is bounded
below by 1 on the anti-conservative side and can be as large as the
ratio across one plateau step. The lambda*-referenced amplification is
therefore

    m(cell) = G(Lam_e)/G(Lam_o) + (b_e - b_o) / (kappa a),

a quantity that (i) survives n_cal -> infinity, (ii) moves with alpha
through where the crossing lands on the plateau structure, (iii)
depends on the environment's local score geometry, and (iv) vanishes in
the own-threshold coordinate.

VERDICTS (wp2qsw, wp2mc). The plateau-ownership prediction CONFIRMED
(risky): at n_cal 100000, alpha 0.05, m runs 2.16 (K=8) down to 1.01
(continuous), and the arithmetic also COMPRESSES (0.78 at K=8, alpha
0.10) because the sign is set by where alpha's crossing lands, which
retroactively explains B2's non-monotone granularity result. On the
gated tier the identity-derived per-cell m reproduces measured m at
slopes 0.97 to 1.12 with R2 0.79 to 0.96 wherever plateau spread
exists, reproduces the environment ordering at alpha 0.05, and owns
the tickets anomaly (m_derived 1.68 at 0.05 vs 1.07 at 0.10).

GRADE, stated precisely: m is DERIVED in the large-budget limit
(deterministic crossing arithmetic, computable from calibration draws)
and BOUNDED at finite budgets: empirical-crossing noise SOFTENS the
plateau arithmetic from below at intermediate budgets (measured m
1.084 vs derived 1.513 at K=16, n_cal 10000, converging by 100000),
while the Lemma-4 charge adds from above at small budgets. The exact
finite-budget composition is an open refinement, CONJECTURED-WITH-
EVIDENCE: max(1, m_derived) + charge bounds m at the measured grids.
None of this is load-bearing for the certificate (Section 6).

## 6. What enters the certificate

From Identity 1, the deployable form needs an observable upper bound on
a(Lam_e) only; b_e is measured directly from the calibration sample and
is nonpositive by Lemma 2. The lambda*-coordinate (kappa, m) machinery
is explanatory, not load-bearing, for the certificate; it becomes
load-bearing only if a(Lam_e) must be forecast BEFORE the threshold is
computed, in which case m_max(alpha, n_cal) from Lemmas 4 and Section 5
bounds the translation. Phase 3 builds on this section.
