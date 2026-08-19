# WP2 notes: the first-order theory the empirics now demand

Working notes, started 2026-08-18 while the aligned-error collapse
(fd2279c8f7dc2df6) is fresh. Not a paper section; a seed for one.
All numbers cited are placeholder-pilot tier.

## 1. The perturbation identity behind the collapse

Notation. Losses L_i(lam) = 1{s_i >= lam and wrong_i}, non-increasing in
lam, bounded by B = 1. True ratio w, estimate w_hat, both normalized to
mean 1 under the source P0. Target risk rho(lam) = E_Q[L(lam)] =
E_P0[w L(lam)]. Ignore the finite-sample B-term throughout (it is
O(1/n_cal) and conservative).

The population estimating equation of the weighted procedure picks the
threshold where weighted source risk crosses alpha:

    under w:      E_P0[w     L(lam_star)] = alpha        (oracle threshold)
    under w_hat:  E_P0[w_hat L(lam_hat )] = alpha        (estimated threshold)

Subtract, evaluate both at lam_star, and expand to first order in
(lam_hat - lam_star):

    E_P0[w_hat L]'(lam_star) * (lam_hat - lam_star)  ~=  a,

where a = E_P0[(w - w_hat) L(lam_star)] is exactly the registered aligned
error. Realized excess risk is then

    rho(lam_hat) - alpha  ~=  rho'(lam_star) * (lam_hat - lam_star)
                          ~=  a * [ rho'(lam_star) / E_P0[w_hat L]'(lam_star) ].

So to first order,

    EXCESS  ~=  kappa * a,   kappa = (E_P0[w L])' / (E_P0[w_hat L])' at lam_star,

and kappa = 1 whenever the estimated weighting places the same local loss
mass near the threshold as the true one. The same monotone loss enters the
estimating equation and the evaluation; that shared structure is why the
coefficient is order one rather than problem-dependent.

## 2. What the data say about the identity

Fitting a line through the origin to the 58 collapse cells:

    |a| <= 0.010 (28 cells):  slope 1.19
    |a| <= 0.020 (38 cells):  slope 1.12
    |a| <= 0.025 (40 cells):  slope 1.11

Slope approximately 1, as the identity predicts, with mild amplification
(kappa slightly above 1: under-weighting thins the loss mass w_hat sees
near the threshold, shrinking the denominator derivative first).

The first-order regime has a measured boundary. The two preregistered
residual failures, temper gamma=0 at beta=0.75 (residual -0.0107) and
temper gamma=0.25 at beta=1.25 (+0.0107), sit at nearly identical aligned
error (a ~ 0.053) with different excess (0.094 vs 0.115). Beyond
|a| ~ 0.04 the excess-to-a ratio grows through 1.5 to 2.7 on the
anti-conservative side: kappa is no longer locally constant because badly
under-weighted estimating equations cross alpha in a region where their
local slope has collapsed. On the conservative far side the ratio
compresses to ~0.76: over-weighting pushes the threshold into territory
where rho flattens (risk cannot fall below the floor set by deferral), so
the response saturates. The second-order theory has a sign-asymmetric
shape: convex on the dangerous side, concave on the safe side. Both
asymptotes are visible in the pilot figure.

## 3. The two jobs WP2 actually has

Job 1, the theorem: make Section 1 rigorous. Prove that for bounded
monotone losses, excess target risk of Proposition 2 run with w_hat equals
kappa * a + O(a^2) with an explicit kappa and an explicit second-order
remainder, presumably under a local density condition on the loss curve
near lam_star (rho' bounded away from 0, Lipschitz). The empirical
boundary above says the remainder must be allowed to be sign-asymmetric.

Job 2, the certificate, and it is the harder half: a is ORACLE-REFERENCED.
It is computed from the true ratio and the oracle threshold, neither of
which deployment has. An explanation is not a certificate. Decompose

    a = E_Q[L(lam_star)] - E_P0[w_hat L(lam_star)],

first term unobservable (target labels), second term computable. The
natural observable surrogate: the density-ratio classifier's held-out
calibration error, LOCALIZED to the loss region {s >= lam_hat, wrong-prone}.
Since w - w_hat = (odds - odds_hat) at fixed x, a is the classifier's
signed calibration bias integrated against the loss indicator; a held-out
reliability analysis restricted to cases near and above the operating
threshold upper-bounds |a| in terms of measurable miscalibration mass.
Candidate certificate shape:

    excess  <=  kappa_max * CalErr_loc(w_hat)  +  second-order term,

with CalErr_loc estimated on held-out source data plus unlabelled target
covariates only. The B*L1 envelope (valid 58/58 but loose by >= 5x, F4) is
the degenerate, non-localized version of this: L1 charges for weight error
everywhere, while only error mass inside the loss region matters, and only
its signed component moves the threshold.

## 4. Empirical checks WP2 should commission from WP1

1. Temper family at two more beta levels: does kappa near the origin stay
   at ~1.1 and does the |a| ~ 0.04 boundary move with shift level?
   (Registered addendum, cheap.)
2. Realized (lam_hat - lam_star) distribution per cell: the identity
   predicts it, and it is already computable from stored per-trial data
   paths with a small runner change. Verifies the mechanism, not just the
   endpoint relation.
3. A localization check: recompute a with L replaced by 1{s >= lam} alone
   (commit indicator, no wrong) and by 1{wrong} alone (no commitment):
   the collapse should degrade under both, confirming it is the loss
   region, not either margin, that matters.
4. On the real environment: whether kappa stays near 1 when the loss curve
   rho is not synth2-smooth. This is the load-bearing external validity
   question and belongs in the environment-1 registration.

## 5. Boundary conditions to carry into the theorem statement

- Two shift levels only so far; mild beta-dependence already visible at
  the far anti-conservative end. The theorem's constants may depend on
  chi2; the empirics so far cannot distinguish kappa(chi2) from a pure
  second-order-in-a effect.
- Everything is placeholder-tier: one tilt direction, Gaussian covariates,
  a smooth strictly-decreasing loss curve. Kinked or plateaued loss curves
  (real environments with rule-level evidence scores produce plateaus,
  since s takes finitely many values) will stress the rho' condition;
  the grid-based inf in the code already handles plateaus mechanically,
  but the first-order identity needs rho' > 0 in a neighborhood, which
  plateaus violate. Expect the real-environment collapse to be piecewise.
