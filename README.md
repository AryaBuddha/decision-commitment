# Commitment under Distribution Shift: WP1 scaffold

Runnable skeleton for WP1 (shift-response curves), built against a synthetic
placeholder environment so the pipeline and the metric definitions are settled
before instance generation cost is incurred.

```
python experiments/wp1_shift_response/run.py
python experiments/wp1_shift_response/plot.py artifacts/wp1_<hash>
```

## Methodology

The programme is a **controlled-violation study**. For each guarantee under
examination:

1. Take a procedure proven correct under assumption A.
2. Deploy it where A is violated by a known, dialled amount δ.
3. Measure realized against certified performance as a function of δ.
4. Report the decay curve, then try to certify what survives.

The platform earns its place because δ is *set*, not inferred. Six design
properties each remove a confound that a found dataset cannot:

| Property | Confound removed |
| --- | --- |
| Shift is a parameter | true likelihood ratio unknown |
| Ground truth recomputed in code | annotation cost, label noise |
| Oracle solver ≥ 0.98 | environment unsolvable, not method failing |
| Decisions read from call ledger | stated-reasoning measurement artifact |
| Abstention scored | selective prediction has no real target |
| Preregistration enforced by runner | post-hoc threshold movement |

The oracle-ratio arm is the load-bearing one. It exists nowhere in the
empirical literature on covariate shift, because no natural dataset pair
supplies it. It is what separates "the estimator is bad" from "the weighted
procedure is wrong for risk control."

## What λ, the loss, and α are here

`λ` is a commitment threshold: commit iff evidence score `s ≥ λ`. The loss is

    L_i(λ) = 1{ s_i ≥ λ AND case i would be answered wrong }

which is non-increasing in λ and bounded by 1, satisfying Angelopoulos et al.
(2022) Theorem 1. Deferral rate is reported separately as the efficiency cost,
playing the role prediction-set size plays in ordinary conformal.

This controls the **marginal** commit-error rate, not the **selective** rate
`E[wrong | commit]`, which is a ratio and needs Learn-then-Test machinery. Both
are reported. Do not present one as the other.

## Findings from the pilot run

α = 0.10, 200 trials per point, n_cal = n_eval = 1000.

- Unweighted CRC decays from 0.100 to **0.314**, a 3.1× overshoot, once shift
  falls on an evidence-blind feature.
- Both weighted arms hold at ≈ 0.099 across the whole sweep. The weighted
  risk-control procedure is a conjecture, not a theorem, so this is a
  substantive result and not a sanity check.
- Effective sample size falls 1000 → 134. This is the price, and it bounds how
  much shift is certifiable at fixed calibration budget.
- **Region 1 sits at 0.43 with zero shift.** Marginal control transfers to
  groups not at all, and this failure needs no shift to appear.
- Estimated weights track oracle almost exactly, because logistic regression is
  correctly specified for an exponential tilt. **Q1 therefore has no bite until
  the estimator is misspecified.** This is the first thing to fix.

## Next three steps

1. **Misspecify the estimator.** Hide the tilted dimension from
   `estimate_weights`, or raise dimension until the classifier struggles.
   Sweep it the way Sachdeva et al. sweep "DM quality." Without this, WP2 has
   no phenomenon to explain.
2. **Run the tilt-location ablation.** Set `tilt_direction` to `[1,0,0,0,0]`
   (visible driver) and compare. Decay depends on *where* shift falls relative
   to the evidence signal, not on magnitude alone. No found dataset can
   separate these.
3. **Add group-conditional calibration** and measure what it costs in deferral.
   Exact conditional coverage is provably impossible; group-conditional is the
   right target. See Barber et al., *The limits of distribution-free
   conditional predictive inference*.

## Registration

`registrations/wp1_shift_response.json` carries predictions, falsifiers, and a
config hash. `run.py` refuses to execute on a hash mismatch. Two amendments are
already on file from the pilot, including one where the pilot showed the
registered decision rule was itself wrong. That trail is the point; do not
clean it up.
