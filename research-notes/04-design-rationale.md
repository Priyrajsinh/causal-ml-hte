# 04 — Design rationale for this project

Notes on the engineering choices, and the alternatives that were considered and rejected.

## Why LightGBM for the nuisance models
DML needs two nuisance estimators: `model_y` (outcome regression) and `model_t` (treatment /
propensity classification). LightGBM was chosen because:

- **Small-N friendly and fast.** With only 445 NSW rows, training is near-instant, so the cv=5
  cross-fitting loop (which refits the nuisances K times) stays cheap.
- **No feature scaling required.** Gradient-boosted trees are invariant to monotone feature
  transforms, so the raw covariates (earnings in dollars, age in years, binary indicators) go in
  unscaled — one fewer preprocessing step to leak through.
- **Handles mixed types and nonlinearity** out of the box, which is the whole point of using ML
  nuisances: we want `g(X)` and `m(X)` flexible while DML's orthogonality protects θ.

*Rejected:* a linear/logistic nuisance would be fast too, but defeats the purpose — if the
nuisances were linear we would not need DML's orthogonality at all. A deep net is overkill and
unstable at n=445.

## Why `inference="bootstrap"` over `"blb"`
For the per-individual CIs we use ordinary **bootstrap** resampling rather than the Bag of Little
Bootstraps (`blb`). BLB is an efficiency optimisation designed for *very large* samples where
resampling the full dataset many times is computationally prohibitive. At n=445 the full bootstrap
is cheap, and it is the more standard, more transparent choice for a small sample — easier to
reason about and to defend in a model card. Speed is not the binding constraint here; clarity and
convention are.

## Why no train/test split (rule C33)
It is tempting to hold out a test set, but for this estimator that is actively harmful:

- **DML already cross-fits.** The `cv=5` loop *is* the out-of-sample validation — nuisance models
  never predict on the rows used to fit them. Adding an outer test split would double-count the
  same idea while throwing away data the inner cross-fitting needs.
- **The sample is tiny.** Carving 20% off 445 rows shrinks the effective sample below what the
  bootstrap CIs need to be stable, widening intervals for no inferential gain.
- **The validation target is external.** Correctness here is graded against the *experimental
  ground truth* (+$1,794, rule C37), not against held-out predictive accuracy. The honest causal
  question — "does the estimator recover the RCT ATE?" — is answered by the RCT, not by a split.

So the pipeline fits on the **full** sample, validates via cross-fitting + the bootstrap + the
ground-truth coverage assertion, and never shuffles a test set.

## Other choices, briefly
- **pandera before estimation** (C34): catch dtype/range errors at the source, never after a
  silent miscast corrupts an estimate.
- **`safe_predict()` wrapper** (C36): NaN/inf/shape guard + input logging on every CATE call, so
  out-of-support extrapolation is visible rather than silent.
- **Results frozen to `reports/results.json`**: the UI and docs read computed numbers from one
  artifact, so the README table, MODEL_CARD, and dashboards never drift apart.
