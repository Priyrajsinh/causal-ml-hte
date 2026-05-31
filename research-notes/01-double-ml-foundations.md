# 01 — Double / Debiased Machine Learning: foundations

**Anchor:** Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018),
*Double/Debiased Machine Learning for Treatment and Structural Parameters*,
The Econometrics Journal 21(1):C1–C68. arXiv:1608.00060.

## The problem DML solves
We want one low-dimensional causal parameter — here the average treatment effect θ of job
training on 1978 earnings — in the presence of high-dimensional nuisance functions: the outcome
model `g(X) = E[Y | X]` and the propensity / treatment model `m(X) = E[T | X]`. We would love to
estimate `g` and `m` with flexible machine learning (they can be genuinely nonlinear), but naively
plugging ML estimates into a causal moment condition imports the ML estimators' **regularisation
bias** directly into θ. Because ML methods trade variance for bias (shrinkage, early stopping,
tree depth limits), that bias does not vanish fast enough, and θ inherits it.

## The two ideas that fix it

**1. Neyman-orthogonal moments (the Robinson decomposition).** Instead of regressing `Y` on `T`,
DML works with *residuals*. Partial out the covariates from both sides:

- `Ỹ = Y − g(X)` — the part of the outcome the covariates cannot explain;
- `T̃ = T − m(X)` — the part of treatment assignment the covariates cannot explain;
- then `θ` is the coefficient in `Ỹ = θ·T̃ + ε`.

This is Robinson's (1988) partially-linear-model trick. Its decisive property is **Neyman
orthogonality**: the moment condition's derivative with respect to the nuisance functions is zero
at the truth. So a *first-order* error in `g` or `m` only produces a *second-order* error in θ —
the product of two small estimation errors. ML estimators that each converge slowly (slower than
the √n parametric rate) can still deliver a √n-consistent, asymptotically normal θ, as long as the
product of their rates beats n^(−1/2).

**2. Cross-fitting.** Orthogonality removes the bias from *estimating* the nuisance; cross-fitting
removes the bias from *overfitting* it. If you fit `g` and `m` on the same rows you then evaluate
residuals on, the estimator can fit its own noise, creating a spurious correlation between `Ỹ` and
`T̃`. DML avoids this by sample-splitting: partition the data into K folds, fit the nuisance models
on K−1 folds, and compute residuals on the held-out fold — then rotate and average. Single-fold
sample-splitting wastes half the data; cross-fitting recovers full efficiency while keeping the
"don't predict on your training rows" guarantee. This is exactly why **rule C35** pins `cv=5` and
forbids `cv=1` / `cv=None`: cv=5 *is* the Robinson guarantee in code.

## How it shows up in this repo
`LinearDML(model_y=LGBMRegressor(...), model_t=LGBMClassifier(...), cv=5, random_state=seed)`.
On the observational CPS construction, naive OLS reports −$8,497 (pure selection bias); DML on the
RCT data lands at +$1,834, indistinguishable from the +$1,794 ground truth. The method recovers
the truth precisely because orthogonality + cross-fitting strip the confounding that OLS cannot.
