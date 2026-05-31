# 02 — Causal forests and honest splitting

**Anchor:** Athey & Wager (2019), *Estimating Treatment Effects with Causal Forests: An
Application*, Observational Studies 5(2):37–51 (arXiv:1902.07409); building on Wager & Athey
(2018), *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests*, JASA.

## From ATE to CATE
DML (note 01) gives one number, the average effect. But the +$1,794 average hides large
variation: who gains the most from training? The **Conditional Average Treatment Effect**
`τ(x) = E[Y(1) − Y(0) | X = x]` is the per-individual quantity we want, and a **causal forest**
estimates it nonparametrically. A causal forest is a random forest whose splits are chosen to
maximise *heterogeneity in the treatment effect* rather than to predict the outcome — leaves
group together individuals whose effects are similar, and the local treatment effect within a
leaf's neighbourhood becomes the CATE estimate.

## Honest splitting — the key idea
A standard regression tree uses the same observations both to *choose where to split* and to
*estimate the value in each leaf*. That double-use makes leaf estimates biased and invalidates
classical confidence intervals — the tree has already "seen" the responses it then averages.
**Honest** trees split the training subsample in two:

- the **splitting** half decides the tree's structure (where the cuts go);
- the **estimating** half fills in each leaf's treatment effect, on data the structure never
  touched.

Because the leaf values are computed on observations independent of the split decisions, the
per-leaf estimates are approximately unbiased and admit valid inference. You pay for it in
variance (each half is smaller), but you buy back honest standard errors — a worthwhile trade
when the whole point is a trustworthy, individualised effect.

## Sub-sampling, not bagging
Classic random forests bootstrap (sample *with* replacement). Causal forests instead draw
**sub-samples without replacement** (a fraction of the rows per tree). This matters for theory:
sub-sampling lets the forest be analysed as a **U-statistic**, which yields asymptotic normality
and a usable variance estimator (the infinitesimal jackknife). Bagging's with-replacement draws
break that clean limiting distribution. So the sub-sampling choice is not a tuning detail — it is
what makes *local* inference (a CI around each individual's τ(x)) well-defined.

## How it shows up in this repo
`CausalForestDML` fits the honest forest on the NSW data (small but unconfounded). Every
prediction carries a 95% interval via `effect_interval` / bootstrap, and `safe_predict()` logs
input stats so out-of-support extrapolation is visible. The heterogeneity tables (age the
strongest moderator, CATE rising from $776 to $3,231 across age quartiles) and the policy-targeting
curve are all downstream of these per-individual estimates.
