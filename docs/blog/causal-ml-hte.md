---
title: "Causal ML in plain English: when naive regression lies, and what to do about it"
author: Priyrajsinh Parmar
date: 2026-05-31
tags: [causal-ml, double-ml, econml, lalonde, eu-ai-act, shap]
---

# Causal ML in plain English: when naive regression lies, and what to do about it

Run an ordinary regression on observational labour-market data and you can reach a
disturbing conclusion: a government job-training programme *lowers* the earnings of
the people who took it. In my own reproduction the naive number comes out around
**−$8,500** — as if training actively destroyed eight thousand dollars of income per
participant. It is the kind of result that, taken at face value, kills a programme.

Now run a randomised controlled trial of the *same* job-training intervention and
you get the opposite sign: training **raises** 1978 earnings by about **+$1,800**.
Two analyses, same treatment, a ten-thousand-dollar disagreement about whether it
helps or harms.

Both numbers are real. Only one is causal. The gap between them is the entire reason
the field of causal machine learning exists — and Double Machine Learning (DML) is
the modern toolbox that closes it. This post walks through how, using the canonical
LaLonde dataset and the code in [this repository](https://github.com/Priyrajsinh/causal-ml-hte).

## The LaLonde dataset and the bias story

Robert LaLonde's 1986 paper is the canonical stress-test for causal estimators. He
took the **National Supported Work (NSW)** demonstration — a genuine randomised
experiment where disadvantaged workers were *randomly* assigned to a subsidised
job-training programme — and asked a pointed question: if you throw away the control
group and instead compare the treated workers to a non-experimental comparison group
drawn from national survey data (the **Current Population Survey, CPS**), can the
econometric methods of the day recover the experimental answer?

They could not. The treated group in NSW were unemployed and low-income by
construction; the CPS comparison group were, on average, employed and far
better-off. Comparing their later earnings conflates the *effect of training* with
the *pre-existing gap between the groups* — textbook selection bias. Dehejia and
Wahba (1999) later showed that propensity-score methods could partly rescue the
observational estimate, reviving the dataset as a benchmark. It remains the standard
proving ground: any honest estimator must recover the experimental **+$1,794** ATE
from the confounded construction, or admit it cannot.

## Double ML in plain English

Double Machine Learning (Chernozhukov et al., 2018) starts from a deceptively simple
idea due to Robinson (1988). Suppose the outcome *Y* (1978 earnings) depends on a
treatment *T* (training, 0/1) and a pile of confounders *X* (age, education, prior
earnings, race, marital status). Instead of regressing *Y* on *T* and *X* all at once
— which lets a flexible model quietly absorb the treatment effect into the
confounders — you do it in two stages.

First, predict the **outcome** from the confounders alone: a model *ŷ = E[Y | X]*.
Then predict the **treatment** from the same confounders: a propensity model
*t̂ = E[T | X]*. Subtract both predictions to get residuals — the part of *Y* and the
part of *T* that the confounders *cannot* explain. Finally, regress the outcome
residual on the treatment residual. That last, simple regression is your treatment
effect.

This is "double" because there are **two nuisance models**, one per side of the
problem. It is "debiased" because of a near-magical robustness property: if *either*
the outcome model *or* the propensity model is correctly specified, the
first-order bias term cancels. You get two shots at getting it right. In this project
both nuisance models are gradient-boosted trees (LightGBM) — flexible enough to soak
up nonlinear confounding without me hand-specifying functional forms.

## Cross-fitting

There is a catch. If you fit the nuisance models and estimate the treatment effect on
the *same* rows, the models overfit those rows, the residuals get artificially small,
and that overfitting leaks straight into a biased ATE. The fix is **cross-fitting**:
split the data into *K* folds (here `cv=5`), and for each fold predict its residuals
using nuisance models trained *only on the other folds*. No row is ever scored by a
model that has seen it. The folds rotate until every row has an out-of-fold residual,
and the treatment effect is estimated on those.

Cross-fitting is anti-leakage made principled — it is the formal guarantee that
overfitting in the nuisance stage cannot poison the causal estimate. It is also why,
in this repo, the causal model is fit on the **full** NSW sample with no separate
"test split": the internal *K*-fold rotation *is* the validation strategy, and
carving off a holdout would only shrink an already-small 445-row sample.

## Causal forests

DML gives you one number — the *average* effect. But averages hide people. Does
training help everyone equally, or are there subgroups it helps a lot and subgroups it
barely touches? That is the **heterogeneous treatment effect (HTE)** question, and the
**Causal Forest** (Athey & Wager, 2018; here via EconML's `CausalForestDML`) is built
to answer it.

A causal forest is a random forest whose splits are chosen to maximise *treatment-effect
heterogeneity* rather than outcome prediction. Its defining trick is **honest
estimation**: within each tree the data are split in two, one half used to *decide
where to split*, the other, untouched half used to *estimate the treatment effect in
each leaf*. Decoupling the split-selection data from the effect-estimation data is what
makes the per-individual estimates statistically honest — it is what licenses valid
confidence intervals on each person's **CATE** (Conditional Average Treatment Effect),
which I obtain by bootstrapping. The result is not one effect but a *distribution* of
effects, one per profile, each with its own interval.

## SHAP moderators vs drivers

Here is the conceptual punch of the project, and the part most people get backwards.
You can run SHAP on a causal forest, and it looks just like running SHAP on a
predictive model — but it answers a *completely different* question.

SHAP on a **predictive** model identifies **drivers** of the outcome: which features
push 1978 earnings up or down. SHAP on a **causal forest** identifies **moderators**
of the *treatment effect*: which features make training work *better or worse*. These
are not the same list, and conflating them is a classic error.

The cleanest example in this data is prior earnings, `re74`. It is a powerful *driver*
— what you earned in 1974 strongly predicts what you earn in 1978, regardless of
training. But it is a weak *moderator* — training helps low-earners and high-earners
by roughly similar amounts, so it barely changes the *effect*. Education flips the
picture: a modest driver of the outcome, but a real moderator — the estimated benefit
of training climbs steadily across education quartiles. A model that only knew about
outcome drivers would target the wrong people.

## Real numbers from this project

The headline three-way ATE table is the whole argument in one place:

| Estimator | Data | ATE (1978 earnings) | 95% CI |
|---|---|---|---|
| Naive OLS (unadjusted) | CPS observational | **−$8,498** | [−9,893, −7,102] |
| LinearDML | CPS observational | **−$3,591** | [−8,475, +1,293] |
| LinearDML | NSW (RCT covariates) | **+$1,834** | [+513, +3,155] |
| RCT ground truth | NSW experiment | **+$1,794** | [+542, +3,113] |

Naive OLS says training destroys $8.5k. DML on the *same* observational data halves
the bias and its interval already straddles zero. DML on the experimental design lands
at **+$1,834** — within $40 of the randomised truth. The narrative holds.

Moderator scores (SHAP on the causal forest) rank **education**, **age**, and prior
earnings as the features that most shape *who benefits*. And policy targeting bears
this out: selecting the top 30% of profiles by predicted CATE yields a targeted mean
effect of **~$3,528**, against a population mean of **~$1,871** — roughly an 1.9×
lift from sending training to the people the model thinks it helps most.

## The EU AI Act angle

A model that decides *who* gets job training is not a toy. Under the EU AI Act it is
**Annex III §4 (Employment)** high-risk — AI used for recruitment, task allocation, or
access to training. That classification triggers concrete obligations, and this repo
is built to map onto them:

- **Article 9 (risk management):** a CI assertion fails the build if the DML estimate
  drifts away from the RCT ground truth — continuous risk monitoring as code.
- **Article 10 (data governance):** every dataset passes a `pandera` schema (dtype and
  range checks) *before* any estimation runs; provenance is recorded in calibration
  metadata.
- **Article 13 (transparency):** every prediction ships a plain-English summary —
  "training is predicted to raise this person's earnings by ~$X (95% CI …)" — not a
  bare number.
- **Article 14 (human oversight):** the API returns an explicit TREAT / DEFER
  recommendation with its interval, and defers when the CI crosses zero, leaving the
  final call to a human.

## Limitations and what's next

Honesty cuts both ways. NSW is small — 445 rows — so the confidence intervals are
wide, and a $1,834 point estimate with a $513–$3,155 band is not a precise
instrument. The data are from a 1970s programme; generalising to today's labour
market is *not* established and would need fresh data. CATE estimates on small samples
are noisy at the individual level, so the policy lift should be read as directional,
not contractual. A production deployment would need a defined recalibration cadence
and drift monitoring on live CATE distributions (which the repo scaffolds).

## Closing

Naive regression lied by $10,000 and got the *sign* wrong. Double ML, cross-fitting,
and causal forests recovered the experimental truth from confounded data and told us
*who* benefits most. Explore it yourself: the
[live Gradio CATE explorer](https://github.com/Priyrajsinh/causal-ml-hte) and the
4-tab Streamlit dashboard are linked from the
[GitHub repo](https://github.com/Priyrajsinh/causal-ml-hte), with the full derivations
in the `research-notes/` folder.
