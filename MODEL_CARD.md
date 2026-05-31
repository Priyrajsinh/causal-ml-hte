# Model Card — Causal ML · Heterogeneous Treatment Effects

*This document is the technical documentation required for a high-risk AI system under
EU AI Act Article 11 (see "EU AI Act framing" below).*

## Model details
- **Name:** Causal Forest + Double Machine Learning for Heterogeneous Treatment Effects
- **Owner:** Priyrajsinh Parmar — https://github.com/Priyrajsinh
- **Repository:** https://github.com/Priyrajsinh/causal-ml-hte
- **License:** MIT
- **Version / date:** v1.0 — 2026-05-31
- **Stack:** Python 3.12 · EconML (`LinearDML`, `CausalForestDML`) · LightGBM (nuisance) ·
  SHAP `TreeExplainer` · pandera · Pydantic v2 · MLflow · FastAPI · Gradio · Streamlit ·
  slowapi · Prometheus · DVC
- **Estimators shipped:**
  - `LinearDML` (Robinson decomposition, `cv=5` cross-fitting) — average treatment effect.
  - `CausalForestDML` (honest forest, bootstrap CIs) — per-individual CATE; served by the API.

## Intended use
- **Primary use:** estimating and targeting treatment effects for **active-labour-market
  job-training programmes** structurally similar to the NSW demonstration — i.e. deciding,
  under a budget constraint, which applicants gain the most from enrolment.
- **Users:** policy analysts and case-workers who treat the output as **decision support**,
  always alongside its confidence interval (see Human oversight).
- **NOT intended for:**
  - Criminal-justice risk scoring or any setting where a false positive/negative carries
    asymmetric ethical or liberty consequences.
  - Healthcare resource allocation or clinical decisions.
  - Automated, human-out-of-the-loop enrolment or rejection decisions.
  - Any population materially different from the training cohort (see Limitations).

## Training data
- **LaLonde NSW** — 445 rows, **randomised controlled trial**, 1976–1978 US labour market.
  Random assignment makes the ATE identifiable by a simple difference of means and provides
  the ground-truth target (≈ **$1,794**) the observational estimators must recover.
- **LaLonde CPS** — ~16,177 rows, **observational** controls drawn from the Current Population
  Survey. Used to *demonstrate confounding*: the treated (formerly unemployed) are compared
  against a largely employed control population, producing severe selection bias.
- **Schema:** validated by the pandera `LALONDE_SCHEMA` (8 covariates + `treat` + `re78`)
  *before* any estimation (rule C34). See `CLAUDE.md` → Dataset Schema.
- **Provenance / governance:** both raw CSVs are DVC-tracked with SHA-256 checksum sidecars;
  they are never committed to git (rule C41).
- **Time-of-data limit:** these data describe the 1976–1978 US labour market. The labour
  market has changed substantially in ~50 years, so generalisability beyond the original
  cohort is **not established** and must not be assumed.

## Headline ATE results — OLS bias vs DML recovery
The scientific narrative (rule C40): naive OLS on observational data is badly biased; Double
ML recovers the RCT truth. **Never present naive OLS as the causal estimate.**

| Estimator | Data | ATE ($) | 95% CI | Note |
|---|---|---|---|---|
| OLS (unadjusted) | CPS (observational) | −8,497 | [−9,893, −7,102] | naive — confounded; "training hurts" |
| OLS (adjusted) | CPS (observational) | +699 | [−374, +1,773] | controls help but CI crosses zero |
| Double ML (`LinearDML`) | CPS (observational) | −3,591 | [−8,475, +1,293] | honest wide CI on hard data |
| Double ML (`LinearDML`) | NSW (RCT) | +1,834 | [+513, +3,155] | recovers the RCT truth |
| **RCT ground truth** | **NSW (RCT)** | **+1,794** | **[+542, +3,113]** | unbiased by construction |

(For reference, OLS on the RCT data itself gives +1,676 — close to the truth precisely because
randomisation removed the confounding.)

## CATE / heterogeneity
`CausalForestDML` estimates a Conditional Average Treatment Effect (CATE) per individual.
Moderator strength = range of mean CATE across covariate bins:

| Covariate | CATE range | Min → Max | Mean \|SHAP\| on CATE |
|---|---|---|---|
| **age** | **$2,455** | $776 → $3,231 | 933 |
| education | $1,564 | $1,088 → $2,652 | 379 |
| nodegree | $998 | $1,654 → $2,652 | 28 |
| married | $631 | $1,765 → $2,396 | 18 |
| re74 | $611 | $1,413 → $2,024 | 138 |
| re75 | $447 | $1,536 → $1,983 | 155 |

**Top moderator: age** — older participants gain far more from training. `re74` (1974 earnings)
is a strong *outcome driver* but a comparatively *weak moderator* of the treatment effect.

**Policy targeting (top-30% by predicted CATE):** funding the highest-CATE 134 of 445
applicants yields a mean lift of **$3,528** (vs $1,157 for the untreated 70%) and an estimated
**$472,801** total earnings lift — roughly 3× the lift of an untargeted selection.

## Training-serving skew
- `safe_predict()` wraps every CATE prediction (rule C36): it checks each input row against
  `models/training_stats.json` — the per-feature mean / std / min / max of the 8 covariates
  computed from the NSW training data (validated with pandera on load, rule C23).
- Inputs that fall **outside the training range** are logged as warnings. The `/api/v1/cate`
  route **still returns a prediction**, but the response should be treated as lower-confidence
  and reviewed by a human, because the causal forest is extrapolating beyond observed support.
- The bootstrap CI on every prediction widens naturally in sparse regions of covariate space,
  giving a second, quantitative signal of low support.

## Limitations
- **Small RCT sample** (445 rows) — bootstrap CIs are wide; the forest cannot resolve
  fine-grained interactions, and rare covariate combinations are extrapolations.
- **Temporal validity** — trained on the 1976–1978 labour market; do not assume the effects
  transfer to today's economy.
- **Unconfoundedness** — DML on the observational CPS construction assumes the 8 covariates
  capture the relevant confounding. The DML·CPS estimate has a very wide CI, honestly
  reflecting that the observational data carry limited information.
- **Scope** — calibrated for NSW-style job-training targeting only (see Intended use).

## EU AI Act framing (Annex III §4 — Employment)
A treatment-effect estimator used to target a job-training programme is **high-risk** under
Annex III §4 (employment, workers management, access to self-employment).

- **Article 9 (Risk management):** CATE-drift monitor (`cate_distribution_drift_total`,
  `cate_mean_gauge`, `cate_std_gauge` on `/metrics`, PSI > 0.2 alarm) + a CI assertion
  (rule C37) that fails the build if the NSW DML CI drifts away from the $1,794 ground truth.
- **Article 10 (Data governance):** pandera `LALONDE_SCHEMA` validation before estimation,
  DVC + SHA-256 checksums on both raw CSVs, documented three-data-source layout.
- **Article 13 (Transparency):** every `/api/v1/cate` response carries an `nl_summary`
  translating the estimate into a human-readable TREAT / DEFER recommendation.
- **Article 14 (Human oversight):** each "TREAT" surfaces its 95% bootstrap CI; ambiguous
  cases (CI crossing zero) auto-route to "DEFER" (escalate-to-human).
- **Article 11 (Technical documentation):** this MODEL_CARD is that document.

*Engineering framing, not a legal opinion. A real deployment would additionally require
conformity assessment, post-market monitoring, and CE marking — out of scope here.*

## Citations
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters.
  *The Econometrics Journal*, 21(1), C1–C68. arXiv:1608.00060
- Athey, S., & Wager, S. (2019). Estimating Treatment Effects with Causal Forests: An
  Application. *Observational Studies*, 5(2), 37–51. arXiv:1902.07409
- LaLonde, R. J. (1986). Evaluating the Econometric Evaluations of Training Programs with
  Experimental Data. *American Economic Review*, 76(4), 604–620.
- Dehejia, R. H., & Wahba, S. (1999). Causal Effects in Nonexperimental Studies: Reevaluating
  the Evaluation of Training Programs. *JASA*, 94(448), 1053–1062.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and
  Biomedical Sciences: An Introduction*. Cambridge University Press. ISBN 978-0-521-88588-8.
