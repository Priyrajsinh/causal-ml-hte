# 05 — The EU AI Act and employment AI

**Anchor:** Regulation (EU) 2024/1689 (the EU AI Act), in particular **Annex III §4** and the
high-risk obligations in Articles 9–15. Primer: https://artificialintelligenceact.eu/.

## Why a job-training targeting model is high-risk
The Act takes a risk-tiered approach: prohibited, high-risk, limited-risk, minimal-risk. **Annex
III** enumerates the high-risk use cases, and **§4 (Employment, workers management and access to
self-employment)** covers *"AI systems intended to be used … for making decisions affecting the
terms of work-related relationships, the promotion or termination … or for monitoring and
evaluating the performance and behaviour"* of people, as well as systems used in recruitment and
selection. A model that scores applicants by predicted benefit and decides **who gets enrolled in
a publicly funded job-training programme** affects access to work-related opportunity — so it sits
inside §4 and inherits the high-risk obligations. This is framing for engineering discipline, not
a legal determination.

## The articles, mapped to a HTE estimator
The high-risk obligations are not abstract; each maps onto a concrete artifact in this repo.

- **Article 9 — Risk management system.** A *continuous* process to identify and mitigate risks.
  Here: a CATE-distribution drift monitor (`cate_distribution_drift_total`, `cate_mean_gauge`,
  `cate_std_gauge` on `/metrics`, PSI > 0.2 alarm) plus a CI gate (rule C37) that fails the build
  if the NSW DML interval ever drifts off the +$1,794 experimental ground truth. Drift is caught,
  not discovered in production.
- **Article 10 — Data and data governance.** Training data must be relevant, representative, and
  examined for bias. Here: pandera `LALONDE_SCHEMA` validation *before* estimation, DVC + SHA-256
  checksums on both raw CSVs, and an explicit documented limit — the data describe the 1976–1978
  labour market, so representativeness for today is *not* claimed.
- **Article 13 — Transparency and provision of information.** Outputs must be interpretable by the
  deployer. Here: every `/api/v1/cate` response carries an `nl_summary` translating the estimate
  into a plain-English TREAT / DEFER recommendation with its dollar effect and interval.
- **Article 14 — Human oversight.** A human must be able to understand, override, and not
  over-rely on the system. Here: every "TREAT" surfaces its 95% bootstrap CI so a case-worker
  sees the uncertainty; ambiguous cases (CI crossing zero) auto-route to "DEFER" — the
  escalate-to-human path.
- **Article 11 — Technical documentation.** High-risk systems require documentation before going
  to market. Here: `MODEL_CARD.md` carries the headline ATE table, heterogeneity audit, moderator
  analysis, training-serving skew note, intended-use limits, and citations.

## What is deliberately out of scope
A real deployment would also need third-party / internal **conformity assessment**, registration,
**post-market monitoring**, incident reporting, and **CE marking** before lawful use. Those are
organisational and legal processes beyond a portfolio repository. The point of this note is that
the *technical* obligations — risk monitoring, data governance, transparency, human oversight,
documentation — are exactly the engineering practices already wired into this project.
