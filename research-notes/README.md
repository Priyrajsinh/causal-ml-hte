# Research notes

A short reading log behind the methods in this repository. Each note is ~1 page: what the
paper says, why it matters here, and how it shows up in the code.

| # | Topic | Anchor reference | Key takeaway |
|---|-------|------------------|--------------|
| [01](01-double-ml-foundations.md) | Double / Debiased ML foundations | Chernozhukov et al. (2018) | Orthogonal moments + cross-fitting let you plug ML nuisance models into a causal estimate without first-order bias. |
| [02](02-causal-forest-honest-splitting.md) | Causal forests & honest splitting | Athey & Wager (2019) | Splitting and estimating on *disjoint* samples is what makes per-leaf treatment effects valid for inference. |
| [03](03-lalonde-classic-dataset.md) | The LaLonde benchmark | LaLonde (1986); Dehejia & Wahba (1999) | The canonical test: can an observational estimator recover the experimental ATE? |
| [04](04-design-rationale.md) | Design choices in this project | — | Why LightGBM nuisance, why bootstrap inference, why no test split. |
| [05](05-eu-ai-act-employment.md) | EU AI Act, employment AI | EU AI Act, Annex III §4 | A job-training targeting model is high-risk; Articles 9–14 map onto this repo. |

See the [README References](../README.md#references) section for full citations.
