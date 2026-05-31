# 03 — The LaLonde benchmark: why this dataset

**Anchors:** LaLonde (1986), *Evaluating the Econometric Evaluations of Training Programs with
Experimental Data*, American Economic Review 76(4):604–620; Dehejia & Wahba (1999), *Causal
Effects in Nonexperimental Studies*, JASA 94(448):1053–1062.

## LaLonde's 1986 challenge
The **National Supported Work (NSW)** demonstration was a 1970s randomised job-training programme:
applicants were *randomly* assigned to treatment or control, so a simple difference of means
identifies the true average effect (≈ +$1,794 on 1978 earnings). LaLonde then asked a pointed
question: if we *throw away* the experimental control group and instead use **observational**
comparison groups drawn from national surveys (PSID, CPS), can the standard econometric methods of
the day recover that experimental answer? His finding was damning: the observational estimates
were all over the map — often the wrong sign, frequently far from the experimental truth. The
non-experimental controls differ systematically from the treated (the treated were unemployed and
low-earning; survey controls were largely employed), so the methods measured **selection**, not
the programme. The paper became a foundational cautionary tale for observational causal inference.

## Dehejia & Wahba's 1999 revisit
Dehejia & Wahba returned to LaLonde's data with **propensity-score** methods. Two contributions
made the dataset the modern benchmark it is today:

1. They restricted the sample to a subset with the pre-treatment earnings variables **re74** and
   **re75** observed — a richer covariate set that makes unconfoundedness more plausible. This
   "Dehejia–Wahba subset" (the ~445-row NSW treated/control split used here) is now the standard
   construction.
2. They showed that with the right covariates and propensity-score matching, observational
   methods *could* get much closer to the experimental ATE — turning LaLonde's pessimism into a
   constructive test: *given good covariates, can your estimator recover the experimental
   number?*

## Why it remains the test bench
The dataset is small, messy, and confounded in a *known* way, and — crucially — it comes with an
**experimental ground truth** to grade against. That combination makes it the natural unit test
for any new HTE estimator: fit on the observational construction, and check whether you land near
+$1,794. Most papers introducing a causal ML method still report a LaLonde result.

## How it shows up in this repo
Two CSVs (NSW RCT, ~445 rows; CPS observational, ~16K rows), both validated by the same pandera
`LALONDE_SCHEMA`. The headline three-way table grades naive OLS (−$8,497, fails), DML on CPS, and
DML on NSW (+$1,834, recovers the truth) against the RCT ground truth — LaLonde's challenge,
answered with modern tooling. Rule C37 wires the "CI must contain $1,794" check into CI itself.
