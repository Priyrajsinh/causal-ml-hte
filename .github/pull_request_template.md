## Summary
<!-- 1-3 bullets -->

## v3 + Causal-ML checklist (CLAUDE.md C1-C46)
- [ ] `make ci` green locally
- [ ] No `print()` — `get_logger(__name__)` only
- [ ] `pandera LALONDE_SCHEMA.validate(df)` runs before any analysis
- [ ] `safe_predict()` wraps every CATE prediction call
- [ ] No naive-OLS-as-causal-estimate slippage (rule C40)
- [ ] If DML touched: NSW ATE 95% CI still contains $1,794 (rule C37)
- [ ] No new secrets (detect-secrets clean)
- [ ] No new CVEs (pip-audit clean)
- [ ] Test coverage >= 70%

## Test plan
- [ ] ...
