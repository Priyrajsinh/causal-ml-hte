"""Translate a CATE point estimate + 95% bootstrap CI into a recommendation.

Rule C45 — the API's natural-language output. Used by the FastAPI ``/cate``
route and by the Gradio + HF-Space CATE explorer. The decision rule:

- If the CI crosses zero (``ci_lower <= 0 <= ci_upper``): no detectable
  effect → DEFER. This is the most common output for low-CATE individuals
  in NSW — bootstrap CIs naturally widen when the signal is weak.
- Otherwise, the recommendation tracks the sign of the CATE: positive →
  TREAT (training raises earnings), negative → DEFER (with a warning that
  training is predicted to *lower* earnings for this profile).
"""


def translate_cate(cate: float, ci_lower: float, ci_upper: float) -> tuple[str, str]:
    """Return ``(recommendation, nl_summary)`` from a CATE + 95% CI.

    The recommendation is one of {``TREAT``, ``DEFER``}. The summary is a
    plain-English description suitable for surfacing in the FastAPI response,
    Gradio UI, and HF Space.
    """
    if ci_lower <= 0 <= ci_upper:
        rec = "DEFER"
        nl = (
            f"No detectable effect for this profile. Predicted CATE is "
            f"${cate:,.0f} but the 95% bootstrap CI [${ci_lower:,.0f}, "
            f"${ci_upper:,.0f}] crosses zero. Recommendation: DEFER."
        )
    elif cate > 0:
        rec = "TREAT"
        nl = (
            f"Job training is predicted to raise this person's 1978 earnings "
            f"by ${cate:,.0f} (95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]). "
            f"Recommendation: TREAT."
        )
    else:
        rec = "DEFER"
        nl = (
            f"Training is predicted to *lower* earnings by ${-cate:,.0f} "
            f"for this profile (95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]). "
            f"Recommendation: DEFER."
        )
    return rec, nl
