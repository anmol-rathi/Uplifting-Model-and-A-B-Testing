# Experiment Decision Memo
**Project:** Feature Rollout Experiment — Personalized Engagement Nudge  
**Author:** [Your Name]  
**Date:** [Date]  
**Status:** RECOMMENDATION — TARGETED ROLLOUT

---

## Executive Summary

The experiment ran for 4 weeks across ~100K users (50K treatment, 50K control). The feature produced a **statistically and practically significant lift in conversion** at the aggregate level (+2.4pp, p < 0.001, 95% CI: [+1.8pp, +3.0pp]). However, average treatment effect masks critical heterogeneity: **the lift is almost entirely concentrated in mobile users with < 90 days tenure.** For desktop users and long-tenure users (> 365 days), the point estimate is indistinguishable from zero and the confidence interval extends into negative territory.

**Recommendation: Ship to new mobile users only (top 20% by predicted uplift). Do not ship to long-tenure desktop users at this time.**

---

## The Numbers

| Metric | Control | Treatment | Δ | p-value | Practical? |
|---|---|---|---|---|---|
| Conversion rate (primary) | 8.1% | 10.5% | +2.4pp | < 0.001 | ✅ Yes (MDE was 1.5pp) |
| 7-day retention (guardrail) | 41.2% | 41.0% | -0.2pp | 0.61 | ✅ No harm |
| Avg. revenue per user (guardrail) | $3.42 | $3.39 | -$0.03 | 0.74 | ✅ No harm |
| Page load latency P95 (guardrail) | 420ms | 418ms | -2ms | 0.81 | ✅ No harm |

Guardrail metrics show no statistically significant degradation. The feature does not harm retention or revenue.

---

## Why Not Ship to Everyone

The uplift model (X-Learner + Causal Forest ensemble) revealed substantial heterogeneity:

- **New mobile users (tenure < 90d):** predicted ITE ≈ +7–9pp. This segment is ~22% of users but drives **~78% of total incremental conversions.**
- **Mid-tenure users:** predicted ITE ≈ +1–3pp. Marginal; borderline practical significance.
- **Long-tenure desktop users:** predicted ITE ≈ -0.5 to +0.5pp. Confidence intervals straddle zero. Shipping to this segment is waste and may cause mild annoyance (novelty fatigue signal).

Shipping to everyone gives us the same expected incremental conversions from the high-uplift segment, while burning resources (infrastructure cost, user attention budget) on segments where the feature does nothing. **Targeted rollout is the efficient choice.**

---

## Targeting Rule (Deployable Today)

Use the trained X-Learner model's predicted uplift score. Apply a threshold of **uplift_score ≥ 0.04** (roughly the top 20% of the user base) as the rollout gate. At this threshold:

- Expected incremental conversions captured: ~78% of total possible lift
- Users receiving the feature: ~20% of base
- Cost efficiency vs. blanket rollout: **~3.9× more efficient per incremental conversion**

The model is already trained and serialized to `models/xlearner_final.joblib`. It requires the same 5 input features available at request time: `device`, `tenure_days`, `engagement_score`, `past_spend_bucket`, `age_bucket`.

---

## Monitoring Plan — First 2 Weeks Post-Launch

| What to watch | Tool | Alert threshold |
|---|---|---|
| Conversion rate for targeted users | Dashboards | Drop > 1pp vs. experiment estimate |
| Guardrail: 7-day retention | Dashboards | Drop > 0.5pp |
| Model prediction distribution drift | ML monitoring | KS statistic > 0.1 vs. validation set |
| SRM / assignment check | Automated job | Chi-square p < 0.01 |
| Latency P95 | APM | > 500ms (20% regression) |

If conversion rate drops > 1pp below the experiment point estimate within the first 7 days, trigger an automatic rollback and file an incident. Do not wait for statistical significance before rolling back — in production, speed of detection matters more than certainty.

---

## What I'd Test Next

1. **Personalization of the nudge content itself.** We proved the *feature* works for new mobile users. The next question is whether the *message* can be optimized — a multivariate test on copy and CTA, targeting the same new-mobile segment.

2. **Second-order effect: does targeting create a "neglected segment" problem?** Long-tenure desktop users don't convert more from this nudge, but are they noticing they're not getting it? Run a brief survey/exit intent test on that segment.

3. **Long-term retention for targeted users (90-day holdout).** Our experiment measured 7-day retention. High conversion in the short term with poor 30–90 day retention would be a bad outcome. Run a 90-day holdout on a small slice of the rollout population.

---

## Caveats & Limitations

- **No long-run data:** The experiment ran 4 weeks; we cannot rule out novelty effects fading after 6–8 weeks. The monitoring plan above partially addresses this.
- **Uplift model uncertainty:** Confidence intervals on individual ITE estimates are wide. The targeting rule uses a threshold that is practically reasonable, but individual predictions should not be treated as precise.
- **External validity:** Criteo validation (Notebook 04) confirmed the pipeline works on research-grade real data, but the Criteo population is display-ad users, not our specific user base. Transfer assumptions are implicit.
- **CUPED only partially helped:** CUPED reduced variance by ~18% in the overall test, but engagement_score (our pre-experiment covariate) is only moderately correlated with the outcome (r ≈ 0.31). With a better pre-experiment metric, we could have achieved the same power with ~25% fewer users.

---

*This memo reflects the analysis in notebooks 01–04 of this repository. All numbers are derived from the synthetic simulation with known ground truth (Notebooks 01–03) and confirmed directionally on the Criteo dataset (Notebook 04).*
