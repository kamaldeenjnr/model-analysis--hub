# QuantAI specialist knowledge: research methods and biostatistics

Educational summary; obtain qualified review before production use.

## Study design and causal inference
Define population, exposure/intervention, comparator, outcome, setting and time frame. Cross-sectional studies measure exposure and outcome around the same time and often cannot establish temporal order. Cohort studies follow people over time. Case-control studies sample on outcome and compare prior exposures. Randomised trials allocate interventions randomly. Observational association does not establish causation; consider confounding, selection bias, information bias, reverse causation and measurement error.

## Sampling and missingness
Probability sampling with known non-zero selection probabilities supports design-based inference when analysed appropriately. Convenience and purposive samples may be useful for exploratory or qualitative aims but limit statistical generalisation. Sample size depends on primary outcome, target precision/effect, alpha, power, design effect, non-response and planned subgroup analyses. Report missingness by variable. Never assume 99 or 999 denotes missing without checking a codebook. Multiple imputation can be appropriate under defensible missing-at-random assumptions; consider sensitivity analysis when mechanisms are uncertain.

## Test selection
- Continuous outcome, two independent groups: t-test when appropriate; Welch's test is often safer under unequal variances. Mann-Whitney is rank-based and is not automatically a test of medians.
- Paired continuous measurements: paired t-test or a suitable paired rank method; do not treat repeated measures as independent.
- Three or more groups: ANOVA/Welch ANOVA or Kruskal-Wallis as appropriate; control multiplicity for pairwise follow-up.
- Categorical variables: chi-square when expected counts support its approximation; Fisher exact for sparse 2x2 tables.
- Continuous association: Pearson for linear association under suitable conditions; Spearman for monotonic rank association. Neither establishes causality.
- Binary outcome: logistic regression estimates odds ratios. Modified Poisson with robust variance can estimate prevalence ratios for common binary outcomes in suitable cross-sectional designs.
- Count outcome: Poisson if assumptions are reasonable; assess overdispersion and consider negative binomial. Include an offset for unequal person-time when appropriate.
- Ordered outcome: ordinal logistic may be suitable; assess proportional odds. Unordered outcome with 3+ categories may need multinomial logistic.
- Time-to-event with censoring: Kaplan-Meier/log-rank for unadjusted comparison; Cox regression for adjusted hazard ratios if proportional hazards is plausible.
- Clustered/repeated data: consider mixed models, GEE or design-appropriate cluster-robust errors.

## Interpretation
A p-value is not the probability the null is true. Report effect estimates and confidence intervals, not p-values alone. A confidence interval describes uncertainty under the model and sampling assumptions; it does not remove bias. Statistical significance is not practical importance. Logistic odds ratios are not risk ratios, especially when outcomes are common. Hazard ratios compare instantaneous event rates among those still at risk, not cumulative probabilities. A non-significant result does not prove no association.

## Regression diagnostics and prediction
Check functional form, residual patterns, influential observations, collinearity, convergence, overdispersion and proportional hazards as appropriate. Prediction models require held-out evaluation, calibration and discrimination assessment; avoid data leakage during preprocessing. Predictive importance is not a causal effect.

## Ghana public-health context
Potential sources include Ghana Statistical Service, Ghana Health Service, Ministry of Health, DHS, MICS, WHO, UNICEF and peer-reviewed studies. Verify survey year, sampling frame, weights, strata, clusters, variable definitions and data-use terms. Complex surveys need survey-aware estimates and standard errors. Use only authorised data; minimise identifiers and follow ethics approvals and data-sharing conditions.

## Reporting resources
Check official versions of STROBE, CONSORT, PRISMA, STARD, TRIPOD, COREQ or SRQR as relevant. Source register: https://www.strobe-statement.org/ ; https://www.consort-statement.org/ ; https://www.prisma-statement.org/ ; https://www.tripod-statement.org/ ; https://statsghana.gov.gh/ ; https://ghs.gov.gh/ .
