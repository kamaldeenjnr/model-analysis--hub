# QuantAI local knowledge: statistical test selection

## Start with the design
Before choosing a test, identify the outcome type, number of groups, independence or pairing, sampling design, target estimand and assumptions. A variable's storage type alone does not determine its scientific role.

## Common starting points
- Continuous outcome, two independent groups: independent-samples t-test if its assumptions are reasonable; Welch's t-test is often a safer default when group variances differ; Mann-Whitney compares distributions/ranks and is not automatically a test of medians.
- Continuous outcome, paired measurements: paired t-test for approximately normal within-person differences; Wilcoxon signed-rank for suitable non-normal paired differences. Do not treat paired measurements as independent.
- Continuous outcome, three or more independent groups: ANOVA with suitable diagnostics; Welch ANOVA for unequal variances; Kruskal-Wallis for rank-based comparison. If the overall test motivates pairwise comparisons, control multiplicity.
- Categorical variables: chi-square test when expected cell counts support its approximation; Fisher's exact test for sparse 2x2 tables and suitable exact procedures for other tables.
- Two continuous variables: Pearson correlation for a linear relationship under suitable conditions; Spearman correlation for monotonic rank association or when robust rank-based analysis is appropriate. Neither establishes causality.
- Binary outcome: logistic regression estimates odds ratios. For cross-sectional studies with a common outcome, modified Poisson regression with robust variance can estimate prevalence ratios when correctly specified. Choice depends on design and estimand.
- Count outcome: Poisson regression when conditional mean/variance assumptions are reasonable; assess overdispersion and consider negative binomial models. Include an offset for unequal person-time when appropriate.
- Ordered categorical outcome: ordinal logistic regression may be appropriate; assess the proportional-odds assumption.
- Unordered outcome with three or more categories: multinomial logistic regression may be appropriate.
- Time-to-event outcome with censoring: Kaplan-Meier curves and log-rank test for unadjusted group comparisons; Cox regression for adjusted hazard ratios if proportional hazards is plausible. Check censoring and model assumptions.
- Clustered or repeated observations: consider mixed-effects models or GEE, or design-appropriate cluster-robust standard errors. Standard independent-observation methods may understate uncertainty.

## Assumptions and reporting
Check independence, distributional conditions relevant to the estimator, residual patterns, linearity on the link scale, influential observations, collinearity, convergence, overdispersion and proportional hazards as applicable. Report the analysis sample size, missing-data handling, effect estimate, confidence interval and exact p-value where available. The software must calculate the result; a language model should never invent numeric output.
