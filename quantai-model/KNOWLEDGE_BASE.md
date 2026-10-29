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

## Sample size and power
Specify one primary outcome and estimand before calculating sample size. For a mean, inputs commonly include target difference, outcome standard deviation, alpha, power, allocation ratio and design. For a proportion, use the anticipated proportion and desired precision or target difference, depending on the design. Account for non-response, attrition, clustering, unequal allocation and complex sampling where relevant. A larger sample does not fix selection bias, measurement error or confounding. Do not report a sample-size number unless the assumptions and calculation are available.

## Diagnostic accuracy
For a defined threshold and reference standard, sensitivity is the proportion of reference-positive cases identified as positive; specificity is the proportion of reference-negative cases identified as negative. Positive and negative predictive values depend on prevalence in the tested population. Report confidence intervals and describe the reference standard, participant spectrum, threshold selection and handling of indeterminate results. ROC AUC summarises discrimination across thresholds; it does not by itself show calibration or clinical usefulness. Avoid selecting and evaluating a threshold on the same data without accounting for optimism.

## Multiple testing and subgroup analysis
Repeated testing increases the chance of at least one false-positive result. Define primary and secondary outcomes, distinguish prespecified analyses from exploratory analyses, and consider Holm, Bonferroni, false-discovery-rate or other methods suited to the inferential goal. Subgroup claims require evidence of interaction, not merely significance in one subgroup and non-significance in another. Report all analyses needed to understand the selection process.

## Longitudinal and clustered data
Repeated measurements from the same participant are correlated. Mixed-effects models represent subject or cluster-specific variation; GEE targets population-average effects with a working correlation structure and robust variance under suitable conditions. State the level of clustering and the number of clusters. Cluster-robust standard errors can be unreliable with very few clusters. For longitudinal studies, describe time origin, follow-up, missing visits, time-varying exposures and whether the target is within-person or between-person change.

## Complex survey analysis
Weights, strata and primary sampling units can affect both point estimates and uncertainty. Use the correct survey design declaration and variance estimator; do not treat weighted data as automatically representative or ignore non-response and coverage problems. For pooled survey rounds, verify comparable variable definitions, weight scaling and design features. For small-area or subgroup estimates, inspect effective sample size and confidence interval width. Document any domain/subpopulation analysis method.

## Survival analysis details
Define the event, time origin, censoring rule and time scale. Kaplan-Meier estimates rely on appropriate independent censoring assumptions. The log-rank test is an unadjusted comparison and may be less informative when hazards cross. Cox regression assumes proportional hazards unless extended; inspect diagnostics such as Schoenfeld residuals and consider time-varying effects when appropriate. A hazard ratio is conditional on the risk set and model, not a risk ratio. Competing events may require competing-risk methods rather than treating all other events as ordinary independent censoring.

## Missing data and sensitivity analysis
Distinguish item missingness, unit non-response, loss to follow-up and structurally inapplicable values. Describe missingness by key variable and study group. Multiple imputation should include variables associated with missingness and the outcome, preserve relevant structure and reflect the analysis model. Imputation does not guarantee unbiasedness when assumptions fail. Use sensitivity analysis for plausible departures from missing-at-random assumptions when the question is consequential. Do not silently delete rows or recode unusual values without checking documentation.

## Prediction model development
Define the prediction time, target population, outcome horizon and intended decision. Avoid data leakage by separating participants or sites as needed and fitting preprocessing only within training folds. Use cross-validation or bootstrap internal validation to estimate optimism, and external or temporal validation where possible. Report discrimination, calibration and clinically or operationally relevant utility as appropriate. A random train/test split may be misleading for clustered, temporal or geographically structured data. Compare against simple baselines and check subgroup performance and drift.

## Causal inference cautions
A causal estimand must be stated, such as an average treatment effect or risk difference over a defined period. A causal diagram can help distinguish confounders, mediators, colliders and selection variables. Adjusting for a mediator can remove part of a total effect; adjusting for a collider can induce bias. Propensity-score methods require overlap, measured confounding control and correctly implemented design; matching or weighting does not address unmeasured confounding. Instrumental variables and difference-in-differences rely on strong assumptions that must be argued, not merely named. A statistical model alone cannot verify these assumptions.

## Meta-analysis and evidence synthesis
Define eligibility criteria, outcomes, effect measures and synthesis methods before examining study results where possible. Distinguish fixed-effect and random-effects assumptions. Heterogeneity statistics do not explain why results differ, and a random-effects model does not make incompatible studies comparable. Assess risk of bias, publication bias where feasible, small-study effects and certainty of evidence. Do not pool estimates with incompatible definitions or populations without a justified harmonisation strategy.

## Reproducibility and reporting
Record data provenance, cleaning decisions, software and version, model specification, exclusions, missing-data handling and deviations from the analysis plan. Preserve code and analysis logs while protecting confidential data. Use version-controlled scripts and avoid editing results by hand. Report estimates with units and denominators, explain reference groups, and label exploratory analyses. Follow relevant reporting standards and the target journal's instructions.

## Boundaries for QuantAI
The model can explain these principles and suggest candidate methods, but it must ask for missing design details, state assumptions, and distinguish recommendations from executed analyses. It must not claim a calculation was performed unless the deterministic engine returned the result. It must not generate citations or official current statistics without verifying the source. For high-stakes research, recommend review by a qualified statistician, supervisor or ethics committee.
