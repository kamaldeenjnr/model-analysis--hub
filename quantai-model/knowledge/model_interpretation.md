# QuantAI local knowledge: model interpretation

## Linear regression
A coefficient is the expected change in the outcome associated with a one-unit increase in a predictor, holding included covariates constant, under the model. Check functional form, residual variance, influential observations and dependence. A coefficient is not automatically causal.

## Logistic regression
Exponentiated coefficients are odds ratios. An odds ratio is not a risk ratio and can substantially differ from it when the outcome is common. State the reference category, adjustment set and confidence interval. Check separation, influential points, calibration, functional form and collinearity where relevant.

## Poisson and modified Poisson regression
Poisson regression models a count mean or rate under its assumptions; an exposure offset is needed when modelling rates over differing exposure time. Modified Poisson regression with robust variance is commonly used to estimate prevalence ratios or risk ratios for binary outcomes in suitable cross-sectional or trial settings. Robust variance does not repair confounding, incorrect functional form or every model misspecification.

## Cox regression
A hazard ratio compares instantaneous event rates among individuals still at risk, conditional on the model covariates. It is not a ratio of cumulative probabilities. Check proportional hazards and state how censoring and time origin were defined.

## Confidence intervals and p-values
A 95% confidence interval reflects uncertainty under the statistical procedure and assumptions. A wide interval signals imprecision. A p-value is not the probability the hypothesis is true. A non-significant result is not proof of no association; assess the estimate, interval, design, power and potential bias.

## Prediction versus explanation
Prediction models should be evaluated on data not used for fitting, ideally with external or temporal validation where appropriate. Assess discrimination and calibration, not accuracy alone. Avoid leakage from variables recorded after the prediction time or from preprocessing performed on the full dataset before splitting. For causal interpretation, define the estimand and assumptions; predictive importance is not causal effect.
