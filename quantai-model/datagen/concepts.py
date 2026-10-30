"""Concept questions for QuantAI v4 data.

1. Handbook questions: QuantAI's own vetted handbook entries (RESEARCH_KB and
   quantai-kb.js, written by Model Analysis Hub) paired with new question
   wordings. The answer is the handbook text, so the model learns the app's
   reviewed explanations instead of new, unreviewed ones.
2. "In more depth:" questions: the chat sends "In more depth: <question>" to the
   model when the user wants more than the built-in answer. The answers below
   are fuller explanations of core topics; they contain no study results.
"""
from __future__ import annotations

import hashlib
import random

T_HANDBOOK = {
    "train": ["Can you tell me about {t}?", "I'm confused about {k}. Can you help?", "{K}: what should I know?",
              "Teach me about {k} briefly.", "Help me understand {k}."],
    "eval": ["Give me a quick overview of {t}.", "I keep seeing '{k}' in papers. What's that about?"],
}


def handbook_split(title: str) -> str:
    """Stable 85/15 split of handbook entries by title."""
    return "eval" if int(hashlib.sha256(title.encode()).hexdigest(), 16) % 100 < 15 else "train"


def handbook_examples(kb: list[dict], split: str, rng: random.Random) -> list[dict]:
    out = []
    for entry in kb:
        if handbook_split(entry["t"]) != split:
            continue
        k = rng.choice(entry["k"])
        q = rng.choice(T_HANDBOOK[split]).format(t=entry["t"], k=k, K=k[:1].upper() + k[1:])
        out.append({"intent": "concept_handbook", "message": q, "reply": entry["a"], "action": None, "topic": entry["t"]})
    return out


DEEPER = [
    ("What is a p-value?", "train",
     "A **p-value** is the probability of observing data at least as extreme as yours if the null hypothesis were true and the model's assumptions held.\n\n"
     "What it is **not**: the probability that the null hypothesis is true, the probability the result is due to chance, or a measure of effect size. A tiny effect can have a very small p-value in a large sample, and an important effect can have a large p-value in a small one.\n\n"
     "Good practice:\n- Report the effect estimate with its 95% confidence interval, and give the exact p-value (p = .03 rather than p < .05).\n- Pre-specify your main hypotheses; many unplanned tests inflate false positives.\n- Judge results by the size and precision of the effect and by study quality, not by crossing 0.05."),
    ("What does a confidence interval tell me?", "train",
     "A **95% confidence interval (CI)** is a range of effect values that are reasonably compatible with your data, given the model. The method is designed so that, over many repeated studies, 95% of such intervals would contain the true value.\n\n"
     "How to read it:\n- **Width** shows precision: narrow intervals come from larger samples and less variability.\n- **Position** shows plausible effect sizes. For ratios (OR, RR, HR), an interval including 1 is compatible with no effect; for differences, one including 0 is.\n- Values near the ends are less compatible with the data than values near the centre.\n\n"
     "A single interval either contains the true value or not; the 95% describes the method, not that one interval. Report the CI alongside every estimate, because it says more than a p-value alone."),
    ("What is the difference between an odds ratio and a prevalence ratio?", "train",
     "Both compare a binary outcome between groups, but on different scales.\n\n"
     "- **Prevalence (or risk) ratio**: the proportion with the outcome in one group divided by that in another. A PR of 1.5 means the outcome is 1.5 times as common.\n- **Odds ratio**: the ratio of the odds, p/(1 − p). It approximates the risk ratio only when the outcome is rare (roughly under 10%).\n\n"
     "When the outcome is common, the OR is further from 1 than the PR and is easy to misread as a risk ratio. In cross-sectional studies of common outcomes, prevalence ratios from **modified Poisson regression** (Poisson with robust standard errors) are often easier to interpret. Logistic regression and ORs remain standard in case-control studies, where risks cannot be estimated directly."),
    ("What is confounding?", "train",
     "**Confounding** distorts the association between an exposure and an outcome because a third factor is linked to both and is not on the causal pathway.\n\n"
     "Example: an apparent link between coffee drinking and lung cancer can be confounded by smoking, which is more common among coffee drinkers and causes lung cancer.\n\n"
     "How to handle it:\n- **Design**: randomisation, restriction or matching.\n- **Analysis**: adjust for confounders in a regression model, or stratify.\n\n"
     "Choose confounders from subject knowledge and a causal diagram (DAG), not from p-values. Do not adjust for mediators (on the pathway) or colliders, which can introduce bias. Even after adjustment, unmeasured or poorly measured confounders can leave residual confounding, so observational associations should not be read as proof of cause."),
    ("What is effect modification?", "train",
     "**Effect modification** (interaction) means the effect of an exposure differs across levels of another variable. For example, a treatment may reduce blood pressure more in older than in younger patients.\n\n"
     "It is different from confounding: confounding is a bias to remove, while effect modification is a real feature of the data to describe.\n\n"
     "How to assess it:\n- Estimate the exposure effect separately within each level of the modifier and compare the estimates and their confidence intervals.\n- Formally, test an interaction term in a regression model.\n\n"
     "Effect modification depends on the scale: it can be present on the ratio scale but not the difference scale. Pre-specify the modifiers you will examine, because many subgroup analyses produce chance findings. QuantAI's regression tool doesn't fit interaction terms yet, but you can run the model within each subgroup."),
    ("When should I use modified Poisson regression?", "train",
     "Use **modified Poisson regression** when the outcome is binary and you want **risk or prevalence ratios** rather than odds ratios, especially when the outcome is common (roughly 10% or more).\n\n"
     "How it works: a Poisson model with a log link is fitted to the 0/1 outcome, and **robust (sandwich) standard errors** correct the variance, which would otherwise be wrong for binary data. Exponentiated coefficients are adjusted prevalence or risk ratios.\n\n"
     "Why not logistic regression? With a common outcome, odds ratios are noticeably further from 1 than risk ratios and are often misread. Log-binomial regression estimates the same ratios but frequently fails to converge; modified Poisson is the usual robust alternative.\n\n"
     "In QuantAI, choose model 'modpoisson' in the regression tool or ask for prevalence ratios."),
    ("What is an offset in Poisson regression?", "train",
     "An **offset** lets a Poisson model compare **rates** when observation time or population size differs between rows.\n\n"
     "The model is log(expected count) = β0 + β1·X + … + log(exposure). The log of the exposure (person-years, bed-days, population) enters with its coefficient fixed at 1, so the model describes count ÷ exposure, and exp(β) are **rate ratios**.\n\n"
     "Key points:\n- The exposure must be strictly positive for every analysed row.\n- The offset is not a predictor: it is not estimated and should not also appear in the predictor list.\n- Check overdispersion (variance much larger than the mean). QuantAI switches to robust standard errors when the Pearson χ²/df is above 1.5; a negative binomial model is another common option.\n\n"
     "In QuantAI, set the exposure variable as the offset and choose Poisson regression."),
    ("What is a hazard ratio?", "train",
     "A **hazard ratio (HR)** compares the instantaneous rate of an event (death, cure, discharge) between groups over follow-up, usually from **Cox proportional hazards regression**.\n\n"
     "- HR = 1: no difference; HR = 2: at any time, those still at risk in one group experience the event at twice the rate; HR = 0.5: half the rate.\n- It is a ratio of rates, not of cumulative risks, and it is not the same as 'twice as likely to die by the end of the study'.\n\n"
     "The Cox model assumes **proportional hazards**: the HR is constant over time. Check it with Schoenfeld residuals or by looking at whether Kaplan-Meier curves cross. QuantAI assumes it rather than testing it; its exported Stata and R code include the check. Report HRs with 95% confidence intervals, the number of events, and how censoring was handled."),
    ("What is censoring in survival analysis?", "train",
     "**Censoring** happens when a participant's event time is not fully observed. Most common is **right censoring**: the person leaves the study, is lost to follow-up, or the study ends before the event occurs. We know only that the event had not happened by their last contact.\n\n"
     "Survival methods (Kaplan-Meier, log-rank test, Cox regression) use censored participants correctly: they count as at risk until they are censored and then leave the risk set. Dropping them, or treating them as event-free for the whole study, biases results.\n\n"
     "The key assumption is **non-informative censoring**: people censored have the same future risk as those who remain. If sicker people drop out more often, estimates will be biased. Report how many participants were censored and why."),
    ("What is a mixed model and the intraclass correlation?", "train",
     "A **mixed (multilevel) model** handles data where observations are grouped: pupils in schools, patients in clinics, repeated visits within a person. Members of a group tend to be more alike, so treating them as independent understates standard errors.\n\n"
     "A **random intercept** gives each cluster its own baseline level, drawn from a distribution; fixed effects describe the associations of interest.\n\n"
     "The **intraclass correlation (ICC)** is the share of total variance that lies between clusters. An ICC of 0.10 means 10% of the variation is between clusters. Even small ICCs matter when clusters are large, through the design effect 1 + (m − 1) × ICC.\n\n"
     "With fewer than about 10 to 30 clusters, between-cluster variance is estimated imprecisely. QuantAI fits linear mixed models with a random intercept for continuous outcomes."),
    ("What is a design effect?", "train",
     "The **design effect (DEFF)** is how much a complex sampling design inflates the variance compared with simple random sampling of the same size.\n\n"
     "For cluster sampling, DEFF ≈ 1 + (m − 1) × ICC, where m is the average cluster size and ICC the intraclass correlation. With 20 people per cluster and an ICC of 0.05, DEFF ≈ 1.95: you need about twice the sample to get the same precision.\n\n"
     "Uses:\n- **Planning**: multiply the simple-random-sample size by the DEFF, then allow for non-response.\n- **Analysis**: account for clustering (and weights or strata) so that confidence intervals are not too narrow.\n\n"
     "Multistage household surveys often assume a DEFF of 1.5 to 2 when no prior estimate exists."),
    ("Why does multiple testing matter?", "train",
     "Each test at a 5% significance level has a 5% chance of a false positive when there is truly no effect. Run 20 independent tests on null data and you expect about one 'significant' result by chance; the chance of at least one is about 64%.\n\n"
     "Ways to handle it:\n- **Pre-specify** a small number of primary hypotheses and label the rest as exploratory.\n- **Adjust**: Bonferroni (divide alpha by the number of tests) is simple but conservative; Holm is uniformly better; false discovery rate methods (Benjamini-Hochberg) suit many exploratory comparisons.\n- Report all tests you ran, not only significant ones.\n\n"
     "Emphasise effect sizes and confidence intervals, and treat isolated significant findings among many tests as hypotheses to confirm."),
    ("How do sample size and power work?", "train",
     "**Power** is the probability that a study detects an effect of a given size if it truly exists. Studies are usually planned for 80% or 90% power at a 5% significance level.\n\n"
     "The required sample size grows when:\n- the effect you want to detect is smaller,\n- the outcome is more variable (or rarer),\n- you want more power or a stricter significance level,\n- the design is clustered (multiply by the design effect),\n- you expect non-response or loss to follow-up (inflate accordingly).\n\n"
     "For surveys estimating a prevalence, the calculation instead targets precision: the margin of error around the estimate. Base assumptions on published data or a pilot study, and report every assumption so reviewers can check it. QuantAI's sample size calculator covers common designs."),
    ("How should I handle missing data?", "train",
     "Start by describing how much is missing for each variable and whether missingness relates to other characteristics.\n\n"
     "Missingness mechanisms:\n- **MCAR** (completely at random): unrelated to anything; complete-case analysis is unbiased but loses precision.\n- **MAR** (at random given observed data): related to recorded variables; multiple imputation or adjustment can reduce bias.\n- **MNAR** (not at random): related to the missing value itself; no standard method fully fixes it, so sensitivity analyses are needed.\n\n"
     "QuantAI analyses complete cases and reports how many rows each analysis excluded. If more than a few percent are missing on key variables, report it, compare people with and without missing data, and consider multiple imputation in Stata or R. Never fill gaps with means without saying so."),
    ("Should I use a t-test or a Mann-Whitney test?", "train",
     "Both compare a numeric outcome between two independent groups.\n\n"
     "- **Independent-samples t-test** compares means. It assumes roughly normal data in each group (less important when each group has more than about 30 observations) and is robust to unequal variances in Welch's version.\n- **Mann-Whitney U (Wilcoxon rank-sum)** compares ranks. It doesn't assume normality and suits skewed data, ordinal scores and small samples with outliers. It tests whether values in one group tend to be larger, not specifically a difference in means or medians.\n\n"
     "Choose based on the distribution and what you want to report: mean difference with CI (t-test) or medians with IQR (Mann-Whitney). QuantAI checks normality and variances and selects the test, reporting the rule it applied."),
    ("When should I use Fisher's exact test instead of chi-square?", "train",
     "Both test the association between two categorical variables.\n\n"
     "The **chi-square test** relies on a large-sample approximation that works when expected counts are big enough; a common rule is that no expected cell count is below 5 (for larger tables, no more than 20% below 5 and none below 1).\n\n"
     "**Fisher's exact test** calculates the probability exactly and is preferred when expected counts are small, typically in 2×2 tables with sparse cells.\n\n"
     "Check the expected counts, not the observed ones. With very sparse larger tables, consider combining categories that make sense together. QuantAI computes the expected counts and chooses between chi-square and Fisher's exact test automatically, reporting the rule it used."),
    ("How many events do I need for logistic regression?", "train",
     "A common rule of thumb is about **10 events per variable (EPV)**: for each model parameter, at least 10 participants in the less common outcome category. A categorical predictor with k levels uses k − 1 parameters.\n\n"
     "Example: with 60 cases of the outcome, aim for no more than about 6 parameters.\n\n"
     "With too few events, estimates are biased away from 1, confidence intervals are wide, and the model can fail to converge or show separation (a predictor perfectly predicting the outcome).\n\n"
     "Options when events are limited: choose predictors from prior knowledge rather than screening many, combine sparse categories, or use penalised methods in specialist software. EPV is a guide, not a guarantee; recent work suggests the right number depends on the setting. QuantAI reports EPV for logistic models."),
    ("What is multicollinearity?", "eval",
     "**Multicollinearity** means two or more predictors in a regression are strongly correlated, so the model struggles to separate their individual effects.\n\n"
     "Consequences: large standard errors, unstable coefficients that can change sign when a variable is added or removed, and non-significant predictors even when the model as a whole fits well. Predictions are usually unaffected.\n\n"
     "Detection: the **variance inflation factor (VIF)**; values above about 5 deserve attention and above 10 indicate serious collinearity.\n\n"
     "Remedies: drop or combine redundant variables (for example, keep either weight or BMI), create a score, or accept it if the collinear variables are only adjustment variables and not of direct interest. QuantAI reports VIFs for linear and logistic regression."),
    ("What is the proportional odds assumption?", "eval",
     "**Ordinal logistic regression** (the proportional odds model) handles ordered outcomes such as poor / fair / good. It models the odds of being at or below each category versus above it.\n\n"
     "The **proportional odds assumption** says each predictor has the same odds ratio at every cut-point: the effect of age on being 'fair or better' versus 'poor' equals its effect on 'good' versus 'fair or poor'. This gives one odds ratio per predictor.\n\n"
     "If the assumption fails for a predictor, its single odds ratio misrepresents the data. Options include a partial proportional odds model, a multinomial model (which ignores the ordering), or collapsing the outcome into a binary one when that is scientifically sensible. QuantAI assumes proportional odds rather than testing it; the exported Stata code (brant test) or R code can check it."),
    ("What does Kaplan-Meier show?", "eval",
     "A **Kaplan-Meier curve** estimates the probability of remaining event-free over time, accounting for censored participants.\n\n"
     "At each event time, the survival probability is multiplied by the proportion of those still at risk who did not have the event. Censored people count as at risk until they leave. Steps down mark events; tick marks often show censoring.\n\n"
     "Reading it:\n- The **median survival time** is where the curve crosses 0.5, if it does.\n- Comparing curves between groups uses the **log-rank test**.\n- Show the number at risk under the plot; the right tail is based on few people and is imprecise.\n\n"
     "Kaplan-Meier is unadjusted. To adjust for other variables, use Cox regression."),
]


def deeper_examples(split: str) -> list[dict]:
    return [{"intent": "concept_deeper", "message": "In more depth: " + q, "reply": a, "action": None, "topic": q}
            for q, s, a in DEEPER if s == split]
