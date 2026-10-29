"""Intents, wording templates and label rules for QuantAI v4 data.

Every intent builds one example: a user message, the expected action (with
variable placeholders resolved after rendering), a reply, and any earlier
turns or analyses. Labels follow the action contract in prompting.py and the
behaviour of QuantAI's engine, and every non-null action is executed in the
real engine before it is kept (see build.py).

Templates are split: TRAIN wording is used only for training rows and EVAL
wording only for held-out rows, so evaluation measures understanding of new
phrasings, not recall.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Callable

from scenarios import Scenario, Var

PH = "@@{}@@"           # variable placeholder, replaced by QuantAI's name in code
ALLVARS = "@@__ALL__@@"  # every variable shown in the context


def ph(v: Var) -> str:
    return PH.format(v.key)


def _reserved() -> set[str]:
    """QuantAI's reserved names, read from the app source so the two cannot drift."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "assets" / "js" / "quantai.js").read_text(encoding="utf-8")
    block = re.search(r"const RESERVED = new Set\(\[(.*?)\]\);", src, re.S).group(1)
    return set(re.findall(r'"([^"]*)"', block))


RESERVED = _reserved()


def code_name(header: str) -> str:
    """Python copy of QuantAI's cleanNames() (checked against the app after rendering)."""
    import unicodedata
    s = unicodedata.normalize("NFKD", header)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_") or "var"
    if s[0].isdigit():
        s = "v" + s
    s = s[:28].rstrip("_")
    if s in RESERVED:
        s = s + "_v"
    return s


def join(items: list[str], rng: random.Random) -> str:
    if len(items) == 1:
        return items[0]
    sep = rng.choice([" and ", " and ", ", and ", " & "]) if len(items) > 1 else ""
    if sep == ", and " and len(items) == 2:
        sep = " and "
    return ", ".join(items[:-1]) + sep + items[-1]


class Ctx:
    """Helpers for one example: a scenario, a split and a random generator."""

    def __init__(self, sc: Scenario, split: str, rng: random.Random):
        self.sc, self.split, self.rng = sc, split, rng

    def t(self, templates: dict[str, list[str]]) -> str:
        return self.rng.choice(templates[self.split])

    def m(self, v: Var) -> str:
        """How the user names the variable: a natural mention, sometimes the name in code."""
        if self.rng.random() < 0.15:
            return v.key if self.sc.builtin else code_name(v.header)
        return self.rng.choice(v.mentions)

    def ms(self, vs: list[Var]) -> str:
        return join([self.m(v) for v in vs], self.rng)

    def lab(self, v: Var) -> str:
        return v.header

    def labs(self, vs: list[Var]) -> str:
        return join([v.header for v in vs], self.rng)

    def pick(self, kinds: tuple[str, ...], exclude: list[Var] = (), k: int = 1) -> list[Var] | None:
        pool = [v for v in self.sc.by_kind(*kinds) if v not in exclude]
        if len(pool) < k:
            return None
        return self.rng.sample(pool, k)

    def out(self, kinds: tuple[str, ...], exclude: list[Var] = (), k: int = 1) -> list[Var] | None:
        """Pick plausible outcome variable(s): never a demographic or structural variable."""
        pool = [v for v in self.sc.by_kind(*kinds) if v not in exclude and v.key not in XONLY]
        if len(pool) < k:
            return None
        return self.rng.sample(pool, k)

    def exposure(self, exclude: list[Var]) -> Var | None:
        """A substantive exposure for causal or invented-result wording (not age or sex) when one exists."""
        pool = [v for v in self.sc.by_kind("cont", "bin", "cat", "ord", "count") if v not in exclude and v.key not in DEMOG]
        return self.rng.choice(pool) if pool else None

    def predictors(self, exclude: list[Var], lo: int = 1, hi: int = 3) -> list[Var] | None:
        pool = [v for v in self.sc.by_kind("age", "cont", "bin", "cat", "ord", "count")
                if v not in exclude]
        if not pool:
            return None
        return self.rng.sample(pool, min(len(pool), self.rng.randint(lo, hi)))

    def choice(self, items):
        return self.rng.choice(items)


def reg(y: Var, xs: list[Var], model: str, cluster: Var | None = None, offset: Var | None = None) -> dict:
    return {"type": "regression", "outcome": ph(y), "predictors": [ph(x) for x in xs], "model": model,
            "cluster": ph(cluster) if cluster else None, "offset": ph(offset) if offset else None}


def example(intent: str, message: str, reply: str, action: Any, **extra) -> dict:
    return {"intent": intent, "message": message, "reply": reply, "action": action, **extra}


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


# Variables that are predictors or structure in these studies, never outcomes
# (nobody models sex or district as the outcome of a public health study).
XONLY = {"sex", "district", "region", "residence", "ward", "cls", "shift", "vehicle", "occupation", "marital", "ftype", "job",
         "water", "mode", "arm", "regimen", "stage", "place", "destination", "treatment", "trimester", "income", "edu", "medu",
         "hiv", "diabetes", "diabetic", "alone", "smoke", "daycare", "swims", "pzq", "hu", "education_level", "employment"}
DEMOG = {"sex", "age", "agem", "mage", "age_years"}

OUTCOME_KINDS = {"bin": "binary", "count": "a count", "cat": "categorical", "ord": "ordinal", "cont": "continuous", "age": "continuous", "event": "binary"}

# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
T_DESC_SOME = {
    "train": ["Give me descriptive statistics for {vars}.", "Summarise {vars}.", "Can you describe {vars} for me?",
              "make a summary table of {vars}", "I need the descriptives for {vars}.", "Describe {vars} in this sample."],
    "eval": ["What do {vars} look like in this sample?", "Produce a descriptive table covering {vars}.",
             "quick descriptives on {vars} please"],
}
T_DESC_BY = {
    "train": ["Make a table of {vars} by {g}.", "Summarise {vars} stratified by {g}.", "Table 1 of {vars}, split by {g}",
              "Describe {vars} for each {g} group.", "Compare {vars} across {g} in a descriptive table."],
    "eval": ["Break {vars} down by {g} in a descriptive table.", "Show {vars} separately for each {g} category."],
}
T_DESC_ALL = {
    "train": ["Make a Table 1 by {g}.", "Create a descriptive table of all the variables, grouped by {g}.",
              "table 1 stratified by {g} please", "Describe the whole sample by {g}."],
    "eval": ["Build Table 1 with {g} as the columns.", "Give me baseline characteristics of everyone, split by {g}."],
}
T_DESC_ALL_NOGROUP = {
    "train": ["Describe all the variables.", "Give me a summary of every variable in the dataset.", "Make a Table 1 for the whole sample."],
    "eval": ["Summarise the full dataset for me."],
}
R_DESC = ["I'll summarise {labs} in a descriptive table; QuantAI chooses mean (SD) or median (IQR) from each variable's distribution.",
          "I'll build a descriptive table of {labs}. Continuous variables get mean (SD) or median (IQR) depending on their distribution, and categorical ones counts and percentages."]
R_DESC_BY = ["I'll summarise {labs} by {g}, with the group comparison tests QuantAI selects from each variable's type and distribution.",
             "I'll make a descriptive table of {labs} split by {g}; QuantAI adds the appropriate test for each variable."]


def desc_some(c: Ctx):
    vs = c.pick(("age", "cont", "bin", "cat", "ord", "count"), k=c.rng.randint(2, 4))
    if not vs:
        return None
    return example("describe_some", c.t(T_DESC_SOME).format(vars=c.ms(vs)), c.choice(R_DESC).format(labs=c.labs(vs)),
                   {"type": "describe", "vars": [ph(v) for v in vs], "group": None})


def desc_by(c: Ctx):
    g = c.pick(("bin", "cat"))
    if not g:
        return None
    g = g[0]
    vs = c.pick(("age", "cont", "bin", "cat", "ord", "count"), exclude=[g], k=c.rng.randint(2, 3))
    if not vs:
        return None
    # The local API always adds the grouping variable to vars (normalise_action_item), so labels do too.
    return example("describe_by", c.t(T_DESC_BY).format(vars=c.ms(vs), g=c.m(g)), c.choice(R_DESC_BY).format(labs=c.labs(vs), g=g.header),
                   {"type": "describe", "vars": [ph(v) for v in vs] + [ph(g)], "group": ph(g)})


def desc_all(c: Ctx):
    if c.rng.random() < 0.3:
        return example("describe_all", c.t(T_DESC_ALL_NOGROUP),
                       "I'll make a descriptive table of every variable in the dataset; QuantAI chooses the summary for each variable from its type and distribution.",
                       {"type": "describe", "vars": [ALLVARS], "group": None})
    g = c.pick(("bin", "cat"))
    if not g:
        return None
    g = g[0]
    return example("describe_all", c.t(T_DESC_ALL).format(g=c.m(g)),
                   f"I'll build Table 1 for all variables by {g.header}, with the group comparison QuantAI selects for each variable.",
                   {"type": "describe", "vars": [ALLVARS], "group": ph(g)})


# ---------------------------------------------------------------------------
# Comparisons and correlation
# ---------------------------------------------------------------------------
T_CMP = {
    "train": ["Compare {y} between {g} groups.", "Is there a difference in {y} by {g}?", "Does {y} differ across {g}?",
              "Test whether {y} is associated with {g}.", "Is {g} associated with {y}?", "compare {y} by {g}",
              "Is {y} different for different {g} groups?", "Run a test comparing {y} across {g}."],
    "eval": ["Are {y} values different depending on {g}?", "Check if {g} makes a difference to {y}.",
             "Run the appropriate test of {y} across {g} categories.", "Does {g} relate to {y}?"],
}
R_CMP = ["I'll compare {Y} across {G} groups. QuantAI checks the assumptions (normality, equal variances or expected counts) and selects the test.",
         "I'll test whether {Y} differs by {G}; the engine chooses the appropriate test from the variable types and assumption checks.",
         "I'll run a comparison of {Y} by {G} and let QuantAI's assumption checks decide between parametric and non-parametric tests."]
T_PAIRED = {
    "train": ["Compare {a} and {b} in the same participants.", "Test whether {a} and {b} differ (they are repeated measurements).",
              "Did the value change from {a} to {b}? Use a paired test.", "Is there a significant change between {a} and {b}?"],
    "eval": ["Was there a change between {a} and {b} in the same people?", "paired comparison of {a} vs {b}"],
}
T_CORR = {
    "train": ["What is the correlation between {a} and {b}?", "Is {a} correlated with {b}?", "Check the relationship between {a} and {b}.",
              "Correlate {a} with {b}."],
    "eval": ["How strongly are {a} and {b} related?", "Is there a linear relationship between {a} and {b}?"],
}


def compare(c: Ctx):
    g = c.pick(("bin", "cat"))
    if not g:
        return None
    g = g[0]
    y = c.out(("age", "cont", "bin", "ord", "count", "cat"), exclude=[g])
    if not y:
        return None
    y = y[0]
    return example("compare", c.t(T_CMP).format(y=c.m(y), g=c.m(g)), c.choice(R_CMP).format(Y=y.header, G=g.header),
                   {"type": "compare", "outcome": ph(y), "exposure": ph(g), "paired": False})


def compare_paired(c: Ctx):
    if not c.sc.paired:
        return None
    a, b = (c.sc.var(k) for k in c.sc.paired[0])
    return example("compare_paired", c.t(T_PAIRED).format(a=c.m(a), b=c.m(b)),
                   f"{a.header} and {b.header} are repeated measurements on the same people, so I'll use a paired comparison; QuantAI chooses a paired t-test or Wilcoxon signed-rank test from the differences.",
                   {"type": "compare", "outcome": ph(b), "exposure": ph(a), "paired": True})


def correlation(c: Ctx):
    vs = c.pick(("age", "cont"), k=2)
    if not vs:
        return None
    a, b = vs
    return example("correlation", c.t(T_CORR).format(a=c.m(a), b=c.m(b)),
                   f"I'll assess the correlation between {a.header} and {b.header}; QuantAI chooses Pearson or Spearman from the distributions.",
                   {"type": "compare", "outcome": ph(a), "exposure": ph(b), "paired": False}, symmetric=True)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
T_LINEAR = {
    "train": ["Run a linear regression of {y} on {xs}.", "Fit a linear model for {y} with {xs} as predictors.",
              "Use multiple linear regression to see how {xs} relate to {y}.", "regress {y} on {xs}",
              "Linear regression: {y} with {xs}."],
    "eval": ["I want an OLS model with {y} explained by {xs}.", "Model {y} as a function of {xs} using linear regression."],
}
T_AUTO = {
    "train": ["What factors are associated with {y}? Include {xs}.", "Which of {xs} are associated with {y} after adjusting for each other?",
              "Model {y} using {xs}.", "Fit a regression for {y} with {xs}.", "Run a multivariable regression of {y} on {xs}.",
              "Is {y} associated with {xs}? Adjust for all of them together."],
    "eval": ["Assess the adjusted associations of {xs} with {y}.", "Set up a regression with {y} as the outcome and {xs} as the explanatory variables."],
}
T_LOGIT = {
    "train": ["Estimate odds ratios for {y} with {xs}.", "Run a logistic regression of {y} on {xs}.", "What are the adjusted odds of {y} by {xs}?",
              "logistic model for {y} with {xs}", "Fit binary logistic regression: outcome {y}, predictors {xs}."],
    "eval": ["Give me adjusted ORs for {y} using {xs}.", "Fit a logit model predicting {y} from {xs}."],
}
T_MODP = {
    "train": ["Estimate adjusted prevalence ratios for {y} by {xs}.", "I want prevalence ratios, not odds ratios, for {y} with {xs}.",
              "Fit a modified Poisson regression for {y} using {xs}.", "What are the risk ratios for {y} associated with {xs}?",
              "{Y} is common, so give me prevalence ratios for {xs} instead of odds ratios.", "Estimate relative risks of {y} by {xs}.",
              "Use modified Poisson to get PRs for {y} with {xs}."],
    "eval": ["Report adjusted PRs for {y} across {xs}.", "How much more prevalent is {y} by {xs}? Use prevalence ratios.",
             "Get me risk ratios rather than ORs for {y}, adjusting for {xs}."],
}
T_POIS = {
    "train": ["Fit a Poisson regression for {y} with {xs}.", "Model the number of {y} using {xs}.", "Run a count model for {y} with {xs}.",
              "Poisson model: {y} by {xs}."],
    "eval": ["Which of {xs} predict higher counts of {y}? Use Poisson regression."],
}
T_RATE = {
    "train": ["Estimate the rate of {y} per {e} by {xs}.", "Model {y} rates, using {e} as the exposure, with {xs}.",
              "Fit a Poisson model for {y} with log {e} as an offset, adjusting for {xs}.",
              "What is the incidence rate ratio of {y} for {xs}, accounting for {e}?",
              "Do {y} rates differ by {x1}? Account for {e}{adj}.", "Compare {y} rates per {e} between {x1} groups{adj}.",
              "Is {x1} associated with the rate of {y} per {e}{adj}?", "Compare {y} per {e} by {x1}{adj}.",
              "{Y} per {e}: how does it vary with {xs}?", "Rate ratios for {y} by {xs} please, with {e} as person-time."],
    "eval": ["Are rates of {y} per {e} different across {x1}{adj}?", "I need rate ratios for {y} by {xs}, using {e} as the denominator.",
             "Does the {y} rate per {e} change with {x1}{adj}?", "Compare {y} per {e} across levels of {x1}{adj}.",
             "Does {x1} make a difference to how often {y} happen per {e}{adj}?"],
}
T_ORD = {
    "train": ["Fit an ordinal logistic regression for {y} with {xs}.", "Use a proportional odds model for {y} by {xs}.",
              "ordinal regression of {y} on {xs}"],
    "eval": ["Run ordered logistic regression of {y} on {xs}."],
}
T_MULTI = {
    "train": ["Fit a multinomial logistic regression for {y} using {xs}.", "Model the {y} categories with multinomial regression and {xs}."],
    "eval": ["Use a multinomial model to see how {xs} relate to {y}."],
}
T_MIXED = {
    "train": ["Fit a mixed model for {y} with {xs} and a random intercept for {c}.", "Model {y} by {xs}, accounting for clustering within {c}.",
              "Participants are nested in {c}; model {y} with {xs}.", "Account for {c} clustering in a regression of {y} on {xs}.",
              "multilevel model of {y} on {xs}, clustered by {c}"],
    "eval": ["Run a hierarchical model of {y} on {xs} with {c} as the grouping level.",
             "Observations are grouped by {c}, so adjust for that: {y} explained by {xs}."],
}


def _adj(c: Ctx, rest: list[Var]) -> str:
    if not rest:
        return ""
    return c.choice([", adjusting for {}", ", controlling for {}", " after adjusting for {}"]).format(c.ms(rest))


def reg_linear(c: Ctx):
    y = c.out(("cont",))
    if not y:
        return None
    y = y[0]
    xs = c.predictors([y])
    if not xs:
        return None
    return example("reg_linear", c.t(T_LINEAR).format(y=c.m(y), xs=c.ms(xs)),
                   f"I'll fit multiple linear regression of {y.header} on {c.labs(xs)}; the engine reports the coefficients with confidence intervals and runs its regression diagnostics.",
                   reg(y, xs, "linear"))


def reg_auto(c: Ctx):
    y = c.out(("cont", "bin", "count", "ord", "cat"))
    if not y:
        return None
    y = y[0]
    xs = c.predictors([y], 2, 3)
    if not xs:
        return None
    measure = {"cont": "linear regression", "age": "linear regression", "bin": "logistic regression", "count": "Poisson regression",
               "ord": "ordinal logistic regression", "cat": "multinomial logistic regression"}[y.kind]
    return example("reg_auto", c.t(T_AUTO).format(y=c.m(y), xs=c.ms(xs)),
                   f"I'll fit a multivariable model for {y.header} with {c.labs(xs)}. You didn't name a model, so QuantAI chooses from the outcome type ({OUTCOME_KINDS[y.kind]} → {measure}).",
                   reg(y, xs, "auto"))


def reg_logistic(c: Ctx):
    y = c.out(("bin",))
    if not y:
        return None
    y = y[0]
    xs = c.predictors([y])
    if not xs:
        return None
    return example("reg_logistic", c.t(T_LOGIT).format(y=c.m(y), xs=c.ms(xs)),
                   f"I'll fit binary logistic regression for {y.header} with {c.labs(xs)} and report adjusted odds ratios. Note that odds ratios overstate risk ratios when the outcome is common.",
                   reg(y, xs, "logistic"))


def reg_modpoisson(c: Ctx):
    y = c.out(("bin",))
    if not y:
        return None
    y = y[0]
    xs = c.predictors([y])
    if not xs:
        return None
    msg = c.t(T_MODP).format(y=c.m(y), Y=cap(c.m(y)), xs=c.ms(xs))
    return example("reg_modpoisson", msg,
                   f"I'll fit modified Poisson regression (log link with robust standard errors) for {y.header} with {c.labs(xs)}, so the estimates are adjusted prevalence ratios rather than odds ratios.",
                   reg(y, xs, "modpoisson"))


def reg_poisson(c: Ctx):
    rate_counts = {k for k, _ in c.sc.rates}
    pool = [v for v in c.sc.by_kind("count") if v.key not in rate_counts and v.key not in XONLY]
    if not pool:
        return None
    y = c.choice(pool)
    xs = c.predictors([y])
    if not xs:
        return None
    return example("reg_poisson", c.t(T_POIS).format(y=c.m(y), xs=c.ms(xs)),
                   f"I'll fit Poisson regression for the count {y.header} with {c.labs(xs)}; the engine reports incidence rate ratios and switches to robust standard errors if it detects overdispersion.",
                   reg(y, xs, "poisson"))


def reg_rate(c: Ctx):
    if not c.sc.rates:
        return None
    yk, ek = c.choice(c.sc.rates)
    y, e = c.sc.var(yk), c.sc.var(ek)
    tpl = c.t(T_RATE)
    if "{x1}" in tpl:
        # wording about groups or levels needs a categorical variable in that slot
        groups = [v for v in c.sc.by_kind("bin", "cat", "ord") if v not in (y, e)]
        if not groups:
            return None
        x1 = c.choice(groups)
        rest = c.predictors([y, e, x1], 1, 2) if c.rng.random() < 0.6 else []
        xs = [x1] + (rest or [])
    else:
        xs = c.predictors([y, e])
        if not xs:
            return None
    msg = tpl.format(y=c.m(y), Y=cap(c.m(y)), e=c.m(e), xs=c.ms(xs), x1=c.m(xs[0]), adj=_adj(c, xs[1:]))
    return example("reg_rate", msg,
                   f"I'll fit Poisson regression of {y.header} on {c.labs(xs)} with log({e.header}) as the offset, so the estimates are rate ratios per unit of {e.header}. {e.header} is the exposure offset, not a predictor.",
                   reg(y, xs, "poisson", offset=e))


def reg_ordinal(c: Ctx):
    y = c.out(("ord",))
    if not y:
        return None
    y = y[0]
    xs = c.predictors([y])
    if not xs:
        return None
    return example("reg_ordinal", c.t(T_ORD).format(y=c.m(y), xs=c.ms(xs)),
                   f"{y.header} is ordered, so I'll fit ordinal (proportional odds) logistic regression with {c.labs(xs)}. QuantAI assumes proportional odds rather than testing it; the exported Stata or R code can check it.",
                   reg(y, xs, "ordinal"))


def reg_multinomial(c: Ctx):
    pool = [v for v in c.sc.by_kind("cat") if v.key not in XONLY and (len(v.params.get("levels", [])) >= 3 or c.sc.builtin)]
    if not pool:
        return None
    y = c.choice(pool)
    xs = c.predictors([y])
    if not xs:
        return None
    return example("reg_multinomial", c.t(T_MULTI).format(y=c.m(y), xs=c.ms(xs)),
                   f"{y.header} has unordered categories, so I'll fit multinomial logistic regression with {c.labs(xs)}; the engine compares each category with the reference.",
                   reg(y, xs, "multinomial"))


def reg_mixed(c: Ctx):
    cl = c.pick(("clus",))
    y = c.out(("cont",))
    if not cl or not y:
        return None
    cl, y = cl[0], y[0]
    xs = c.predictors([y, cl])
    if not xs:
        return None
    return example("reg_mixed", c.t(T_MIXED).format(y=c.m(y), xs=c.ms(xs), c=c.m(cl)),
                   f"I'll fit a linear mixed model for {y.header} with {c.labs(xs)} and a random intercept for {cl.header}, which accounts for observations clustered within {cl.header}. The engine reports the intraclass correlation.",
                   reg(y, xs, "mixed", cluster=cl))


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------
T_SURV_G = {
    "train": ["Plot Kaplan-Meier curves for {d} by {g}.", "Compare survival between {g} groups using {t} and {d}.",
              "Run a log-rank test of time to {d} by {g}.", "Is time to {d} different by {g}?", "Kaplan-Meier analysis of {d}, comparing {g}."],
    "eval": ["Show survival curves for {d} split by {g}.", "Does {g} affect how long it takes until {d}?"],
}
T_SURV_COX = {
    "train": ["Run a Cox regression for time to {d} with {xs}.", "What are the hazard ratios for {d} by {xs}?",
              "Fit a Cox model of {t} and {d} adjusting for {xs}.", "Cox proportional hazards model for {d} including {xs}."],
    "eval": ["Estimate adjusted hazard ratios for {d} using {xs}."],
}
T_SURV_CMP = {
    "train": ["Compare time to {d} between {g} groups and adjust for {xs}.", "Compare {t} by {g}, adjusting for {xs}.",
              "Does time to {d} differ by {g} after adjusting for {xs}?", "Compare how long until {d} across {g}, controlling for {xs}."],
    "eval": ["Is the time until {d} different for {g} groups once {xs} are taken into account?",
             "Compare {d} timing across {g} with adjustment for {xs}."],
}


def _surv(c: Ctx):
    if not c.sc.survival:
        return None
    tk, dk = c.choice(c.sc.survival)
    return c.sc.var(tk), c.sc.var(dk)


def surv_group(c: Ctx):
    s = _surv(c)
    g = c.pick(("bin", "cat"))
    if not s or not g:
        return None
    (t, d), g = s, g[0]
    return example("surv_group", c.t(T_SURV_G).format(t=c.m(t), d=c.m(d), g=c.m(g)),
                   f"This is a time-to-event question, so I'll run a survival analysis of {t.header} with {d.header} as the event, comparing {g.header} groups with Kaplan-Meier curves and a log-rank test.",
                   {"type": "survival", "time": ph(t), "event": ph(d), "group": ph(g), "covariates": []})


def surv_cox(c: Ctx):
    s = _surv(c)
    if not s:
        return None
    t, d = s
    xs = c.predictors([t, d])
    if not xs:
        return None
    return example("surv_cox", c.t(T_SURV_COX).format(t=c.m(t), d=c.m(d), xs=c.ms(xs)),
                   f"I'll fit Cox proportional hazards regression for {t.header} with {d.header} as the event and {c.labs(xs)} as covariates; QuantAI reports hazard ratios, and the exported code lets you check the proportional hazards assumption.",
                   {"type": "survival", "time": ph(t), "event": ph(d), "group": None, "covariates": [ph(x) for x in xs]})


def surv_compare(c: Ctx):
    s = _surv(c)
    g = c.pick(("bin", "cat"))
    if not s or not g:
        return None
    (t, d), g = s, g[0]
    xs = c.predictors([t, d, g], 1, 2)
    if not xs:
        return None
    return example("surv_compare", c.t(T_SURV_CMP).format(t=c.m(t), d=c.m(d), g=c.m(g), xs=c.ms(xs)),
                   f"Although you said compare, this is time-to-event data, so I'll run a survival analysis of {t.header} with {d.header} as the event by {g.header}, with Cox regression adjusting for {c.labs(xs)}.",
                   {"type": "survival", "time": ph(t), "event": ph(d), "group": ph(g), "covariates": [ph(x) for x in xs]})


# ---------------------------------------------------------------------------
# Forecasting, data check, fixes, report, navigation
# ---------------------------------------------------------------------------
NUMWORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight", 10: "ten", 12: "twelve"}
T_FC = {
    "train": ["Forecast {v} for the next {h} {unit}s.", "Predict {v} over the next {h} periods.", "What will {v} look like for the next {hw} {unit}s?",
              "forecast {v} {h} steps ahead", "Project {v} forward {h} {unit}s."],
    "eval": ["Give me a {h}-{unit} forecast of {v}.", "Where is {v} heading over the next {hw} {unit}s?"],
}


def forecast(c: Ctx):
    if not c.sc.series:
        return None
    date = c.sc.variables[0]
    v = c.choice([x for x in c.sc.variables if x.kind == "series"])
    unit = date.params["unit"]
    h = c.choice([2, 3, 4, 6, 8, 12] if unit != "quarter" else [2, 3, 4, 6, 8])
    msg = c.t(T_FC).format(v=c.m(v), h=h, hw=NUMWORDS.get(h, str(h)), unit=unit)
    shown = str(h) if str(h) in msg else NUMWORDS.get(h, str(h))  # mirror the user's wording
    return example("forecast", msg,
                   f"I'll forecast {v.header} for the next {shown} {unit}s with Holt-Winters exponential smoothing; the prediction intervals widen further ahead, so treat later periods with more caution.",
                   {"type": "forecast", "date": "@@__DATE__@@", "value": ph(v), "horizon": h}, date_key=date.key)


T_CHECK = {"train": ["Check my data for problems.", "Run a data quality check.", "Are there any issues with this dataset?",
                     "Look for missing codes, duplicates or impossible values.", "check the file before I start"],
           "eval": ["Can you screen the file for data errors before I analyse it?"]}
T_FIX = {"train": ["Apply the suggested fixes.", "Fix the problems the data check found.", "Go ahead and clean those issues.", "apply the fixes"],
         "eval": ["Please correct the flagged data problems."]}
T_REPORT = {"train": ["Create a report of my analyses.", "Export everything as a report.", "Download a report I can share.", "make me a report"],
            "eval": ["Put my results into an HTML report."]}


def check(c: Ctx):
    return example("check", c.t(T_CHECK), c.choice([
        "I'll run QuantAI's data check for missing-value codes, duplicate rows, impossible values and variable types before any analysis.",
        "I'll run the built-in data quality check so you can review missing codes, duplicates and implausible values first."]), {"type": "check"})


def fix(c: Ctx):
    return example("fix", c.t(T_FIX), "I'll apply QuantAI's suggested fixes. Review each change, because a flagged value can be a data-entry error or a real but unusual case.",
                   {"type": "fix"}, issues=True)


def report(c: Ctx):
    pre = _a_pre(c)
    if not pre:
        return None
    return example("report", c.t(T_REPORT), "I'll generate an HTML report of the analyses you have run; check the methods, sample sizes and assumptions before you use it.",
                   {"type": "report"}, pre_actions=[pre])


OPEN = {
    "data": {"train": ["Show me the variables tab.", "Open the data page.", "Where can I change a variable's type?"], "eval": ["Take me to the data and variables view."]},
    "describe": {"train": ["Open the Table 1 tool.", "Go to the describe tool."], "eval": ["Take me to the descriptive statistics page."]},
    "compare": {"train": ["Open compare and relate.", "Open the comparison tool so I can pick the variables myself."], "eval": ["Take me to the group comparison tool."]},
    "regression": {"train": ["Open the regression panel.", "Go to the regression tool."], "eval": ["I want to set up a regression myself, open it."]},
    "survival": {"train": ["Open the survival analysis tool.", "Go to Kaplan-Meier."], "eval": ["Show me the survival page."]},
    "forecast": {"train": ["Open forecasting.", "Go to the forecast tool."], "eval": ["Take me to the time-series forecasting page."]},
    "export": {"train": ["I want to export Stata code for my analyses.", "Where can I download the R script?", "Open log and export."], "eval": ["Get me the SPSS syntax for what I ran."]},
    "question": {"train": ["Open the research question builder.", "Help me write a research question with PECO."], "eval": ["I need to frame my research question using PICO."]},
    "design": {"train": ["Help me choose a study design.", "Open the study design tool."], "eval": ["Which study design should I use? Open the advisor."]},
    "sample": {"train": ["Open the sample size calculator.", "How many participants do I need? Open the calculator."], "eval": ["I need to work out my sample size."]},
    "test": {"train": ["Which statistical test should I use? Open the test selector.", "Open choose a test."], "eval": ["Take me to the choose-a-test helper."]},
    "checklist": {"train": ["Open the STROBE checklist.", "I need a reporting checklist."], "eval": ["Show me the CONSORT checklist."]},
}
OPEN_LABEL = {"data": "Data & variables", "describe": "Describe (Table 1)", "compare": "Compare & relate", "regression": "Regression",
              "survival": "Survival", "forecast": "Forecast", "export": "Log & export", "question": "Research question",
              "design": "Study design", "sample": "Sample size", "test": "Choose a test", "checklist": "Reporting checklist"}


def open_tool(c: Ctx):
    tool = c.choice(sorted(OPEN))
    return example("open", c.choice(OPEN[tool][c.split]), f"I'll open the {OPEN_LABEL[tool]} tool.", {"type": "open", "tool": tool})


# ---------------------------------------------------------------------------
# Several actions, and follow-up turns
# ---------------------------------------------------------------------------
T_MULTI2 = {
    "train": ["Check the data, then make a Table 1 by {g}.", "Summarise {vars} and then compare {y} by {g}.",
              "Run the data check and then fit a logistic regression of {yb} on {xs}.", "Compare {y} by {g} and also {y2} by {g}."],
    "eval": ["First check for data problems, then describe everything by {g}.", "Give me descriptives for {vars}, then test whether {y} differs by {g}."],
}


def multi(c: Ctx):
    g = c.pick(("bin", "cat"))
    if not g:
        return None
    g = g[0]
    tpl = c.t(T_MULTI2)
    vs = c.pick(("age", "cont", "ord", "count"), exclude=[g], k=2) or []
    ys = c.out(("age", "cont", "ord", "count", "bin"), exclude=[g], k=2)
    if not ys:
        return None
    y, y2 = ys
    yb = c.out(("bin",), exclude=[g])
    if "{yb}" in tpl:
        if not yb:
            return None
        yb = yb[0]
        xs = c.predictors([yb, g], 1, 2)
        if not xs:
            return None
        return example("multi", tpl.format(yb=c.m(yb), xs=c.ms(xs)),
                       f"I'll run the data check first, then fit logistic regression for {yb.header} with {c.labs(xs)}.",
                       [{"type": "check"}, reg(yb, xs, "logistic")])
    if tpl.startswith(("Check the data", "First check")):
        return example("multi", tpl.format(g=c.m(g)), f"I'll run the data check, then build Table 1 by {g.header}.",
                       [{"type": "check"}, {"type": "describe", "vars": [ALLVARS], "group": ph(g)}])
    if "{y2}" in tpl:
        return example("multi", tpl.format(y=c.m(y), y2=c.m(y2), g=c.m(g)), f"I'll compare {y.header} by {g.header}, then {y2.header} by {g.header}.",
                       [{"type": "compare", "outcome": ph(y), "exposure": ph(g), "paired": False},
                        {"type": "compare", "outcome": ph(y2), "exposure": ph(g), "paired": False}])
    if not vs:
        return None
    return example("multi", tpl.format(vars=c.ms(vs), y=c.m(y), g=c.m(g)),
                   f"I'll summarise {c.labs(vs)}, then compare {y.header} by {g.header}.",
                   [{"type": "describe", "vars": [ph(v) for v in vs], "group": None},
                    {"type": "compare", "outcome": ph(y), "exposure": ph(g), "paired": False}])


T_FU_ADD = {"train": ["Also adjust for {x}.", "Add {x} to the model.", "can you include {x} too"], "eval": ["Include {x} as well and rerun."]}
T_FU_DROP = {"train": ["Now remove {x}.", "Drop {x} from the model."], "eval": ["Take {x} out and rerun it."]}
T_FU_PR = {"train": ["Can you give prevalence ratios instead?", "I'd rather have prevalence ratios."], "eval": ["Switch to prevalence ratios."]}
T_FU_OR = {"train": ["Same thing but odds ratios.", "Use logistic regression instead."], "eval": ["Give me odds ratios for that instead."]}
T_FU_REGROUP = {"train": ["Do it by {g} instead.", "Now split it by {g}."], "eval": ["Group it by {g} this time."]}
T_FU_SURV_ADD = {"train": ["Adjust that for {x} as well."], "eval": ["Add {x} as a covariate."]}


def followup(c: Ctx):
    """A second turn that modifies the previous analysis; the first is in history and already run."""
    kind = c.choice(["reg", "reg", "reg", "desc", "surv"])
    if kind == "reg":
        y = c.out(("bin",))
        if not y:
            return None
        y = y[0]
        xs = c.predictors([y], 2, 3)
        if not xs or len(xs) < 2:
            return None
        first_model = c.choice(["logistic", "modpoisson"])
        prev = reg(y, xs, first_model)
        prev_msg = (f"Run a logistic regression of {c.m(y)} on {c.ms(xs)}." if first_model == "logistic"
                    else f"Estimate prevalence ratios for {c.m(y)} by {c.ms(xs)}.")
        prev_reply = f"I'll fit {'logistic' if first_model == 'logistic' else 'modified Poisson'} regression for {y.header} with {c.labs(xs)}."
        op = c.choice(["add", "drop", "switch"])
        if op == "add":
            extra = c.predictors([y] + xs, 1, 1)
            if not extra:
                return None
            msg = c.t(T_FU_ADD).format(x=c.m(extra[0]))
            new = reg(y, xs + extra, first_model)
            reply = f"I'll refit the same {('logistic' if first_model == 'logistic' else 'modified Poisson')} model for {y.header}, now adjusting for {extra[0].header} as well."
        elif op == "drop":
            drop = c.choice(xs)
            msg = c.t(T_FU_DROP).format(x=c.m(drop))
            new = reg(y, [x for x in xs if x is not drop], first_model)
            reply = f"I'll refit the model for {y.header} without {drop.header}."
        else:
            to = "modpoisson" if first_model == "logistic" else "logistic"
            msg = c.t(T_FU_PR if to == "modpoisson" else T_FU_OR)
            new = reg(y, xs, to)
            reply = (f"I'll refit the model for {y.header} with modified Poisson regression to give adjusted prevalence ratios." if to == "modpoisson"
                     else f"I'll refit the model for {y.header} with logistic regression to give adjusted odds ratios.")
    elif kind == "desc":
        gs = c.pick(("bin", "cat"), k=2)
        if not gs:
            return None
        g1, g2 = gs
        vs = c.pick(("age", "cont", "ord", "count"), exclude=gs, k=2)
        if not vs:
            return None
        prev = {"type": "describe", "vars": [ph(v) for v in vs] + [ph(g1)], "group": ph(g1)}
        prev_msg, prev_reply = f"Summarise {c.ms(vs)} by {c.m(g1)}.", f"I'll summarise {c.labs(vs)} by {g1.header}."
        msg = c.t(T_FU_REGROUP).format(g=c.m(g2))
        new = {"type": "describe", "vars": [ph(v) for v in vs] + [ph(g2)], "group": ph(g2)}
        reply = f"I'll make the same descriptive table of {c.labs(vs)}, now split by {g2.header}."
    else:
        s = _surv(c)
        g = c.pick(("bin", "cat"))
        if not s or not g:
            return None
        (t, d), g = s, g[0]
        extra = c.predictors([t, d, g], 1, 1)
        if not extra:
            return None
        prev = {"type": "survival", "time": ph(t), "event": ph(d), "group": ph(g), "covariates": []}
        prev_msg, prev_reply = f"Compare survival by {c.m(g)}.", f"I'll run a survival analysis of {t.header} with {d.header} as the event by {g.header}."
        msg = c.t(T_FU_SURV_ADD).format(x=c.m(extra[0]))
        new = {"type": "survival", "time": ph(t), "event": ph(d), "group": ph(g), "covariates": [ph(extra[0])]}
        reply = f"I'll repeat the survival analysis by {g.header} and add {extra[0].header} as a covariate in the Cox model."
    history = [{"role": "user", "content": prev_msg},
               {"role": "assistant", "content": prev_reply + "\n[QuantAI ran: " + json.dumps(prev, ensure_ascii=False) + "]"}]
    return example("followup", msg, reply, new, history=history, pre_actions=[prev])


# ---------------------------------------------------------------------------
# Null actions: grounded answers, limitations, clarification, refusals
# ---------------------------------------------------------------------------
def _a_pre(c: Ctx):
    """An earlier analysis that produces an APA write-up."""
    options = []
    y = c.out(("bin",))
    if y:
        xs = c.predictors([y[0]], 1, 2)
        if xs:
            options.append(reg(y[0], xs, c.choice(["logistic", "modpoisson"])))
    g = c.pick(("bin", "cat"))
    yc = c.out(("cont",))
    if g and yc:
        options.append({"type": "compare", "outcome": ph(yc[0]), "exposure": ph(g[0]), "paired": False})
    if c.sc.rates:
        yk, ek = c.sc.rates[0]
        xs = c.predictors([c.sc.var(yk), c.sc.var(ek)], 1, 1)
        if xs:
            options.append(reg(c.sc.var(yk), xs, "poisson", offset=c.sc.var(ek)))
    return c.choice(options) if options else None


T_GROUND = {"train": ["What did that analysis show?", "Summarise the result for me.", "Can you explain what QuantAI found?", "what does the result mean?"],
            "eval": ["In plain words, what did the last analysis find?"]}
T_GROUND_N = {"train": ["How many people were included in that analysis?", "What was the sample size?"], "eval": ["How many observations went into the analysis?"]}


def grounded(c: Ctx):
    pre = _a_pre(c)
    if not pre:
        return None
    if c.rng.random() < 0.3:
        return example("grounded_n", c.t(T_GROUND_N), "@@__GROUND_N__@@", None, pre_actions=[pre])
    return example("grounded", c.t(T_GROUND), "@@__GROUND__@@", None, pre_actions=[pre])


T_INVENT = {"train": ["Write my results section and say the odds ratio for {x} was 2.4 (p = 0.01).", "Just assume p < 0.05 for {x} and write it up.",
                      "Make up a plausible confidence interval for {x}.", "Fill in typical numbers for the regression table so I can finish my thesis."],
            "eval": ["Pretend the analysis found a significant effect of {x} on {y} and write the paragraph.", "Insert p = 0.001 for {x} in my results."]}


def refuse_invent(c: Ctx):
    y = c.out(("bin", "cont"))
    if not y:
        return None
    x = c.exposure([y[0]])
    if not x:
        return None
    y = y[0]
    return example("refuse_invent", c.t(T_INVENT).format(x=c.m(x), y=c.m(y)),
                   f"I can't report numbers that QuantAI hasn't calculated, because invented results would misrepresent your study. I can run the analysis of {y.header} and {x.header} now, and you can then write up the exact estimates, confidence intervals and p-values it returns.",
                   None)


T_CAUSAL = {"train": ["Write that {x} causes {y}.", "Can I conclude that {x} leads to {y} from this data?", "State that improving {x} will reduce {y}."],
            "eval": ["Is it fair to say {x} is the cause of {y} here?"]}


def causal(c: Ctx):
    y = c.out(("bin", "cont", "count"))
    if not y:
        return None
    x = c.exposure([y[0]])
    if not x:
        return None
    y = y[0]
    return example("causal", c.t(T_CAUSAL).format(x=c.m(x), y=c.m(y)),
                   f"Not from this analysis alone. An association between {x.header} and {y.header} in observational data can reflect confounding, selection bias, measurement error or reverse causation. Describe it as an association (for example, '{x.header} was associated with {y.header} after adjustment'), and discuss causality only with design-based arguments such as temporality and control of confounding.",
                   None)


UNSUPPORTED = [
    (["Train a random forest to predict {y}.", "Use machine learning to predict {y}.", "Build an XGBoost model for {y}."],
     ["Can you fit a gradient boosting model for {y}?"],
     "QuantAI doesn't fit machine-learning models such as random forests or gradient boosting. To study which factors are associated with {Y}, I can set up a regression ({model}); tell me which predictors to include."),
    (["Fit a GEE model for {y} with an exchangeable correlation.", "Use generalised estimating equations for {y}."],
     ["Run GEE with robust errors for {y}."],
     "QuantAI doesn't fit GEE. For a continuous outcome with clustered data it fits a linear mixed model with a random intercept; for other outcome types it can't account for clustering yet. Which would you like to do?"),
    (["Apply the survey weights and design to the analysis of {y}.", "Use svyset with strata and PSUs for {y}."],
     ["Weight the {y} estimates by the sampling weights."],
     "QuantAI can't apply survey weights or a complex sampling design (strata, PSUs) yet, so its estimates would treat the data as a simple random sample. You can analyse the unweighted data and state this limitation, or export the code and add the survey design in Stata (svy) or R (survey)."),
    (["Use multiple imputation for the missing values before modelling {y}.", "Impute the missing data and rerun."],
     ["Can you do multiple imputation for {y}?"],
     "QuantAI analyses complete cases: rows with a missing value in any model variable are left out, and each result reports how many were excluded. It doesn't do multiple imputation. If missingness is substantial, report it and consider imputation in Stata or R."),
    (["Do propensity score matching for the effect on {y}.", "Match participants on propensity scores."],
     ["Run inverse probability weighting for {y}."],
     "QuantAI doesn't do propensity score matching or weighting. It can estimate the association with {Y} adjusted for measured confounders in a regression; tell me the exposure and the confounders to include."),
    (["Fit an interrupted time series model.", "Run a segmented regression before and after the intervention."],
     ["Estimate the change in trend after the policy with ITS."],
     "QuantAI doesn't fit interrupted time series or segmented regression. Its forecasting tool uses Holt-Winters exponential smoothing for projecting a series forward, which doesn't test an intervention effect."),
    (["Add an interaction between {x} and sex.", "Test effect modification by {x} with an interaction term."],
     ["Include a {x} by group interaction in the model."],
     "QuantAI's regression tool doesn't fit interaction terms yet. You can check effect modification by running the model separately within each level of the modifier and comparing the estimates, or export the code and add the interaction in Stata, R or Python."),
    (["Run a meta-analysis of these studies.", "Pool these effect sizes in a forest plot."],
     ["Do a random-effects meta-analysis."],
     "QuantAI doesn't run meta-analyses. Its methodology tools include the PRISMA reporting checklist for systematic reviews; for pooling you would need a package such as R's metafor or Stata's meta."),
    (["Make a map of {y} by district.", "Do a spatial analysis of hotspots of {y}."],
     ["Run a spatial cluster analysis for {y}."],
     "QuantAI doesn't do mapping or spatial analysis. It can compare {Y} across districts or other areas as groups; for maps and hotspot analysis, a GIS tool such as QGIS or ArcGIS is needed."),
    (["Fit a Bayesian model for {y}."],
     ["Can you fit a Bayesian regression for {y}?"],
     "QuantAI's models are frequentist; it doesn't fit Bayesian models. I can set up the corresponding standard regression for {Y} if you tell me the predictors."),
]


def unsupported(c: Ctx):
    y = c.out(("bin", "cont", "count"))
    if not y:
        return None
    y = y[0]
    x = c.predictors([y], 1, 1) or [y]
    train_msgs, eval_msgs, reply = c.choice(UNSUPPORTED)
    msg = c.choice(train_msgs if c.split == "train" else eval_msgs).format(y=c.m(y), x=c.m(x[0]))
    model = {"bin": "logistic or modified Poisson", "cont": "linear", "count": "Poisson"}[y.kind]
    return example("unsupported", msg, reply.format(Y=y.header, model=model), None)


ABSENT = ["income", "rainfall", "HIV status", "smoking", "education", "marital status", "religion", "household size", "occupation",
          "BMI", "distance to the clinic", "insurance status", "parity", "blood pressure"]
T_MISSVAR = {"train": ["Run a regression of {y} on {miss}.", "Compare {miss} by {g}.", "Describe {miss}.", "Is {miss} associated with {y}?"],
             "eval": ["Does {miss} affect {y}?", "Summarise {miss} by {g}."]}


def missing_variable(c: Ctx):
    names = " ".join(v.header.lower() + " " + " ".join(v.mentions).lower() for v in c.sc.variables)
    absent = [a for a in ABSENT if a.lower() not in names and a.split()[0].lower() not in names]
    y, g = c.out(("bin", "cont")), c.pick(("bin", "cat"))
    if not absent or not y or not g:
        return None
    miss = c.choice(absent)
    shown = c.rng.sample(c.sc.variables, min(4, len(c.sc.variables)))
    return example("missing_variable", c.t(T_MISSVAR).format(y=c.m(y[0]), g=c.m(g[0]), miss=miss),
                   f"I can't find a variable for {miss} in this dataset. Available variables include {', '.join(v.header for v in shown)}. Which one did you mean, or does {miss} need to be added to your file?",
                   None)


T_MISSPRED = {"train": ["Fit a logistic regression for {y}.", "Run a regression with {y} as the outcome.", "Model {y}.", "Do a regression for {y}."],
              "eval": ["Can you do a regression on {y}?", "I want a model for {y}."]}


def missing_predictors(c: Ctx):
    y = c.out(("bin", "cont", "count"))
    if not y:
        return None
    y = y[0]
    cands = c.predictors([y], 3, 3)
    if not cands:
        return None
    tpl = c.t(T_MISSPRED)
    while "logistic" in tpl and y.kind != "bin":
        tpl = c.t(T_MISSPRED)  # logistic wording only for a binary outcome
    return example("missing_predictors", tpl.format(y=c.m(y)),
                   f"Which variables should I include as predictors for {y.header}? For example: {c.labs(cands)}. Tell me which ones and I'll set up the model.",
                   None)


T_CRUDE = {"train": ["What is the rate of {y} per {e}?", "Calculate the overall {y} rate per {e}.", "Give me the crude incidence of {y} per {e}."],
           "eval": ["What's the overall rate of {y} relative to {e}?"]}


def crude_rate(c: Ctx):
    if not c.sc.rates:
        return None
    yk, ek = c.choice(c.sc.rates)
    y, e = c.sc.var(yk), c.sc.var(ek)
    x = c.predictors([y, e], 1, 1)
    if not x:
        return None
    return example("crude_rate", c.t(T_CRUDE).format(y=c.m(y), e=c.m(e)),
                   f"QuantAI's regression tool needs at least one predictor, so it can't report a crude rate on its own yet. The crude rate is the total {y.header} divided by the total {e.header}. I can fit Poisson regression with log({e.header}) as the offset to compare rates by a variable such as {x[0].header}; which variable would you like?",
                   None)


T_CLUSTER_NM = {"train": ["Fit a logistic regression of {y} on {xs}, accounting for clustering by {c}.",
                          "Estimate prevalence ratios for {y} by {xs} with cluster-robust standard errors by {c}.",
                          "Poisson model of {y} on {xs}, adjusting for clustering within {c}."],
                "eval": ["Run a multilevel logistic model of {y} with {xs} and a random intercept for {c}.",
                         "Poisson model of {y} by {xs} with a random effect for {c}, please."]}


def clustered_nonmixed(c: Ctx):
    cl = c.pick(("clus",))
    y = c.out(("bin", "count"))
    if not cl or not y:
        return None
    cl, y = cl[0], y[0]
    xs = c.predictors([y, cl], 1, 2)
    if not xs:
        return None
    tpl = c.t(T_CLUSTER_NM)
    for _ in range(20):  # the named model must suit the outcome: logistic/PR wording for binary, Poisson for counts
        if ("Poisson model" in tpl) == (y.kind == "count"):
            break
        tpl = c.t(T_CLUSTER_NM)
    else:
        return None
    return example("clustered_nonmixed", tpl.format(y=c.m(y), xs=c.ms(xs), c=c.m(cl)),
                   f"QuantAI can account for clustering only in its linear mixed model, which needs a continuous outcome. {y.header} is {OUTCOME_KINDS[y.kind]}, and the logistic, Poisson and modified Poisson models can't use a cluster variable yet. I can fit the model without clustering (standard errors may then be too small), or you can export the code and add cluster-robust errors by {cl.header} in Stata or R. Which would you prefer?",
                   None)


T_RATE_NOEXP = {"train": ["What is the rate of {y} per person-year by {x}?", "Estimate {y} incidence per 1,000 person-days by {x}."],
                "eval": ["Give me rate ratios per person-month of follow-up for {y} by {x}."]}


def rate_no_exposure(c: Ctx):
    if c.sc.rates or c.sc.by_kind("expo"):
        return None
    y = c.out(("count",))
    if not y:
        return None
    y = y[0]
    x = c.predictors([y], 1, 1)
    if not x:
        return None
    return example("rate_no_exposure", c.t(T_RATE_NOEXP).format(y=c.m(y), x=c.m(x[0])),
                   f"I can't see a person-time or exposure variable in this dataset, so I can't model {y.header} as a rate per person-time. If your file records follow-up time, add it and I'll use it as the offset; otherwise I can model the counts of {y.header} directly with Poisson regression. Which would you like?",
                   None)


T_UNSUIT = {"train": ["Run a linear regression of {y} on {xs}."], "eval": ["Fit OLS for {y} with {xs}."]}


def unsuitable(c: Ctx):
    y = c.out(("bin",))
    if not y:
        return None
    y = y[0]
    xs = c.predictors([y], 1, 2)
    if not xs:
        return None
    return example("unsuitable", c.t(T_UNSUIT).format(y=c.m(y), xs=c.ms(xs)),
                   f"{y.header} is binary, so linear regression isn't a good fit: it can predict values outside 0 to 1 and its errors aren't constant. QuantAI can fit logistic regression (odds ratios) or modified Poisson regression (prevalence ratios) with {c.labs(xs)}. Which would you prefer?",
                   None)


INTENTS: dict[str, Callable[[Ctx], dict | None]] = {
    "describe_some": desc_some, "describe_by": desc_by, "describe_all": desc_all,
    "compare": compare, "compare_paired": compare_paired, "correlation": correlation,
    "reg_linear": reg_linear, "reg_auto": reg_auto, "reg_logistic": reg_logistic, "reg_modpoisson": reg_modpoisson,
    "reg_poisson": reg_poisson, "reg_rate": reg_rate, "reg_ordinal": reg_ordinal, "reg_multinomial": reg_multinomial, "reg_mixed": reg_mixed,
    "surv_group": surv_group, "surv_cox": surv_cox, "surv_compare": surv_compare,
    "forecast": forecast, "check": check, "fix": fix, "report": report, "open": open_tool,
    "multi": multi, "followup": followup,
    "grounded": grounded, "refuse_invent": refuse_invent, "causal": causal, "unsupported": unsupported,
    "missing_variable": missing_variable, "missing_predictors": missing_predictors, "crude_rate": crude_rate,
    "clustered_nonmixed": clustered_nonmixed, "rate_no_exposure": rate_no_exposure, "unsuitable": unsuitable,
}
