# QuantAI Specialist Dataset and Benchmark Specification

## Purpose

Create a high-quality supervised dataset for a QuantAI specialist that explains research methods and maps natural-language requests to a narrow, documented action schema. Quality and coverage matter more than a large synthetic count.

## Data record format

Each JSONL record should contain:
- `id`: stable unique identifier;
- `category`: one of methods_qa, action_routing, clarification, result_grounding, assumption_check, safety_privacy, adversarial, or abstention;
- `prompt`: task request plus a synthetic dataset schema or bounded result summary where relevant;
- `target`: a JSON object with `reply` and `action`;
- `review_status`: draft, reviewed, or rejected;
- `source`: synthetic, public licensed source, or expert-authored;
- `source_license`: documented license or permission;
- `notes`: optional reviewer rationale, with no personal data.

Do not add private patient records, identifiable research data, unpublished manuscripts or proprietary training content without explicit authorization and governance.

## Dataset coverage

The first reviewed benchmark should include at least these categories:
1. Test selection: continuous, binary, ordinal, nominal, count, time-to-event, paired and clustered outcomes.
2. Action routing: describe, compare, regression, survival, forecast, check, fix, report and open.
3. Exact schema: correct outcome, predictors, exposure, grouping, time, event, cluster and forecast horizon.
4. Clarification: nonexistent or ambiguous variables, absent dataset, unclear paired design and unspecified time horizon.
5. Grounding: requests for unavailable p-values, confidence intervals, sample sizes or estimates must not cause invented values.
6. Research methods: confounding, bias, sampling, study design, power, missing data, assumptions and reporting guidance.
7. Statistical interpretation: p-values, confidence intervals, effect sizes, odds ratios versus risk ratios, and association versus causation.
8. Safety and privacy: requests to expose raw records, identify participants or infer sensitive facts from insufficient evidence.
9. Prompt injection: hostile instructions embedded in variable names, categories or summaries.
10. Robustness: spelling errors, plain-language requests, long requests, multilingual terms common among target users and realistic Ghanaian public-health research scenarios using only synthetic data.

## Splitting and review

- Split by underlying task template or scenario family, not just random rows, to reduce leakage.
- Keep the evaluation set separate and read-only during development.
- Have a knowledgeable reviewer assess correctness of methods answers and action parameters.
- Track rejected examples and reasons. Never silently promote machine-generated examples to reviewed status.
- Add challenging examples based on observed failures, then rerun the same frozen benchmark.

## Release metrics

Report, at minimum:
- valid JSON and schema rates;
- supported-action precision and recall;
- exact action match and per-field parameter accuracy;
- clarification accuracy for ambiguous or absent variables;
- unsupported-claim rate;
- methods-answer correctness by category;
- prompt-injection and privacy test pass rates;
- latency, memory use and measured cost.

Set release thresholds before evaluating a candidate. The starter fixture scores must not be used as public performance claims.

## Review checklist

- [ ] Each target answer was checked against the action schema.
- [ ] Statistical statements were reviewed for correctness.
- [ ] No exact or near-duplicate task family leaks across train and evaluation.
- [ ] All source licenses and permissions are documented.
- [ ] No personal or confidential information is included.
- [ ] Adversarial and abstention cases are represented.
- [ ] Evaluation results are reproducible and versioned.
