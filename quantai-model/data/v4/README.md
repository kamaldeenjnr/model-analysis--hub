# QuantAI v4 dataset

Training and held-out evaluation data for QuantAI's chat model. Built by [`../../datagen/build.py`](../../datagen/build.py); exact counts, file hashes and dropped rows are in [`build_report.json`](build_report.json).

| File | Rows | Rows with no action (answer, clarify or decline) | Research scenarios |
|---|---:|---:|---:|
| `train.jsonl` | 1,124 | 320 | 27 + QuantAI's example dataset |
| `eval.jsonl` | 236 | 73 | 10 + QuantAI's example dataset |

## What a row contains

```json
{"message": "...", "context": "...", "history": [...], "completion": "{\"reply\":\"...\",\"action\":...}",
 "meta": {"intent": "reg_rate", "scenario": "hai", "split": "train", "var_types": {"infections": "count", ...}}}
```

- `message`, `context` and `history` are exactly the fields the browser sends to the local API (`/chat`). The context was produced by QuantAI's own `agentContext()` in a headless browser: dataset summary, variable table (name in code, original name, type, missing, summary), data-check findings, analyses already run, tools list and methodology inputs.
- At training and serving time `prompting.py` turns these into the model input (system prompt, retrieved knowledge, context, question). Training, evaluation and `serve.py` share that code, so the model is trained on the input it will receive.
- `completion` is the target answer, in the action contract the browser validates.
- `meta` is for analysis only (request type, scenario, variable types); it is not shown to the model.

## How the rows were made

1. **Scenarios.** 37 fictional research datasets from public health and clinical research (malaria, anaemia in pregnancy, ART cohorts, TB, child growth, hypertension, diabetes, immunisation, WASH, neonatal care, mental health, hospital infections, trauma, vaccine attitudes, cancer survival, occupational health, family planning, sickle cell, eye health, older adults, adolescents, pharmacy stock, laboratory turnaround, nutrition, and monthly/weekly/quarterly service series), plus QuantAI's built-in example dataset. Data values are synthetic.
2. **Requests.** 38 request types (listed with counts in `build_report.json`) written as templates with natural variable mentions. Training and held-out rows use different wording templates, and held-out rows use 10 scenarios never used in training (QuantAI's example dataset appears in both, as it does for real users).
3. **Labels.** Each expected action follows fixed rules (below), then was **executed in QuantAI's engine**; rows whose action failed were dropped (for example, one logistic model with collinear predictors). Every variable an action names must be visible in the context, and QuantAI must have typed it as the rule assumes.
4. **Replies.** Short, factual descriptions of what will run. Replies about earlier results ("What did that analysis show?") quote QuantAI's own write-up, so every number is the engine's.
5. **Concept answers.** 124 entries of QuantAI's own handbook (`RESEARCH_KB` and `quantai-kb.js`) with new question wordings, plus 20 fuller "In more depth:" answers on core statistics topics. Claims about QuantAI's features in those answers were checked against the engine code.

## Label rules

- Prevalence or risk ratios for a binary outcome → `modpoisson`; odds ratios → `logistic`; rates per person-time or other exposure → `poisson` with that variable as `offset` (never also a predictor), including when the user says "compare" or "differ".
- Time-to-event questions → `survival` (compared groups in `group`, adjustment variables in `covariates`), including when the user says "compare".
- `cluster` only with `mixed` (continuous outcome). Clustering with other models is not supported by the engine (it would be silently ignored), so the model explains the limitation and returns no action.
- Every requested adjustment variable is kept. `auto` only when the user names no model or effect measure.
- No action, with a clear reply, when: predictors are not named (ask which), a variable is not in the dataset, a crude rate is requested (the regression tool needs a predictor), a requested model does not suit the outcome, the method is not available (machine learning, GEE, survey weights, multiple imputation, propensity scores, interrupted time series, interactions, meta-analysis, mapping, Bayesian models), the user asks for invented numbers, or a causal claim is not supported.
- Demographic and structural variables (sex, district, regimen, treatment arm and similar) are never used as outcomes.

## Checks

```bash
python validate_data.py --train data/v4/train.jsonl --eval data/v4/eval.jsonl   # schema, overlap, near-duplicates, scenario separation
python datagen/audit.py data/v4/train.jsonl data/v4/eval.jsonl                  # every action against engine rules
pytest -q test_pipeline_v4.py
```

## Rebuilding

Requires Node.js with Playwright and Chromium (`npm i -g playwright` and `npx playwright install chromium` if they are not already installed).

```bash
cd quantai-model/datagen && python build.py   # deterministic for a given --seed (default 2026)
```

## Limitations

- Requests are template-generated. They cover many wordings, but not the full variety, spelling and ambiguity of real users' questions. Real (consented) questions from QuantAI users, reviewed by a person, would be the most valuable next source.
- Labels encode QuantAI's conventions and the engine's current capabilities, reviewed by rule and spot-check, not by an independent statistician. A sample should be reviewed by a second person.
- Strong scores on `eval.jsonl` show the model applies QuantAI's action contract on unseen scenarios and wording. They do not show expert-level judgement on open-ended research problems.
- The v3 files (`../train.jsonl`, `../eval.jsonl`) are kept unchanged for reproducibility. They use a prompt format the app does not send.
