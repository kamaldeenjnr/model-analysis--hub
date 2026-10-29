# QuantAI Specialist — Model Card (complete before release)

## Status
Development scaffold. No trained or evaluated production model is implied by this repository.

## Intended use
- Explain undergraduate-level research methods and biostatistics.
- Map natural-language requests to a narrow, documented set of QuantAI actions.
- Help users navigate QuantAI's deterministic analysis tools.

## Out-of-scope use
- Calculating or fabricating statistical estimates in natural language.
- Making clinical diagnoses, treatment decisions, or unsupported causal claims.
- Executing arbitrary code, SQL, shell commands, or user-defined actions.
- Replacing a qualified analyst, research supervisor, ethics review, or clinical judgment.

## Base model and license
Record exact model ID, immutable revision, license, tokenizer revision, and any derivative-model obligations before publishing.

## Training data
Record data sources, licenses, synthetic vs. reviewed proportions, preprocessing, exclusions, and privacy review. Do not include personal or confidential research data without authorization.

## Evaluation
Attach held-out metrics, base-model comparison, action-routing confusion matrix, invalid JSON examples, hallucination tests, prompt-injection tests, and regression-test results from the existing QuantAI engine.

## Limitations
The starter data in this repository is synthetic and small. It is for pipeline smoke tests only, not evidence of reliable production performance.

## Release decision
- [ ] Training/evaluation split independently reviewed
- [ ] No private or identifiable data in training set
- [ ] JSON schema and action parameters validated
- [ ] Existing deterministic engine regression suite passes
- [ ] Privacy, cost, latency and abuse controls reviewed
- [ ] Model card and license reviewed
- [ ] Candidate approved for limited staged deployment
