# QuantAI Specialist — Model Card (complete before release)

## Status
Development scaffold. No trained or evaluated production model is implied by this repository. Training and evaluation must be run and results attached before release.

## Base model
Planned base: `Qwen/Qwen3-4B-Instruct-2507`. Hugging Face currently lists this model under Apache-2.0; retain the exact licence file, model revision and upstream notices for the specific checkpoint used. Verify the model card and license again before distributing a derivative. See https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507.

## Intended use
- Explain research methods and biostatistics at undergraduate and practitioner levels.
- Map natural-language requests to a narrow, documented set of QuantAI actions.
- Retrieve relevant locally indexed knowledge and help users navigate QuantAI's tools.
- Produce grounded explanations of results computed by deterministic software.

## Out-of-scope use
- Fabricating p-values, confidence intervals, coefficients, sample sizes or diagnostic output.
- Clinical diagnosis, treatment decisions, or unsupported causal conclusions.
- Executing arbitrary code, SQL, shell commands or user-defined actions.
- Replacing qualified statistical review, research supervision, ethics review or clinical judgment.

## Training data
Record data sources, licenses, synthetic vs reviewed proportions, preprocessing, exclusions, and privacy review. The current starter fixtures are synthetic and small. They are not sufficient evidence of specialist accuracy. Do not include personal or confidential research data without authorisation and appropriate governance.

## Evaluation
Attach held-out metrics, base-model comparison, action-routing confusion matrix, parameter accuracy, unsupported-claim rate, prompt-injection and privacy tests, methods-answer review, and regression-test results from the existing QuantAI engine. Record test-set size and confidence intervals where appropriate.

## Deployment
Record the exact merged model hash, quantisation, runtime, hardware, latency, memory use, and estimated infrastructure cost. The local service is intended to run from local weights and does not need a per-request LLM API. Public deployment still requires secure hosting, access controls, rate limits and monitoring.

## Limitations
The starter data in this repository is synthetic and small. Fine-tuning an open-weight model produces a specialised derivative, not a model pretrained from scratch. It can still produce incorrect explanations or structured actions; independent evaluation and human review remain necessary.

## Release decision
- [ ] Exact base revision and licence reviewed
- [ ] Training data reviewed and permissions recorded
- [ ] Held-out split reviewed and leakage checked
- [ ] Training run reproducible from saved configuration
- [ ] Base-model comparison completed
- [ ] Unsupported-claim and prompt-injection tests passed
- [ ] Action parameters validated server-side and by the browser
- [ ] Existing statistical engine regression suite passes
- [ ] Privacy, cost, latency and abuse controls reviewed
- [ ] Model card and limitations reviewed
- [ ] Candidate approved for limited staged deployment
