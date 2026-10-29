# QuantAI model foundation

QuantAI's model should specialize in **research-methods guidance and routing analysis requests to QuantAI's existing tools**. It must not replace the statistical engine or invent numerical results.

## Current status

This directory is a reproducible starting point, not a claim that a trained model already exists. The included examples are synthetic development fixtures. They are not enough to establish research-grade performance. Before a public release, expand the training set with reviewed examples and pass the held-out evaluation gates below.

## Architecture

1. **Browser (GitHub Pages):** reads CSV/Excel locally and computes analyses using the existing deterministic QuantAI engine.
2. **Cloudflare Worker:** validates requests, keeps provider credentials server-side, sends only the existing summary/context and conversation to the assistant, and validates/falls back between providers.
3. **QuantAI specialist model:** a small instruction model fine-tuned with LoRA on research-methods questions and structured action-routing examples. Initial base: `Qwen/Qwen2.5-3B-Instruct` (verify current model card and license before publishing a derivative).
4. **Deterministic execution:** the model may request a supported action; QuantAI validates it and runs the existing JavaScript engine. All estimates, confidence intervals, p-values, model diagnostics and charts remain engine-generated.
5. **Evaluation:** held-out cases measure valid JSON, action routing, exact action parameters, unsupported claims, and regression tests against the existing engine.

## Low-cost development

- Start with the small 3B base and LoRA adapters rather than pretraining a model from scratch.
- Use a free notebook GPU when available; otherwise run dataset checks and evaluation on CPU and defer training.
- Keep the fine-tuned adapter separate from the base model and record its base revision, dataset version, training configuration and evaluation report.
- Do not assume Hugging Face Inference Providers will serve a custom fine-tuned adapter automatically. Hosting and costs depend on the selected deployment path. See [Hugging Face fine-tuning](https://huggingface.co/docs/transformers/training) and [Inference Providers pricing](https://huggingface.co/docs/inference-providers/pricing).

## Validate the examples first

The validator checks JSONL structure, supported action types, minimum example counts and exact prompt overlap between the training and held-out sets. It does not judge whether the examples are scientifically correct.

```bash
python validate_data.py --train data/train.jsonl --eval data/eval.jsonl
```

## Train

Use Python 3.10 or 3.11 in a GPU notebook or machine with sufficient memory.

```bash
cd quantai-model
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python train_lora.py --train data/train.jsonl --eval data/eval.jsonl --output outputs/quantai-qwen25-3b-lora
```

The script uses 4-bit loading when a CUDA GPU and bitsandbytes are available; it falls back to ordinary loading where possible. A 3B model may still be too large for a CPU-only laptop. Do not commit access tokens or model weights to this repository.

## Evaluate

```bash
python evaluate.py --model outputs/quantai-qwen25-3b-lora --cases data/eval.jsonl --report outputs/fine-tuned-evaluation.json\n\n# Compare the untouched base model on the same held-out cases\npython evaluate.py --model outputs/quantai-qwen25-3b-lora --baseline-only --cases data/eval.jsonl --report outputs/base-evaluation.json
```

The evaluator reports JSON validity, supported action accuracy and exact action match. Inspect individual failures, and compare the fine-tuned adapter against the unfine-tuned base on the same held-out cases before choosing a deployment candidate. These metrics alone do not prove statistical correctness or safety.

## Release gates

Do not route production traffic to a candidate model until:

- The evaluation set is separated from training and contains reviewed edge cases.
- The model returns valid action JSON reliably and never creates unsupported variable names.
- All action parameters are validated server-side and again by the browser before execution.
- Requests to run analyses are checked against the actual loaded dataset schema.
- Regression tests confirm that model text cannot alter numerical calculations.
- Privacy, rate limits, abuse handling, latency and total cost are reviewed.
- The model card documents the training data, license, intended use, limitations, metrics and known failure cases.

## Data policy

The starter examples are synthetic. For future training, use only data that can legally be redistributed or used for training. Remove personal and confidential data. Do not train on identifiable research records, unpublished manuscripts, or patient-level datasets without explicit authorization and appropriate governance. Keep the held-out evaluation set out of training.

## Integration status

The current QuantAI app and Cloudflare Worker remain the source of truth. This foundation is deliberately separate until a candidate model has been trained, evaluated, hosted, and explicitly configured. The existing provider-backed assistant remains available as a fallback; the browser's statistics engine must remain deterministic.
