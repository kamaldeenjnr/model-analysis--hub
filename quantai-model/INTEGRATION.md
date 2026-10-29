# Connecting a trained QuantAI adapter

This is a deployment checklist, not a live connection. QuantAI has **not** been switched to a custom Hugging Face model.

## Recommended first deployment

1. Train and evaluate the LoRA adapter using the instructions in the parent README.
2. Publish the adapter to a Hugging Face model repository only after license and dataset review.
3. Choose an inference path:
   - **Experiment:** use an available Hugging Face Inference Provider model for a small smoke test. The current free allowance is limited and can change; do not promise free production inference.
   - **Custom adapter:** use a compatible hosted inference endpoint that loads the adapter and its base model. Confirm adapter support, cold-start behavior, context length and cost before paying for a dedicated endpoint.
   - **Low-cost staging:** keep the existing assistant as the active provider while testing the candidate behind a server-side feature flag.
4. Store the Hugging Face token as a Cloudflare Worker secret (never in browser JavaScript or a committed file). Keep the model ID as a non-secret Worker variable.
5. Add the Hugging Face provider as an optional route in `cloudflare-worker/worker.js`; leave existing Gemini/Groq routes available as fallbacks. Do not enable the route until the endpoint has passed the acceptance checks below.
6. For data mode, send only the existing compact summary. Never forward the uploaded file, raw rows, or identifiable records to the model.
7. Validate model output against an allowlist. The model may request only documented QuantAI actions. Validate each action's parameters and variable names against the current dataset before execution. Reject unknown keys and malformed action lists.
8. Log aggregate status, latency, model revision and error category—not raw prompts, data summaries, access tokens, or personal data.

## Acceptance checks

- The endpoint returns a response for a methods question and a supported tool-routing request.
- Invalid JSON, unknown actions, nonexistent variable names and malformed action arrays are rejected safely.
- Prompt-injection text in a column name or data summary cannot override the system rules.
- Existing statistical regression tests pass unchanged.
- When the model endpoint is unavailable, the current provider fallback works.
- Rate limits and a hard spending limit are configured before public rollout.

Official references: [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers), [pricing](https://huggingface.co/docs/inference-providers/pricing), and [model cards](https://huggingface.co/docs/hub/model-cards).
