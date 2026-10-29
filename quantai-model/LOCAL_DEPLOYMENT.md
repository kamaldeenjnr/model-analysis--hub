# Run QuantAI as a self-hosted model

This deployment path runs inference on your own machine or server. The application does not call a hosted LLM inference API and does not consume third-party per-token credits. You still need hardware, electricity or server rental, storage, and occasional maintenance. A 4B model may be slow or impractical on a CPU-only laptop; local does not mean resource-free.

## 1. Environment
Use Python 3.10 or 3.11 and a machine with sufficient RAM/storage. A compatible CUDA GPU is strongly recommended for training; inference can run on CPU if enough memory is available, usually more slowly. Do not commit tokens, private datasets, model weights or environment files.

Windows PowerShell:
```powershell
cd quantai-model
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:
```bash
cd quantai-model
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Validate data
```bash
python validate_data.py --train data/v4/train.jsonl --eval data/v4/eval.jsonl
python datagen/audit.py data/v4/train.jsonl data/v4/eval.jsonl
pytest -q
```
The starter fixtures are synthetic smoke tests only. Expand and review the dataset before a meaningful training run. Keep task families separated between training and held-out evaluation.

## 3. Train the adapter
Run on a GPU notebook or machine after checking the base model's license and current model card:
```bash
python train_lora.py --train data/v4/train.jsonl --eval data/v4/eval.jsonl --output outputs/quantai-qwen3-4b-lora-v4 --epochs 1 --max-train-rows 650 --resume
```
(The v3 command used `data/train.jsonl` and `--output outputs/quantai-qwen3-4b-lora-v3`; see `README.md` for the evaluation commands.)
The script uses the Qwen3 4B Instruct instruction base as a starting point. Training it does not mean the model was pretrained from scratch. The quality of the specialist depends heavily on reviewed data and evaluation.

## 4. Evaluate both candidates
```bash
python evaluate.py --model outputs/quantai-qwen3-4b-lora-v3 --cases data/eval.jsonl --report outputs/fine-tuned-evaluation-v3.json
python evaluate.py --model outputs/quantai-qwen3-4b-lora-v3 --baseline-only --cases data/eval.jsonl --report outputs/base-evaluation-v3.json
```
Review each failure, not just the aggregate scores. Synthetic examples cannot substantiate public claims about accuracy or safety.

## 5. Export a standalone local model
```bash
python export_model.py --adapter outputs/quantai-qwen3-4b-lora-v3 --output outputs/quantai-merged-model-v3
```
The merged output includes the base weights plus the fine-tuned changes. Keep it local unless redistribution and all applicable base-model terms permit publishing it.

## 6. Start local inference
```bash
# Windows PowerShell
$env:QUANTAI_MODEL_PATH = (Resolve-Path outputs/quantai-merged-model-v3).Path
uvicorn serve:app --host 127.0.0.1 --port 8000
```
Linux/macOS:
```bash
export QUANTAI_MODEL_PATH="$PWD/outputs/quantai-merged-model-v3"
uvicorn serve:app --host 127.0.0.1 --port 8000
```
Check `http://127.0.0.1:8000/health`; it reports the `prompt_version` in use. A model must be served with the prompt it was trained on: `export_model.py` records it in `quantai_prompt.json` inside the exported folder and `serve.py` reads it (models exported before v4 have no marker and get the v3 prompt). Override with `QUANTAI_PROMPT_VERSION=v3` or `v4` only if you know the model's training prompt. To serve v4 after evaluating it, set `QUANTAI_MODEL_PATH` to `outputs/quantai-merged-model-v4`. Send a POST request to `/chat` with JSON fields `message`, optional `context`, and optional `history`. The knowledge base is indexed locally using SQLite full-text search. No API key is required for inference.


## 7. Connect the existing browser app to this API (opt-in)

The website continues to use its existing assistant unless you explicitly configure the local endpoint. Before changing the production site, test this on a staging copy.

1. Start the service and confirm `http://127.0.0.1:8000/health` reports that the model directory exists. The first chat request loads model weights and can take time.
2. Configure CORS to allow the exact origin serving your website. For example, in PowerShell before starting Uvicorn:
   ```powershell
   $env:QUANTAI_CORS_ORIGINS = "http://localhost:5173"
   uvicorn serve:app --host 127.0.0.1 --port 8000
   ```
   For a hosted site, use its exact HTTPS origin instead of the localhost example. Do not use `*` for a public deployment. A locally running service on one person's computer is not reachable by visitors to a public website.
3. In the HTML page, set the API base URL **before** loading `assets/js/quantai.js`:
   ```html
   <script>window.QUANTAI_LOCAL_AI_URL = "https://your-secured-api.example.com";</script>
   <script src="assets/js/quantai.js"></script>
   ```
   Set the base URL only; the browser client adds `/chat`. The API must use HTTPS when the website uses HTTPS. Do not place API secrets in browser JavaScript.
4. Verify the browser's Network panel shows a POST to `/chat`, the response includes `reply` and `action`, and the browser runs any returned action through the deterministic engine. If the local endpoint is not configured, the existing worker path remains unchanged.

### Count-rate regression offsets

For Poisson rate models, `offset` must name a positive numeric exposure variable such as `patient_days` or `person_years`. It is not an ordinary predictor. QuantAI uses `log(exposure)` as a fixed offset and rejects rows with non-positive exposure values. The exported Stata, Python and R model code should include the same offset. Check the generated code and compare results against a trusted statistical package before research use.

## 8. Before public hosting

Add authentication, HTTPS, request quotas, concurrency limits, monitoring and a spending cap for infrastructure. Bind to 127.0.0.1 for local-only use. Do not expose a local development server directly to the internet. For public use, deploy behind a reverse proxy on a secured host. Configure the website to call your own backend only after browser integration and security review. Never send raw uploaded files or patient-level data unless there is an approved and explicitly disclosed reason.
