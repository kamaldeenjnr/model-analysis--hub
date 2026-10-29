# Run QuantAI as a self-hosted model

This deployment path runs inference on your own machine or server. The application does not call a hosted LLM inference API and does not consume third-party per-token credits. You still need hardware, electricity or server rental, storage, and occasional maintenance. A 3B model may be slow or impractical on a CPU-only laptop; local does not mean resource-free.

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
python validate_data.py --train data/train.jsonl --eval data/eval.jsonl
pytest -q
```
The starter fixtures are synthetic smoke tests only. Expand and review the dataset before a meaningful training run. Keep task families separated between training and held-out evaluation.

## 3. Train the adapter
Run on a GPU notebook or machine after checking the base model's license and current model card:
```bash
python train_lora.py --train data/train.jsonl --eval data/eval.jsonl --output outputs/quantai-qwen25-3b-lora
```
The script uses the Qwen3 4B Instruct instruction base as a starting point. Training it does not mean the model was pretrained from scratch. The quality of the specialist depends heavily on reviewed data and evaluation.

## 4. Evaluate both candidates
```bash
python evaluate.py --model outputs/quantai-qwen25-3b-lora --cases data/eval.jsonl --report outputs/fine-tuned-evaluation.json
python evaluate.py --model outputs/quantai-qwen25-3b-lora --baseline-only --cases data/eval.jsonl --report outputs/base-evaluation.json
```
Review each failure, not just the aggregate scores. Synthetic examples cannot substantiate public claims about accuracy or safety.

## 5. Export a standalone local model
```bash
python export_model.py --adapter outputs/quantai-qwen25-3b-lora --output outputs/quantai-merged-model
```
The merged output includes the base weights plus the fine-tuned changes. Keep it local unless redistribution and all applicable base-model terms permit publishing it.

## 6. Start local inference
```bash
# Windows PowerShell
$env:QUANTAI_MODEL_PATH = (Resolve-Path outputs/quantai-merged-model).Path
uvicorn serve:app --host 127.0.0.1 --port 8000
```
Linux/macOS:
```bash
export QUANTAI_MODEL_PATH="$PWD/outputs/quantai-merged-model"
uvicorn serve:app --host 127.0.0.1 --port 8000
```
Check `http://127.0.0.1:8000/health`. Send a POST request to `/chat` with JSON fields `message`, optional `context`, and optional `history`. The knowledge base is indexed locally using SQLite full-text search. No API key is required for inference.

## 7. Before public hosting
Add authentication, HTTPS, request quotas, concurrency limits, monitoring and a spending cap for infrastructure. Bind to 127.0.0.1 for local-only use. Do not expose a local development server directly to the internet. For public use, deploy behind a reverse proxy on a secured host. Configure the website to call your own backend only after browser integration and security review. Never send raw uploaded files or patient-level data unless there is an approved and explicitly disclosed reason.
