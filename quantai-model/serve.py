#!/usr/bin/env python3
"""Local, self-hosted QuantAI inference service. No third-party inference API is called."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompting  # noqa: E402  shared with train_lora.py and evaluate.py

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("QUANTAI_MODEL_PATH", ROOT / "outputs" / "quantai-merged-model-v3"))
DB_PATH = Path(os.environ.get("QUANTAI_KNOWLEDGE_DB", ROOT / "knowledge_index.json"))
MAX_INPUT_CHARS = int(os.environ.get("QUANTAI_MAX_INPUT_CHARS", "10000"))
MAX_NEW_TOKENS = int(os.environ.get("QUANTAI_MAX_NEW_TOKENS", "500"))
MAX_INPUT_TOKENS = int(os.environ.get("QUANTAI_MAX_INPUT_TOKENS", "3584"))
PROMPT_VERSION = prompting.resolve_prompt_version(MODEL_PATH, os.environ.get("QUANTAI_PROMPT_VERSION"))
ACTION_FIELDS = {
    "describe": {"type", "vars", "group"},
    "compare": {"type", "outcome", "exposure", "paired"},
    "regression": {"type", "outcome", "predictors", "model", "cluster", "offset"},
    "survival": {"type", "time", "event", "group", "covariates"},
    "forecast": {"type", "date", "value", "horizon"},
    "check": {"type"},
    "fix": {"type"},
    "report": {"type"},
    "open": {"type", "tool"},
}
OPEN_TOOLS = {"data", "describe", "compare", "regression", "survival", "forecast", "export", "question", "design", "sample", "test", "checklist"}
REGRESSION_MODELS = {"auto", "linear", "logistic", "modpoisson", "poisson", "ordinal", "multinomial", "mixed"}
SYSTEM_PROMPT = prompting.system_prompt(PROMPT_VERSION)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    context: str = Field(default="", max_length=MAX_INPUT_CHARS)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=8)

class ChatResponse(BaseModel):
    reply: str
    action: Any = None
    model: str
    local_inference: bool = True
    retrieved_knowledge: list[str] = Field(default_factory=list)

app = FastAPI(title="QuantAI Local Specialist", version="0.1.0")
# Set QUANTAI_CORS_ORIGINS to comma-separated exact web origins in deployment.
# Do not use "*" for a public deployment.
ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.environ.get(
        "QUANTAI_CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
_model = None
_tokenizer = None
_model_lock = threading.Lock()


def initialise_knowledge() -> None:
    prompting.initialise_knowledge(prompting.knowledge_sources(ROOT), DB_PATH)


def retrieve(query: str, limit: int = 3) -> list[dict[str, str]]:
    return prompting.retrieve(DB_PATH, query, limit)


def load_model() -> None:
    global _model, _tokenizer
    if _model is not None:
        return
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model directory not found: {MODEL_PATH}. Train and export a model first; "
            "this service intentionally does not download a model or call a hosted API."
        )
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    kwargs: dict[str, Any] = {"local_files_only": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32
    _model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), **kwargs)
    _model.eval()


def valid_action_item(item: Any) -> bool:
    if not isinstance(item, dict) or not isinstance(item.get("type"), str):
        return False
    action_type = item["type"]
    if action_type not in ACTION_FIELDS:
        return False
    # Every action has an exact contract: required keys cannot be omitted and
    # unrecognised keys cannot be passed to the deterministic engine.
    if set(item) != ACTION_FIELDS[action_type]:
        return False
    if action_type == "describe":
        return isinstance(item.get("vars"), list) and all(isinstance(v, str) for v in item["vars"]) and (item.get("group") is None or isinstance(item.get("group"), str))
    if action_type == "compare":
        return isinstance(item.get("outcome"), str) and isinstance(item.get("exposure"), str) and isinstance(item.get("paired"), bool)
    if action_type == "regression":
        return (isinstance(item.get("outcome"), str) and isinstance(item.get("predictors"), list)
                and all(isinstance(v, str) for v in item["predictors"])
                and isinstance(item.get("model"), str) and item["model"] in REGRESSION_MODELS
                and (item.get("cluster") is None or isinstance(item.get("cluster"), str))
                and (item.get("offset") is None or (isinstance(item.get("offset"), str) and bool(item["offset"].strip())))
                and (item.get("offset") is None or (item["model"] == "poisson" and item["offset"] != item["outcome"] and item["offset"] not in item["predictors"])))
    if action_type == "survival":
        return (isinstance(item.get("time"), str) and isinstance(item.get("event"), str)
                and (item.get("group") is None or isinstance(item.get("group"), str))
                and isinstance(item.get("covariates"), list)
                and all(isinstance(v, str) for v in item["covariates"]))
    if action_type == "forecast":
        return ((item.get("date") is None or isinstance(item.get("date"), str))
                and isinstance(item.get("value"), str) and isinstance(item.get("horizon"), int)
                and not isinstance(item.get("horizon"), bool) and 1 <= item["horizon"] <= 120)
    if action_type == "open":
        return isinstance(item.get("tool"), str) and item["tool"] in OPEN_TOOLS
    return action_type in {"check", "fix", "report"}


def normalise_action_item(item: dict[str, Any]) -> dict[str, Any]:
    """Apply safe, deterministic defaults before routing an otherwise valid action.

    The describe tool uses ``vars`` as columns to summarise and ``group`` as the
    stratifier. Include a non-null grouping variable in ``vars`` as well so the
    browser engine has access to the column it must group by.
    """
    normalised = dict(item)
    if normalised.get("type") == "regression" and "offset" not in normalised:
        normalised["offset"] = None
    if (
        normalised.get("type") == "describe"
        and isinstance(normalised.get("vars"), list)
        and isinstance(normalised.get("group"), str)
        and normalised["group"] not in normalised["vars"]
    ):
        normalised["vars"] = [*normalised["vars"], normalised["group"]]
    return normalised

def validate_output(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {"reply": "I couldn't format a reliable response. Please rephrase the request.", "action": None}
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"reply": "I couldn't format a reliable response. Please rephrase the request.", "action": None}
    if not isinstance(parsed, dict) or set(parsed) != {"reply", "action"} or not isinstance(parsed.get("reply"), str):
        return {"reply": "The model returned an unsupported response format. Please rephrase the request.", "action": None}
    action = parsed["action"]
    actions = action if isinstance(action, list) else [action]
    if action is not None:
        if not isinstance(action, (dict, list)) or not actions or len(actions) > 4:
            return {"reply": "I need to clarify the request before selecting a tool.", "action": None}
        actions = [normalise_action_item(item) if isinstance(item, dict) else item for item in actions]
        for item in actions:
            if not valid_action_item(item):
                return {"reply": "I need to clarify the request before selecting a valid supported tool.", "action": None}
        action = actions if isinstance(action, list) else actions[0]
    return {"reply": parsed["reply"][:5000], "action": action}


@app.on_event("startup")
def startup() -> None:
    initialise_knowledge()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready" if MODEL_PATH.exists() else "model_not_installed",
        "model_path": str(MODEL_PATH),
        "model_loaded": _model is not None,
        "inference": "local_only",
        "knowledge_index": DB_PATH.exists(),
        "prompt_version": PROMPT_VERSION,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    retrieved = retrieve(prompting.retrieval_query(request.message))
    messages = prompting.build_messages(
        request.message, request.context, request.history, retrieved, PROMPT_VERSION
    )
    try:
        prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Never silently truncate a live request: the training pipeline rejects
        # over-length examples, and truncation can remove the user's question.
        inputs = _tokenizer(prompt, return_tensors="pt", truncation=False)
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > MAX_INPUT_TOKENS:
            raise HTTPException(
                status_code=413,
                detail=f"Request is {input_tokens} tokens; the local model limit is {MAX_INPUT_TOKENS}. "
                       "Shorten the question or reduce the supplied context and try again.",
            )
        if torch.cuda.is_available():
            inputs = {key: value.to(_model.device) for key, value in inputs.items()}
        with _model_lock, torch.inference_mode():
            output = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=_tokenizer.pad_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )
        generated = _tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local inference failed: {type(exc).__name__}") from exc
    parsed = validate_output(generated)
    return ChatResponse(
        reply=parsed["reply"],
        action=parsed["action"],
        model=MODEL_PATH.name,
        retrieved_knowledge=[item["title"] for item in retrieved],
    )
