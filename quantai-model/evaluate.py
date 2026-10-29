#!/usr/bin/env python3
"""Evaluate QuantAI model outputs for JSON and tool-routing correctness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are QuantAI, a research-methods and statistics assistant. "
    "Return exactly one JSON object with keys reply and action. "
    "Use only supported action types: describe, compare, regression, survival, "
    "forecast, check, fix, report, open. If the request is a question rather "
    "than a request to run a tool, use action null. Never invent result values "
    "or variable names. The existing deterministic engine computes all statistics."
)
SUPPORTED_ACTIONS = {"describe", "compare", "regression", "survival", "forecast", "check", "fix", "report", "open"}


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        expected = json.loads(case["completion"])
        if set(expected) != {"reply", "action"}:
            raise ValueError(f"{path}:{line_no}: invalid expected output schema")
        cases.append({"prompt": case["prompt"], "expected": expected})
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def normalize_action(action: Any) -> Any:
    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local LoRA adapter directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--cases", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, args.model)
    model.eval()

    rows = []
    valid_json = 0
    valid_schema = 0
    action_type_correct = 0
    exact_action = 0
    for case in cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        parsed = None
        try:
            parsed = json.loads(generated)
            valid_json += 1
        except json.JSONDecodeError:
            pass
        expected = case["expected"]
        schema_ok = isinstance(parsed, dict) and set(parsed) == {"reply", "action"} and isinstance(parsed.get("reply"), str)
        if schema_ok:
            valid_schema += 1
            pred_action = parsed.get("action")
            if pred_action is None or (isinstance(pred_action, dict) and pred_action.get("type") in SUPPORTED_ACTIONS):
                if (pred_action is None and expected.get("action") is None) or (
                    isinstance(pred_action, dict) and isinstance(expected.get("action"), dict)
                    and pred_action.get("type") == expected["action"].get("type")
                ):
                    action_type_correct += 1
                if normalize_action(pred_action) == normalize_action(expected.get("action")):
                    exact_action += 1
        rows.append({
            "prompt": case["prompt"],
            "expected": expected,
            "generated_text": generated,
            "parsed": parsed,
            "valid_json": isinstance(parsed, dict),
            "valid_schema": schema_ok,
            "action_type_match": (
                schema_ok and (
                    (parsed.get("action") is None and expected.get("action") is None)
                    or (
                        isinstance(parsed.get("action"), dict)
                        and isinstance(expected.get("action"), dict)
                        and parsed["action"].get("type") == expected["action"].get("type")
                    )
                )
            ),
            "exact_action_match": schema_ok and normalize_action(parsed.get("action")) == normalize_action(expected.get("action")),
        })

    n = len(cases)
    report = {
        "model_path": args.model,
        "base_model": args.base_model,
        "cases": n,
        "valid_json_rate": valid_json / n,
        "valid_schema_rate": valid_schema / n,
        "action_type_accuracy": action_type_correct / n,
        "exact_action_match_rate": exact_action / n,
        "warning": "Synthetic starter cases are only a smoke test, not evidence of research-grade performance.",
        "results": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
