#!/usr/bin/env python3
"""Evaluate QuantAI outputs for JSON schema and supported action-routing accuracy."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are QuantAI, a research-methods and statistics assistant. "
    "Return exactly one JSON object with keys reply and action. "
    "Supported action types: describe, compare, regression, survival, forecast, "
    "check, fix, report, open. For a question rather than a tool request use action null. "
    "Never invent result values or variable names. The deterministic engine computes statistics."
)
SUPPORTED_ACTIONS = {"describe", "compare", "regression", "survival", "forecast", "check", "fix", "report", "open"}


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            expected = json.loads(case["completion"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid evaluation case: {exc}") from exc
        if not isinstance(case.get("prompt"), str) or not isinstance(expected, dict):
            raise ValueError(f"{path}:{line_no}: prompt/completion schema is invalid")
        if set(expected) != {"reply", "action"} or not isinstance(expected["reply"], str):
            raise ValueError(f"{path}:{line_no}: expected completion keys are reply and action")
        cases.append({"prompt": case["prompt"], "expected": expected})
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def parse_output(text: str) -> Any:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def valid_action(action: Any) -> bool:
    if action is None:
        return True
    items = action if isinstance(action, list) else [action]
    return bool(items) and all(isinstance(item, dict) and item.get("type") in SUPPORTED_ACTIONS for item in items)


def action_types(action: Any) -> Any:
    if action is None:
        return None
    if isinstance(action, list):
        return [item.get("type") if isinstance(item, dict) else None for item in action]
    return action.get("type") if isinstance(action, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local LoRA adapter directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--cases", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate the untouched base model for comparison")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    tokenizer_source = args.base_model if args.baseline_only else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not args.baseline_only:
        model = PeftModel.from_pretrained(model, args.model)
    model.eval()

    rows = []
    json_ok = schema_ok = action_ok = type_match = exact_match = 0
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
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        parsed = parse_output(generated)
        if parsed is not None:
            json_ok += 1
        schema_valid = isinstance(parsed, dict) and set(parsed) == {"reply", "action"} and isinstance(parsed.get("reply"), str)
        action_valid = schema_valid and valid_action(parsed.get("action"))
        if schema_valid:
            schema_ok += 1
        expected_action = case["expected"].get("action")
        predicted_action = parsed.get("action") if schema_valid else None
        same_type = action_types(predicted_action) == action_types(expected_action)
        same_action = predicted_action == expected_action
        if action_valid:
            action_ok += 1
        if action_valid and same_type:
            type_match += 1
        if action_valid and same_action:
            exact_match += 1
        rows.append({
            "prompt": case["prompt"],
            "expected": case["expected"],
            "generated_text": generated,
            "parsed": parsed,
            "valid_json": parsed is not None,
            "valid_schema": schema_valid,
            "valid_action": bool(action_valid),
            "action_type_match": bool(action_valid and same_type),
            "exact_action_match": bool(action_valid and same_action),
        })

    n = len(cases)
    report = {
        "evaluated_model": args.base_model if args.baseline_only else args.model,
        "base_model": args.base_model,
        "mode": "untuned baseline" if args.baseline_only else "fine-tuned adapter",
        "cases": n,
        "valid_json_rate": json_ok / n,
        "valid_schema_rate": schema_ok / n,
        "supported_action_rate": action_ok / n,
        "action_type_accuracy": type_match / n,
        "exact_action_match_rate": exact_match / n,
        "warning": "Starter cases are synthetic smoke tests, not evidence of research-grade performance.",
        "results": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
