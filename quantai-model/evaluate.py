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

SYSTEM_PROMPT = """You are QuantAI, a research-methods and biostatistics assistant.
Return exactly one JSON object with exactly two top-level keys: "reply" and "action".
For a question or explanation that does not request a tool, set "action": null.
Never invent variable names, results, coefficients, p-values, confidence intervals, sample sizes, or diagnostics. The deterministic engine computes statistics. Ask a concise clarification and set action to null when required information is unavailable.
For a tool action, use only the schemas below. Include EVERY listed key, including keys whose value is null, false, or an empty list. Do not omit keys and do not add extra keys.
- describe: {"type":"describe","vars":["exact_variable_name"],"group":null}; vars must list the requested variables from context and group is a variable name or null.
- compare: {"type":"compare","outcome":"exact_variable_name","exposure":"exact_variable_name","paired":false}; paired must always be true or false.
- regression: {"type":"regression","outcome":"exact_variable_name","predictors":["exact_variable_name"],"model":"auto","cluster":null}; model must be one of auto, linear, logistic, modpoisson, poisson, ordinal, multinomial, mixed. Use the requested model; use auto only when the user has not specified a model. cluster is a variable name or null.
- survival: {"type":"survival","time":"exact_variable_name","event":"exact_variable_name","group":null,"covariates":[]}; group is a variable name or null and covariates is always a list.
- forecast: {"type":"forecast","date":null,"value":"exact_variable_name","horizon":4}; date is the date/time variable name or null, value is the series variable, and horizon is an integer from 1 to 120.
- check: {"type":"check"}
- fix: {"type":"fix"}
- report: {"type":"report"}
- open: {"type":"open","tool":"regression"}; tool must be one of data, describe, compare, regression, survival, forecast, export, question, design, sample, test, checklist.
If a request needs a feature or parameter that the action schema cannot represent, do not silently drop it: ask for clarification or explain the limitation and set action to null.
Distinguish association from causation and odds ratios from risk or prevalence ratios. Do not claim an analysis has run until the deterministic engine returns results.
"""
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


ACTION_KEYS = {
    "describe": {"type", "vars", "group"},
    "compare": {"type", "outcome", "exposure", "paired"},
    "regression": {"type", "outcome", "predictors", "model", "cluster"},
    "survival": {"type", "time", "event", "group", "covariates"},
    "forecast": {"type", "date", "value", "horizon"},
    "check": {"type"}, "fix": {"type"}, "report": {"type"},
    "open": {"type", "tool"},
}
OPEN_TOOLS = {"data", "describe", "compare", "regression", "survival", "forecast", "export", "question", "design", "sample", "test", "checklist"}
REGRESSION_MODELS = {"auto", "linear", "logistic", "modpoisson", "poisson", "ordinal", "multinomial", "mixed"}


def valid_action_item(item: Any) -> bool:
    if not isinstance(item, dict) or item.get("type") not in ACTION_KEYS:
        return False
    kind = item["type"]
    if set(item) != ACTION_KEYS[kind]:
        return False
    if kind == "describe":
        return isinstance(item["vars"], list) and all(isinstance(v, str) for v in item["vars"]) and (item["group"] is None or isinstance(item["group"], str))
    if kind == "compare":
        return isinstance(item["outcome"], str) and isinstance(item["exposure"], str) and isinstance(item["paired"], bool)
    if kind == "regression":
        return (isinstance(item["outcome"], str) and isinstance(item["predictors"], list)
                and all(isinstance(v, str) for v in item["predictors"])
                and item["model"] in REGRESSION_MODELS
                and (item["cluster"] is None or isinstance(item["cluster"], str)))
    if kind == "survival":
        return (isinstance(item["time"], str) and isinstance(item["event"], str)
                and (item["group"] is None or isinstance(item["group"], str))
                and isinstance(item["covariates"], list)
                and all(isinstance(v, str) for v in item["covariates"]))
    if kind == "forecast":
        return ((item["date"] is None or isinstance(item["date"], str))
                and isinstance(item["value"], str) and isinstance(item["horizon"], int)
                and not isinstance(item["horizon"], bool) and 1 <= item["horizon"] <= 120)
    if kind == "open":
        return item["tool"] in OPEN_TOOLS
    return kind in {"check", "fix", "report"}


def valid_action(action: Any) -> bool:
    if action is None:
        return True
    items = action if isinstance(action, list) else [action]
    return bool(items) and len(items) <= 4 and all(valid_action_item(item) for item in items)


def action_types(action: Any) -> Any:
    if action is None:
        return None
    if isinstance(action, list):
        return [item.get("type") if isinstance(item, dict) else None for item in action]
    return action.get("type") if isinstance(action, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local LoRA adapter directory")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
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
        # This rate now requires a complete, correctly typed action payload,
        # not merely a recognised action name. Keep the legacy key for reports.
        "supported_action_rate": action_ok / n,
        "action_contract_valid_rate": action_ok / n,
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
