#!/usr/bin/env python3
"""Validate QuantAI's JSONL training/evaluation fixtures before training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUPPORTED = {"describe", "compare", "regression", "survival", "forecast", "check", "fix", "report", "open"}
TOP_LEVEL = {"reply", "action"}
ACTION_KEYS = {
    "describe": {"type", "vars", "group"},
    "compare": {"type", "outcome", "exposure", "paired"},
    "regression": {"type", "outcome", "predictors", "model", "cluster", "offset"},
    "survival": {"type", "time", "event", "group", "covariates"},
    "forecast": {"type", "date", "value", "horizon"},
    "check": {"type"}, "fix": {"type"}, "report": {"type"},
    "open": {"type", "tool"},
}
OPEN_TOOLS = {"data", "describe", "compare", "regression", "survival", "forecast", "export", "question", "design", "sample", "test", "checklist"}
REGRESSION_MODELS = {"auto", "linear", "logistic", "modpoisson", "poisson", "ordinal", "multinomial", "mixed"}


def valid_action_item(item: Any) -> bool:
    if not isinstance(item, dict) or not isinstance(item.get("type"), str):
        return False
    kind = item["type"]
    if kind not in ACTION_KEYS:
        return False
    if set(item) != ACTION_KEYS[kind]:
        return False
    if kind == "describe":
        return isinstance(item["vars"], list) and all(isinstance(v, str) for v in item["vars"]) and (item["group"] is None or isinstance(item["group"], str))
    if kind == "compare":
        return isinstance(item["outcome"], str) and isinstance(item["exposure"], str) and isinstance(item["paired"], bool)
    if kind == "regression":
        return (isinstance(item["outcome"], str) and isinstance(item["predictors"], list)
                and all(isinstance(v, str) for v in item["predictors"])
                and isinstance(item["model"], str) and item["model"] in REGRESSION_MODELS
                and (item["cluster"] is None or isinstance(item["cluster"], str))
                and ("offset" in item and (item["offset"] is None or (isinstance(item["offset"], str) and bool(item["offset"].strip()))))
                and (item["offset"] is None or (item["model"] == "poisson" and item["offset"] != item["outcome"] and item["offset"] not in item["predictors"])))
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
        return isinstance(item["tool"], str) and item["tool"] in OPEN_TOOLS
    return kind in {"check", "fix", "report"}


def valid_action_payload(action: Any) -> bool:
    if action is None:
        return True
    items = action if isinstance(action, list) else [action]
    return bool(items) and len(items) <= 4 and all(valid_action_item(item) for item in items)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"{path}:{number}: prompt must be a non-empty string")
        if not isinstance(row.get("completion"), str):
            raise ValueError(f"{path}:{number}: completion must be a JSON string")
        try:
            result = json.loads(row["completion"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: completion is not valid JSON: {exc}") from exc
        if not isinstance(result, dict) or set(result) != TOP_LEVEL or not isinstance(result.get("reply"), str):
            raise ValueError(f"{path}:{number}: expected exactly the keys reply and action")
        action = result["action"]
        if not valid_action_payload(action):
            raise ValueError(
                f"{path}:{number}: action payload is incomplete, has invalid field types, "
                "contains unsupported fields, or uses an unsupported action"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no examples found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--min-train", type=int, default=20)
    parser.add_argument("--min-eval", type=int, default=10)
    args = parser.parse_args()

    if args.train.resolve() == args.eval.resolve():
        raise SystemExit("Training and evaluation paths must be different.")
    train = read_rows(args.train)
    evaluation = read_rows(args.eval)
    train_prompts = {row["prompt"].strip().casefold() for row in train}
    eval_prompts = {row["prompt"].strip().casefold() for row in evaluation}
    overlap = train_prompts & eval_prompts
    errors = []
    if len(train) < args.min_train:
        errors.append(f"training examples: {len(train)} (minimum {args.min_train})")
    if len(evaluation) < args.min_eval:
        errors.append(f"evaluation examples: {len(evaluation)} (minimum {args.min_eval})")
    if overlap:
        errors.append(f"{len(overlap)} exact prompt(s) occur in both splits")
    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))

    print(f"Training examples: {len(train)}")
    print(f"Held-out evaluation examples: {len(evaluation)}")
    print("JSON schema and supported action types: valid")
    print("Exact prompt overlap: none")
    print("Note: structural checks do not establish example quality or model performance.")


if __name__ == "__main__":
    main()
