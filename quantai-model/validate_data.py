#!/usr/bin/env python3
"""Validate QuantAI's JSONL training/evaluation fixtures before training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUPPORTED = {"describe", "compare", "regression", "survival", "forecast", "check", "fix", "report", "open"}
TOP_LEVEL = {"reply", "action"}


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
        actions = action if isinstance(action, list) else [action]
        if action is not None and not isinstance(action, (dict, list)):
            raise ValueError(f"{path}:{number}: action must be an object, a list of objects, or null")
        for index, item in enumerate(actions):
            if item is None and action is None:
                continue
            if not isinstance(item, dict) or item.get("type") not in SUPPORTED:
                raise ValueError(f"{path}:{number}: action {index + 1} has an unsupported action type")
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
