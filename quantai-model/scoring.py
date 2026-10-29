#!/usr/bin/env python3
"""Scoring for QuantAI evaluations (pure Python, no model needed).

Strict match: the normalised action equals the expected one exactly.
Lenient match: the same analysis, allowing only differences that change
nothing in QuantAI's engine:
  - order of predictors, covariates and describe variables,
  - model "auto" versus the model the engine picks for that outcome type,
  - outcome and exposure swapped for a symmetric correlation.
Field-level scores show which part of an action goes wrong.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

from validate_data import valid_action_payload

NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
CONCEPT_INTENTS = {"concept_handbook", "concept_deeper"}
AUTO_MODEL = {"continuous": "linear", "binary": "logistic", "count": "poisson", "ordinal": "ordinal", "categorical": "multinomial"}


def parse_output(text: str) -> Any:
    """Parse model output the way serve.validate_output does: strip fences, take the outer JSON object."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def normalise(item: Any) -> Any:
    """serve.normalise_action_item: a grouping variable is always among the described vars."""
    if not isinstance(item, dict):
        return item
    item = dict(item)
    if item.get("type") == "regression" and "offset" not in item:
        item["offset"] = None
    if item.get("type") == "describe" and isinstance(item.get("vars"), list) and isinstance(item.get("group"), str) \
            and item["group"] not in item["vars"]:
        item["vars"] = [*item["vars"], item["group"]]
    return item


def as_list(action: Any) -> list:
    if action is None:
        return []
    return [normalise(a) for a in (action if isinstance(action, list) else [action])]


def lenient_item(item: dict, types: dict[str, str]) -> dict:
    out = dict(item)
    for key in ("predictors", "covariates", "vars"):
        if isinstance(out.get(key), list):
            out[key] = sorted(out[key])
    if out.get("type") == "regression" and out.get("model") == "auto":
        y_type = types.get(out.get("outcome"))
        out["model"] = "mixed" if (out.get("cluster") and y_type == "continuous") else AUTO_MODEL.get(y_type, "auto")
    return out


def same_action(expected: Any, actual: Any, types: dict[str, str], symmetric: bool = False, lenient: bool = False) -> bool:
    e, a = as_list(expected), as_list(actual)
    if len(e) != len(a):
        return False
    if not lenient:
        return e == a
    e = [lenient_item(x, types) for x in e]
    a = [lenient_item(x, types) for x in a]
    if e == a:
        return True
    if symmetric and len(e) == 1 and e[0].get("type") == "compare" and a[0].get("type") == "compare":
        swapped = dict(a[0], outcome=a[0].get("exposure"), exposure=a[0].get("outcome"))
        return swapped == e[0]
    return False


def numbers_not_in_context(reply: str, message: str, context: str, history: list) -> list[str]:
    allowed = set(NUMBER.findall(message + " " + context + " " + json.dumps(history or [])))
    return [n for n in NUMBER.findall(reply or "") if n not in allowed and n not in {"0", "1"}]


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


FIELDS = {
    "regression": ["outcome", "model", "cluster", "offset"],
    "compare": ["outcome", "exposure", "paired"],
    "survival": ["time", "event", "group"],
    "forecast": ["value", "date", "horizon"],
    "describe": ["group"],
    "open": ["tool"],
}
SET_FIELDS = {"regression": "predictors", "survival": "covariates", "describe": "vars"}


def score_case(case: dict, raw: str) -> dict:
    """Score one model output against one expected row."""
    expected = json.loads(case["completion"])
    meta = case.get("meta") or {}
    types = meta.get("var_types", {})
    parsed = parse_output(raw)
    json_ok = parsed is not None
    schema_ok = isinstance(parsed, dict) and set(parsed) == {"reply", "action"} and isinstance(parsed.get("reply"), str)
    actual_action = parsed.get("action") if schema_ok else None
    contract_ok = schema_ok and (actual_action is None or valid_action_payload([normalise(a) for a in as_list(actual_action)] if isinstance(actual_action, list) else normalise(actual_action)))
    exp_items, act_items = as_list(expected["action"]), as_list(actual_action) if contract_ok else []
    types_match = schema_ok and [i.get("type") for i in exp_items] == [i.get("type") for i in act_items]
    strict = bool(contract_ok and same_action(expected["action"], actual_action, types))
    lenient = bool(contract_ok and same_action(expected["action"], actual_action, types, meta.get("symmetric", False), lenient=True))
    fields: dict[str, bool] = {}
    if contract_ok and len(exp_items) == 1 and len(act_items) == 1 and exp_items[0].get("type") == act_items[0].get("type"):
        e, a = lenient_item(exp_items[0], types), lenient_item(act_items[0], types)
        kind = e["type"]
        for f in FIELDS.get(kind, []):
            fields[f"{kind}.{f}"] = e.get(f) == a.get(f)
        if kind in SET_FIELDS:
            key = SET_FIELDS[kind]
            es, as_ = set(e.get(key) or []), set(a.get(key) or [])
            fields[f"{kind}.{key}_exact"] = es == as_
            fields[f"{kind}.{key}_recall"] = (len(es & as_) / len(es)) if es else (1.0 if not as_ else 0.0)
            fields[f"{kind}.{key}_precision"] = (len(es & as_) / len(as_)) if as_ else (1.0 if not es else 0.0)
    reply = parsed.get("reply", "") if schema_ok else ""
    invented = [] if meta.get("intent") in CONCEPT_INTENTS else numbers_not_in_context(reply, case.get("message", case.get("prompt", "")), case.get("context", ""), case.get("history", []))
    return {
        "intent": meta.get("intent", "v3"),
        "expected_null": expected["action"] is None,
        "valid_json": json_ok, "valid_schema": schema_ok, "action_contract_valid": bool(contract_ok),
        "action_type_match": bool(types_match), "exact_action_match": strict, "lenient_action_match": lenient,
        "returned_null": schema_ok and actual_action is None,
        "fields": fields, "invented_numbers": invented,
        "parsed": parsed, "raw_output": raw,
    }


def summarise(cases: list[dict], scored: list[dict]) -> dict:
    n = len(scored)
    count = lambda key: sum(1 for s in scored if s[key])
    exp_null = [s for s in scored if s["expected_null"]]
    exp_act = [s for s in scored if not s["expected_null"]]
    headline = {}
    for key, name in [("valid_json", "valid_json_rate"), ("valid_schema", "valid_schema_rate"), ("action_contract_valid", "action_contract_valid_rate"),
                      ("action_type_match", "action_type_accuracy"), ("exact_action_match", "exact_action_match_rate"),
                      ("lenient_action_match", "lenient_action_match_rate")]:
        k = count(key)
        headline[name] = k / n if n else 0.0
        headline[name + "_ci95"] = wilson(k, n)
    headline["supported_action_rate"] = headline["action_contract_valid_rate"]  # legacy key used by older notebooks
    abstain_ok = sum(1 for s in exp_null if s["returned_null"])
    over_abstain = sum(1 for s in exp_act if s["returned_null"])
    headline["null_when_expected_rate"] = abstain_ok / len(exp_null) if exp_null else None
    headline["null_when_action_expected_rate"] = over_abstain / len(exp_act) if exp_act else None
    inv = sum(1 for s in scored if s["invented_numbers"])
    headline["replies_with_numbers_not_in_context_rate"] = inv / n if n else 0.0
    fields: dict[str, list[float]] = defaultdict(list)
    for s in scored:
        for f, v in s["fields"].items():
            fields[f].append(float(v))
    field_scores = {f: {"mean": round(sum(v) / len(v), 4), "n": len(v)} for f, v in sorted(fields.items())}
    by_intent: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        groups[s["intent"]].append(s)
    for intent, rows in sorted(groups.items()):
        k = sum(1 for s in rows if s["lenient_action_match"])
        by_intent[intent] = {"n": len(rows), "lenient_match": round(k / len(rows), 4), "strict_match": round(sum(1 for s in rows if s["exact_action_match"]) / len(rows), 4), "ci95": wilson(k, len(rows))}
    return {"cases": n, **headline, "field_scores": field_scores, "by_intent": by_intent}
