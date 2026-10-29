#!/usr/bin/env python3
"""Audit QuantAI v4 rows against the action contract and the engine's rules.

Independent of how the rows were generated: it reads only the JSONL files.
Exit status 1 if any row breaks a rule. Usage: python audit.py file.jsonl [...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MODEL_OUTCOME = {
    "linear": {"continuous"}, "logistic": {"binary"}, "modpoisson": {"binary"}, "poisson": {"count", "continuous"},
    "ordinal": {"ordinal"}, "multinomial": {"categorical"}, "mixed": {"continuous"},
    "auto": {"continuous", "count", "binary", "ordinal", "categorical"},
}
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
CONCEPT_INTENTS = {"concept_handbook", "concept_deeper"}


def problems(row: dict) -> list[str]:
    out = []
    types = row["meta"]["var_types"]
    completion = json.loads(row["completion"])
    action, reply = completion["action"], completion["reply"]
    items = [] if action is None else (action if isinstance(action, list) else [action])

    def visible(name, field):
        if name not in types:
            out.append(f"{field} {name!r} is not in the context")
            return False
        return True

    for a in items:
        t = a["type"]
        if t == "regression":
            y, xs, m, cl, off = a["outcome"], a["predictors"], a["model"], a["cluster"], a["offset"]
            if visible(y, "outcome") and types[y] not in MODEL_OUTCOME[m]:
                out.append(f"model {m} does not suit {types[y]} outcome {y}")
            if not xs:
                out.append("regression without predictors")
            if len(set(xs)) != len(xs):
                out.append("duplicate predictors")
            for x in xs:
                visible(x, "predictor")
            if y in xs:
                out.append("outcome is also a predictor")
            if cl is not None:
                visible(cl, "cluster")
                if m != "mixed":
                    out.append(f"cluster with model {m} (only the mixed model uses it)")
                if cl in xs:
                    out.append("cluster is also a predictor")
            elif m == "mixed":
                out.append("mixed model without a cluster")
            if off is not None:
                if visible(off, "offset") and types[off] not in {"continuous", "count"}:
                    out.append("offset is not numeric")
                if m != "poisson":
                    out.append("offset without the Poisson model")
                if off in xs or off == y:
                    out.append("offset is also the outcome or a predictor")
        elif t == "compare":
            if a["outcome"] == a["exposure"]:
                out.append("compare outcome equals exposure")
            for f in ("outcome", "exposure"):
                visible(a[f], f)
            if a["paired"] and {types.get(a["outcome"]), types.get(a["exposure"])} - {"continuous", "count"}:
                out.append("paired comparison of non-numeric variables")
        elif t == "survival":
            if visible(a["time"], "time") and types[a["time"]] not in {"continuous", "count"}:
                out.append("survival time is not numeric")
            if visible(a["event"], "event") and types[a["event"]] != "binary":
                out.append("event is not binary")
            if a["group"] is not None and visible(a["group"], "group") and types[a["group"]] not in {"binary", "categorical", "ordinal"}:
                out.append("survival group is not categorical")
            for cv in a["covariates"]:
                visible(cv, "covariate")
                if cv in {a["time"], a["event"], a["group"]}:
                    out.append("covariate repeats time, event or group")
        elif t == "describe":
            for v in a["vars"]:
                visible(v, "describe var")
            if a["group"] is not None and (not visible(a["group"], "group") or a["group"] not in a["vars"]):
                out.append("describe group missing from vars")
        elif t == "forecast":
            visible(a["value"], "forecast value")
            if a["date"] is not None:
                visible(a["date"], "forecast date")
    # no number in a reply unless the user or QuantAI's context already contains it
    if row["meta"]["intent"] not in CONCEPT_INTENTS:
        allowed = set(NUMBER.findall(row["message"] + " " + row["context"] + " " + json.dumps(row.get("history", []))))
        invented = [n for n in NUMBER.findall(reply) if n not in allowed and n not in {"0", "1"}]
        if invented:
            out.append(f"reply contains numbers not in message or context: {invented[:5]}")
    return out


def main(paths: list[str]) -> int:
    total_bad = 0
    for path in paths:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        bad = 0
        for i, row in enumerate(rows, 1):
            p = problems(row)
            if p:
                bad += 1
                if bad <= 20:
                    print(f"{path}:{i} [{row['meta']['intent']}] {row['message'][:70]!r}: {'; '.join(p)}")
        print(f"{path}: {len(rows)} rows, {bad} with problems")
        total_bad += bad
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
