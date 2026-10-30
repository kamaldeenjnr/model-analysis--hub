#!/usr/bin/env python3
"""Build QuantAI v4 training and held-out evaluation data.

Pipeline (deterministic for a given seed):
  1. For each intent, generate examples on scenarios from the split
     (training scenarios for train, held-out scenarios for eval; QuantAI's
     built-in example dataset appears in both, with different wording).
  2. Render each example in the real QuantAI app (render.js): the exact chat
     context, earlier analyses, and an engine run of every expected action.
  3. Keep an example only if every action ran without error and every named
     variable is visible in the context. Grounded answers are filled from the
     engine's own write-up, so their numbers are exact.
  4. Write data/v4/{train,eval}.jsonl plus a build report.

Usage: python build.py [--out ../data/v4] [--repo ../..]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from concepts import deeper_examples, handbook_examples  # noqa: E402
from intents import ALLVARS, INTENTS, PH, Ctx, code_name  # noqa: E402
from scenarios import BUILTIN, EVAL, EXPECTED_TYPES, TRAIN, generate_rows  # noqa: E402

# Examples per intent: (train, eval)
COUNTS = {
    "describe_some": (25, 5), "describe_by": (25, 5), "describe_all": (15, 3),
    "compare": (50, 10), "compare_paired": (15, 4), "correlation": (15, 4),
    "reg_linear": (35, 6), "reg_auto": (40, 8), "reg_logistic": (45, 8), "reg_modpoisson": (55, 10),
    "reg_poisson": (25, 5), "reg_rate": (85, 16), "reg_ordinal": (20, 4), "reg_multinomial": (15, 3), "reg_mixed": (35, 8),
    "surv_group": (30, 6), "surv_cox": (30, 6), "surv_compare": (35, 8),
    "forecast": (40, 8), "check": (15, 3), "fix": (12, 3), "report": (12, 3), "open": (60, 12),
    "multi": (35, 7), "followup": (50, 10),
    "grounded": (35, 7), "refuse_invent": (15, 4), "causal": (15, 4), "unsupported": (35, 8),
    "missing_variable": (25, 5), "missing_predictors": (30, 6), "crude_rate": (15, 4),
    "clustered_nonmixed": (20, 5), "rate_no_exposure": (10, 3), "unsuitable": (10, 3),
}
BUILTIN_SHARE = 0.15
SERIES_INTENTS = {"forecast"}


def resolve(obj, mapping: dict[str, str]):
    """Replace @@key@@ placeholders inside any string, list or dict."""
    if isinstance(obj, str):
        if obj in mapping:
            return mapping[obj]
        return re.sub(r"@@([A-Za-z0-9_]+)@@", lambda m: mapping.get(m.group(0), m.group(0)), obj)
    if isinstance(obj, list):
        out = []
        for item in obj:
            if item == ALLVARS:
                out.extend(mapping[ALLVARS])
            else:
                out.append(resolve(item, mapping))
        return out
    if isinstance(obj, dict):
        return {k: resolve(v, mapping) for k, v in obj.items()}
    return obj


def make_examples(split: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = [s for s in (TRAIN if split == "train" else EVAL) if not s.series]
    series = [s for s in (TRAIN if split == "train" else EVAL) if s.series]
    examples = []
    for intent, (n_train, n_eval) in COUNTS.items():
        want = n_train if split == "train" else n_eval
        made, attempts = 0, 0
        while made < want and attempts < want * 40:
            attempts += 1
            if intent in SERIES_INTENTS:
                sc = rng.choice(series)
            else:
                sc = BUILTIN if rng.random() < BUILTIN_SHARE else rng.choice(pool)
            ex = INTENTS[intent](Ctx(sc, split, random.Random(rng.random())))
            if not ex:
                continue
            ex.update(split=split, sid=sc.sid, seed=rng.randrange(10**9))
            examples.append(ex)
            made += 1
        if made < want:
            print(f"warning: {split} {intent}: made {made}/{want}", file=sys.stderr)
    return examples


def concept_examples(split: str, kb: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = [s for s in (TRAIN if split == "train" else EVAL) if not s.series]
    out = handbook_examples(kb, split, rng) + deeper_examples(split)
    for ex in out:
        sc = BUILTIN if rng.random() < 0.35 else rng.choice(pool)
        ex.update(split=split, sid=sc.sid, seed=rng.randrange(10**9))
    return out


def scenario_of(sid: str):
    if sid == BUILTIN.sid:
        return BUILTIN
    for s in TRAIN + EVAL:
        if s.sid == sid:
            return s
    raise KeyError(sid)


def to_job(i: int, ex: dict) -> tuple[dict, dict]:
    sc = scenario_of(ex["sid"])
    mapping = {PH.format(v.key): (v.key if sc.builtin else code_name(v.header)) for v in sc.variables}
    mapping[ALLVARS] = [mapping[PH.format(v.key)] for v in sc.variables if v.kind not in {"date"}]
    if ex.get("date_key"):
        mapping["@@__DATE__@@"] = mapping[PH.format(ex["date_key"])]
    if sc.builtin:
        dataset = "builtin"
    else:
        headers, rows = generate_rows(sc, random.Random(ex["seed"]), issues=bool(ex.get("issues")))
        dataset = {"headers": headers, "rows": rows, "source": sc.title}
    action = resolve(ex["action"], mapping)
    actions = [] if action is None else (action if isinstance(action, list) else [action])
    job = {"id": i, "dataset": dataset, "pre_actions": resolve(ex.get("pre_actions", []), mapping), "actions": actions}
    ex["_resolved"] = {"action": action, "history": resolve(ex.get("history", []), mapping),
                       "message": resolve(ex["message"], mapping), "reply": ex["reply"], "mapping": mapping}
    return job, ex


def render(repo: Path, jobs: list[dict], want_kb: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        jobs_path, out_path = Path(tmp) / "jobs.jsonl", Path(tmp) / "out.jsonl"
        with jobs_path.open("w", encoding="utf-8") as f:
            for job in jobs:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")
            if want_kb:
                f.write(json.dumps({"id": "kbprobe", "dataset": "builtin", "want_kb": True}) + "\n")
        subprocess.run(["node", str(HERE / "render.js"), str(repo), str(jobs_path), str(out_path)], check=True)
        results = {}
        for line in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            results[r["id"]] = r
        return results


GROUND_CAUTION = [
    "", " These are associations in your data; they don't show cause and effect on their own.",
    " Check the assumptions and warnings in the full result before you report it.",
]


def finalize(ex: dict, res: dict, drop: collections.Counter) -> dict | None:
    if "fatal" in res:
        drop["render_error"] += 1
        return None
    r = ex["_resolved"]
    visible = {v["name"]: v for v in res["vars"] if v["type"] != "id"}
    names_in_context = set(visible)
    # every action must have run in the engine
    if any(not v.get("ok") for v in res["pre"]):
        drop["pre_action_failed"] += 1
        return None
    if any(not v.get("ok") for v in res["verify"]):
        drop[f"action_failed:{ex['intent']}"] += 1
        ex["_error"] = [v.get("error") for v in res["verify"] if not v.get("ok")]
        return None
    # QuantAI must have typed every variable as the labels assume (e.g. a count, not an ordinal scale)
    sc = scenario_of(ex["sid"])
    if not sc.builtin:
        by_label = {v["label"]: v["type"] for v in res["vars"]}
        wrong = [v.header for v in sc.variables if by_label.get(v.header) not in EXPECTED_TYPES[v.kind]]
        if wrong:
            drop["unexpected_variable_type"] += 1
            ex["_error"] = [f"unexpected type for {wrong}"]
            return None
    action = r["action"]
    # forecast: the date variable is only named if the model can see it in the context
    if ex["intent"] == "forecast" and isinstance(action, dict) and action.get("date") not in names_in_context:
        action = dict(action, date=None)
    # every variable named in an action must be visible to the model
    def named(a):
        out = []
        for k in ("outcome", "exposure", "time", "event", "group", "cluster", "offset", "value", "date"):
            if isinstance(a.get(k), str):
                out.append(a[k])
        for k in ("vars", "predictors", "covariates"):
            out += list(a.get(k) or [])
        return out
    for a in ([] if action is None else (action if isinstance(action, list) else [action])):
        missing = [n for n in named(a) if n not in names_in_context]
        if missing:
            drop["variable_not_in_context"] += 1
            ex["_error"] = [f"not in context: {missing}; types: {[ (v['name'], v['type']) for v in res['vars'] if v['name'] in missing]}"]
            return None
    reply = r["reply"]
    if reply in ("@@__GROUND__@@", "@@__GROUND_N__@@"):
        pre = res["pre"][-1]
        if not pre.get("writeup"):
            drop["no_writeup"] += 1
            return None
        if reply == "@@__GROUND_N__@@":
            reply = f"{pre['n']} observations were included in {pre['title']} ({pre['method']})."
        else:
            reply = f"Here is what QuantAI's {pre['method'].split(' (')[0].lower()} found: {pre['writeup']}" + random.Random(ex["seed"]).choice(GROUND_CAUTION)
    completion = json.dumps({"reply": reply, "action": action}, ensure_ascii=False, separators=(",", ":"))
    meta = {"intent": ex["intent"], "scenario": ex["sid"], "split": ex["split"],
            "var_types": {n: v["type"] for n, v in visible.items()}}
    if ex.get("symmetric"):
        meta["symmetric"] = True
    return {"message": r["message"], "context": res["context"], "history": r["history"], "completion": completion, "meta": meta}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE.parent / "data" / "v4")
    parser.add_argument("--repo", type=Path, default=HERE.parent.parent)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    kb = render(args.repo, [], want_kb=True)["__kb__"]["kb"]
    report = {"seed": args.seed, "handbook_entries": len(kb), "splits": {}}
    for split, seed in (("train", args.seed), ("eval", args.seed + 1)):
        examples = make_examples(split, seed) + concept_examples(split, kb, seed + 7)
        jobs, exs = zip(*(to_job(i, ex) for i, ex in enumerate(examples)))
        results = render(args.repo, list(jobs))
        drop, rows, seen, failures = collections.Counter(), [], set(), []
        for i, ex in enumerate(exs):
            row = finalize(ex, results[i], drop)
            if row is None:
                if ex.get("_error"):
                    failures.append({"intent": ex["intent"], "message": ex["_resolved"]["message"], "error": ex["_error"]})
                continue
            key = (row["message"].casefold(), row["meta"]["scenario"])
            if key in seen:
                drop["duplicate_message"] += 1
                continue
            seen.add(key)
            rows.append(row)
        random.Random(seed).shuffle(rows)
        path = args.out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        report["splits"][split] = {
            "rows": len(rows), "sha256": sha256(path),
            "by_intent": dict(sorted(collections.Counter(r["meta"]["intent"] for r in rows).items())),
            "null_action_rows": sum(json.loads(r["completion"])["action"] is None for r in rows),
            "dropped": dict(drop), "engine_failures_sample": failures[:15],
        }
        print(f"{split}: {len(rows)} rows written to {path}; dropped {dict(drop)}")
    (args.out / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
