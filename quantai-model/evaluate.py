#!/usr/bin/env python3
"""Evaluate a QuantAI model on held-out cases.

The model sees exactly what the local API sends it (prompting.py), and every
output is scored by scoring.py: contract validity, strict and lenient action
match, field-level accuracy, correct declining/clarifying, numbers that are
not in the context, and per-intent results with 95% confidence intervals.

Examples
  # fine-tuned adapter (v4 prompt)
  python evaluate.py --model outputs/quantai-qwen3-4b-lora-v4 --cases data/v4/eval.jsonl --report outputs/v4-eval.json
  # untouched base model with the same prompt
  python evaluate.py --baseline-only --cases data/v4/eval.jsonl --report outputs/base-eval.json
  # v3 adapter, served with the v3 prompt it was trained on
  python evaluate.py --model outputs/quantai-qwen3-4b-lora-v3 --prompt-version v3 --cases data/v4/eval.jsonl --report outputs/v3-eval.json
  # exported (merged) model in fp16
  python evaluate.py --model outputs/quantai-merged-model-v4 --precision float16 --cases data/v4/eval.jsonl --report outputs/v4-merged-eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import prompting  # noqa: E402
import scoring  # noqa: E402

DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"


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
        if not (isinstance(case.get("prompt"), str) or isinstance(case.get("message"), str)):
            raise ValueError(f"{path}:{line_no}: a case needs a prompt (v3) or message (v4)")
        if set(expected) != {"reply", "action"} or not isinstance(expected["reply"], str):
            raise ValueError(f"{path}:{line_no}: expected completion keys are reply and action")
        cases.append(case)
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def load_model(args):
    is_adapter = bool(args.model) and (Path(args.model) / "adapter_config.json").exists()
    weights = args.base_model if (args.baseline_only or is_adapter or not args.model) else args.model
    tokenizer_source = args.base_model if args.baseline_only or not args.model else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # batched generation continues from the right edge
    load_mode = "cpu-float32"
    kwargs: dict[str, Any] = {"torch_dtype": torch.float32}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory free: {free / 2**30:.1f} of {total / 2**30:.1f} GiB")
        # Keep every layer on the GPU; device_map="auto" would silently offload to the CPU.
        kwargs = {"device_map": {"": 0}}
        if args.precision == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
            load_mode = "cuda-4bit-nf4"
        else:
            kwargs["torch_dtype"] = torch.float16
            load_mode = "cuda-float16"
    try:
        model = AutoModelForCausalLM.from_pretrained(weights, **kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit("The model does not fit in free GPU memory. Run !nvidia-smi, restart the session "
                         "(Runtime > Restart session) and rerun, or use --precision 4bit.") from exc
    if is_adapter and not args.baseline_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model)
    model.eval()
    kind = "untuned baseline" if args.baseline_only or not args.model else ("fine-tuned adapter" if is_adapter else "merged model")
    print(f"Loaded {weights}{' + adapter ' + args.model if is_adapter and not args.baseline_only else ''} as {load_mode} ({kind})")
    return model, tokenizer, load_mode, kind


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    if torch.cuda.is_available():
        batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.inference_mode():
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    width = batch["input_ids"].shape[1]
    return [tokenizer.decode(row[width:], skip_special_tokens=True).strip() for row in out]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="", help="LoRA adapter directory or merged model directory")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate the untouched base model")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "v4" / "eval.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "evaluation.json")
    parser.add_argument("--prompt-version", default=prompting.DEFAULT_PROMPT_VERSION, choices=sorted(prompting.SYSTEM_PROMPTS))
    parser.add_argument("--knowledge-db", type=Path, default=ROOT / "knowledge.sqlite3")
    parser.add_argument("--precision", choices=["4bit", "float16"], default="4bit",
                        help="GPU loading precision; use the same value for every model you compare.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4, help="Cases generated together; lower it if the GPU runs out of memory.")
    parser.add_argument("--limit", type=int, default=0, help="Score a stratified subset of this many cases (0 = all).")
    args = parser.parse_args()
    if not args.model and not args.baseline_only:
        parser.error("pass --model DIR or --baseline-only")

    cases = load_cases(args.cases)
    if args.limit:
        cases = prompting.stratified_subset(cases, args.limit)
    sections = prompting.initialise_knowledge(prompting.knowledge_sources(ROOT), args.knowledge_db)
    model, tokenizer, load_mode, kind = load_model(args)
    prompts = [tokenizer.apply_chat_template(prompting.messages_for_row(c, args.knowledge_db, args.prompt_version),
                                             tokenize=False, add_generation_prompt=True) for c in cases]
    # Shortest first keeps padding small inside each batch; results are restored to file order.
    order = sorted(range(len(cases)), key=lambda i: len(prompts[i]))
    outputs: list[str] = [""] * len(cases)
    started = time.time()
    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        for i, text in zip(idx, generate(model, tokenizer, [prompts[i] for i in idx], args.max_new_tokens)):
            outputs[i] = text
        done = min(start + args.batch_size, len(order))
        print(f"  {done}/{len(order)} cases, {time.time() - started:.0f}s", flush=True)

    scored = [scoring.score_case(c, o) for c, o in zip(cases, outputs)]
    summary = scoring.summarise(cases, scored)
    report = {
        "evaluated_model": args.base_model if args.baseline_only else args.model,
        "base_model": args.base_model, "mode": kind, "load_mode": load_mode, "prompt_version": args.prompt_version,
        "cases_file": str(args.cases), "knowledge_sections": sections, **summary,
        "warning": "Synthetic held-out cases rendered from QuantAI's real context. Strong scores show the model fits QuantAI's "
                   "action contract on these scenarios; they are not evidence of expert-level research judgement.",
        "results": [{"message": c.get("message", c.get("prompt")), "prompt": c.get("prompt", c.get("message")),
                     "expected": json.loads(c["completion"]), **{k: v for k, v in s.items()}} for c, s in zip(cases, scored)],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    show = {k: v for k, v in report.items() if k not in {"results", "field_scores", "by_intent"}}
    print(json.dumps(show, indent=2))
    print("\nLenient match by intent:")
    for intent, r in summary["by_intent"].items():
        print(f"  {intent:22} {r['lenient_match']:6.1%}  (n={r['n']})")


if __name__ == "__main__":
    main()
