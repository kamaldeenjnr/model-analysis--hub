#!/usr/bin/env python3
"""Fine-tune a small instruction model for QuantAI action routing with LoRA.

The training examples are synthetic development fixtures. Passing this script
does not mean the resulting model is suitable for production.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompting  # noqa: E402  identical prompt construction to serve.py

DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"
ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read v3 rows ({prompt, completion}) or v4 rows ({message, context, history, completion})."""
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            has_prompt = isinstance(row.get("prompt"), str)
            has_message = isinstance(row.get("message"), str)
            if has_prompt == has_message or not isinstance(row.get("completion"), str):
                raise ValueError(f"{path}:{line_no}: expected either prompt or message (not both) and a string completion")
            try:
                parsed = json.loads(row["completion"])
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: completion must be valid JSON: {exc}") from exc
            if set(parsed) != {"reply", "action"} or not isinstance(parsed["reply"], str):
                raise ValueError(f"{path}:{line_no}: completion must contain reply and action")
            rows.append(row)
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def build_dataset(rows: list[dict[str, Any]], tokenizer, max_length: int, db_path: Path, prompt_version: str) -> Dataset:
    """Tokenise rows exactly as serve.py builds its input. Never truncates: a row
    that does not fit is an error, because cutting the system prompt or the target
    answer would silently teach the wrong thing."""
    processed, lengths, too_long = [], [], []
    for index, row in enumerate(rows):
        messages = prompting.messages_for_row(row, db_path, prompt_version)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        completion_ids = tokenizer(row["completion"] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        total = len(prompt_ids) + len(completion_ids)
        lengths.append(total)
        if total > max_length:
            too_long.append((index + 1, total))
            continue
        processed.append({
            "input_ids": prompt_ids + completion_ids,
            "attention_mask": [1] * total,
            "labels": [-100] * len(prompt_ids) + completion_ids,
        })
    lengths.sort()
    print(f"Token lengths: n={len(lengths)} median={lengths[len(lengths) // 2]} "
          f"p95={lengths[int(len(lengths) * 0.95) - 1 if len(lengths) > 1 else 0]} max={lengths[-1]} (limit {max_length})")
    if too_long:
        raise SystemExit(
            f"{len(too_long)} example(s) exceed --max-length {max_length} tokens (first: row {too_long[0][0]} "
            f"with {too_long[0][1]}). Raise --max-length; rows are never truncated."
        )
    return Dataset.from_list(processed)


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(args, n_train: int, n_eval: int) -> None:
    """Record what this run was trained on so results can be reproduced and compared."""
    import platform
    import transformers
    import peft
    manifest = {
        "base_model": args.base_model,
        "prompt_version": args.prompt_version,
        "train_file": str(args.train), "train_sha256": sha256(args.train), "train_rows": n_train,
        "eval_file": str(args.eval), "eval_sha256": sha256(args.eval), "eval_rows": n_eval,
        "max_train_rows": args.max_train_rows, "eval_limit": args.eval_limit,
        "epochs": args.epochs, "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation, "max_length": args.max_length, "seed": 42,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.output / 'training_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=Path("outputs/quantai-lora"))
    parser.add_argument("--max-length", type=int, default=3584,
                        help="Maximum tokens per example (prompt + answer). Rows longer than this stop training.")
    parser.add_argument("--prompt-version", default=prompting.DEFAULT_PROMPT_VERSION, choices=sorted(prompting.SYSTEM_PROMPTS),
                        help="System prompt version; serve.py must use the same QUANTAI_PROMPT_VERSION.")
    parser.add_argument("--knowledge-db", type=Path, default=ROOT / "knowledge.sqlite3",
                        help="Retrieval index built from the knowledge files, as serve.py uses.")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="Train on a stratified subset of this many rows (0 = all).")
    parser.add_argument("--eval-limit", type=int, default=60,
                        help="Rows of the held-out file used for the loss printed during training (0 = all). "
                             "The full file is scored separately by evaluate.py.")
    parser.add_argument("--save-steps", type=int, default=10,
                        help="Write a resumable checkpoint every N optimizer steps.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the newest checkpoint in --output if one exists.")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "GPU training was not started: CUDA is unavailable. In Colab select "
            "Runtime > Change runtime type > Hardware accelerator > GPU, reconnect, "
            "and rerun. Run validate_data.py and pytest separately for CPU-safe checks."
        )

    train_rows = read_jsonl(args.train)
    eval_rows = read_jsonl(args.eval)
    n_train_file, n_eval_file = len(train_rows), len(eval_rows)
    if len(train_rows) < 20:
        raise SystemExit("Refusing to train: provide at least 20 reviewed training examples.")
    if len(eval_rows) < 10:
        raise SystemExit("Refusing to train: provide at least 10 held-out evaluation examples.")
    if args.train.resolve() == args.eval.resolve():
        raise SystemExit("Training and evaluation files must be different.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Build and length-check the data before loading the model, so a problem shows up in seconds.
    train_rows = prompting.stratified_subset(train_rows, args.max_train_rows)
    eval_rows = prompting.stratified_subset(eval_rows, args.eval_limit)
    print(f"Training rows: {len(train_rows)} of {n_train_file}; loss-monitoring rows: {len(eval_rows)} of {n_eval_file}")
    sections = prompting.initialise_knowledge(prompting.knowledge_sources(ROOT), args.knowledge_db)
    print(f"Knowledge index: {sections} sections; prompt version {args.prompt_version}")
    train_ds = build_dataset(train_rows, tokenizer, args.max_length, args.knowledge_db, args.prompt_version)
    eval_ds = build_dataset(eval_rows, tokenizer, args.max_length, args.knowledge_db, args.prompt_version)
    write_manifest(args, len(train_rows), len(eval_rows))

    use_cuda = torch.cuda.is_available()
    quantized = False
    model_kwargs: dict[str, Any] = {}
    if use_cuda:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            # Keep every layer on the GPU; "auto" can silently offload layers to the CPU.
            model_kwargs["device_map"] = {"": 0}
            quantized = True
        except (ImportError, RuntimeError):
            model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if quantized:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=5,
        # eval.jsonl is the held-out test set: its loss is logged each epoch for
        # monitoring only. It must not select the checkpoint, so the final
        # weights after the fixed number of epochs are kept.
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=bool(use_cuda and not torch.cuda.is_bf16_supported()),
        bf16=bool(use_cuda and torch.cuda.is_bf16_supported()),
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=bool(use_cuda),
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        ),
    )
    resume_from = None
    if args.resume:
        checkpoints = sorted(args.output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"Resuming from {resume_from}")
        else:
            print("No checkpoint found; starting a fresh run.")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Saved LoRA adapter and tokenizer to {args.output.resolve()}")
    print("Next: evaluate against the held-out cases and compare with the untouched base model.")

    if args.push_to_hub:
        if not os.environ.get("HF_TOKEN"):
            raise SystemExit("Set HF_TOKEN in the environment before --push-to-hub.")
        if not args.hub_model_id:
            raise SystemExit("Pass --hub-model-id owner/quantai-specialist.")
        model.push_to_hub(args.hub_model_id, token=os.environ["HF_TOKEN"])
        tokenizer.push_to_hub(args.hub_model_id, token=os.environ["HF_TOKEN"])
        print(f"Pushed adapter to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
