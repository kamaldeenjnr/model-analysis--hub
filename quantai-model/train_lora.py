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

DEFAULT_BASE = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "You are QuantAI, a research-methods and statistics assistant. "
    "Return exactly one JSON object with keys reply and action. "
    "Use only supported action types: describe, compare, regression, survival, "
    "forecast, check, fix, report, open. If the request is a question rather "
    "than a request to run a tool, use action null. Never invent result values "
    "or variable names. The existing deterministic engine computes all statistics."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
                raise ValueError(f"{path}:{line_no}: expected string prompt and completion")
            try:
                parsed = json.loads(row["completion"])
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: completion must be valid JSON: {exc}") from exc
            if set(parsed) != {"reply", "action"} or not isinstance(parsed["reply"], str):
                raise ValueError(f"{path}:{line_no}: completion must contain reply and action")
            rows.append({"prompt": row["prompt"], "completion": row["completion"]})
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def build_dataset(rows: list[dict[str, str]], tokenizer, max_length: int) -> Dataset:
    processed = []
    for row in rows:
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        completion_text = row["completion"] + (tokenizer.eos_token or "")
        completion_ids = tokenizer(
            completion_text, add_special_tokens=False, truncation=True,
            max_length=max_length // 2
        )["input_ids"]
        prompt_budget = max(1, max_length - len(completion_ids))
        old_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, truncation=True,
            max_length=prompt_budget
        )["input_ids"]
        tokenizer.truncation_side = old_side
        input_ids = (prompt_ids + completion_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
        processed.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        })
    return Dataset.from_list(processed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=Path("outputs/quantai-lora"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default="")
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    eval_rows = read_jsonl(args.eval)
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
            model_kwargs["device_map"] = "auto"
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

    train_ds = build_dataset(train_rows, tokenizer, args.max_length)
    eval_ds = build_dataset(eval_rows, tokenizer, args.max_length)
    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    trainer.train()
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
