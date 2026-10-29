#!/usr/bin/env python3
"""Merge a trained PEFT adapter into a standalone local model directory."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True, help="Trained PEFT adapter directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("outputs/quantai-merged-model"))
    args = parser.parse_args()
    if not (args.adapter / "adapter_config.json").exists():
        raise SystemExit(f"No adapter_config.json found under {args.adapter}")
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), use_fast=True)
    tokenizer.save_pretrained(args.output)
    print(f"Standalone model saved to {args.output.resolve()}")
    print("Review the base-model license and model card before redistribution.")
if __name__ == "__main__":
    main()
