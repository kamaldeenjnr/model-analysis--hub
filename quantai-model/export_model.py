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
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
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
    # Record the prompt version so serve.py uses the prompt this model was trained with.
    import json
    manifest = args.adapter / "training_manifest.json"
    version = json.loads(manifest.read_text(encoding="utf-8")).get("prompt_version", "v3") if manifest.exists() else "v3"
    (args.output / "quantai_prompt.json").write_text(json.dumps({"prompt_version": version}), encoding="utf-8")
    if manifest.exists():
        (args.output / "training_manifest.json").write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Prompt version recorded for serving: {version}")
    print(f"Standalone model saved to {args.output.resolve()}")
    print("Review the base-model license and model card before redistribution.")
if __name__ == "__main__":
    main()
