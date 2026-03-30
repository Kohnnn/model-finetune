from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="CPU-friendly LoRA fine-tuning path for a small Qwen model."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "datasets"
        / "analyst_sft_seed_256.jsonl",
        help="Path to seeded SFT dataset in chat messages format.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name for CPU LoRA fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "cpu_qwen2_5_0_5b_seed",
        help="Directory for checkpoints, adapters, and merged model artifacts.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=384,
        help="Maximum token length for training examples.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Cap the number of rows used for the CPU run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed.",
    )
    parser.add_argument(
        "--log-steps",
        type=int,
        default=5,
        help="Log every N steps.",
    )
    parser.add_argument(
        "--save-merged-model",
        action="store_true",
        help="Merge adapter weights into the base model after training.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_dataset_rows(
    dataset_path: Path, max_samples: int | None
) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            payload = json.loads(line)
            messages = payload.get("messages")
            if not isinstance(messages, list):
                continue
            if not any(
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and str(message.get("content", "")).strip()
                for message in messages
            ):
                continue
            rows.append(payload)

    if not rows:
        raise RuntimeError(
            "No valid training rows with non-empty assistant content found."
        )

    if max_samples is not None and len(rows) > max_samples:
        rows = rows[:max_samples]

    LOGGER.info("Rows ready for CPU training: %d", len(rows))
    return rows


def format_chat_rows(rows: list[dict[str, Any]], tokenizer) -> list[str]:
    formatted: list[str] = []
    for row in rows:
        formatted.append(
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        )
    return formatted


def write_model_card(
    output_dir: Path,
    *,
    base_model: str,
    dataset_path: Path,
    sample_count: int,
    merged: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    title = output_dir.name.replace("_", "-")
    card = f"""---
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
tags:
  - finance
  - lora
  - research
  - synthetic-dataset
---

# {title}

This artifact is a CPU-trained finance-style LoRA run derived from local research-report chunks.

## Training Summary

- Base model: `{base_model}`
- Dataset: `{dataset_path.name}`
- Sample count: `{sample_count}`
- Artifact type: `{"merged model" if merged else "LoRA adapter"}`

## Important Limitations

- This run uses a synthetic seed dataset generated from existing report text.
- It is suitable as a bootstrap artifact, not as a final production-quality analyst model.
- A stronger GPU run with manually labeled completions is still recommended.

## Usage

If this folder contains a merged model, load it directly with `transformers`.
If this folder contains an adapter, load the base model first and then apply the adapter with `peft`.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing CPU training dependencies. Install with: pip install -r finetune/requirements.txt"
        ) from exc

    if sys.version_info[:2] < (3, 10):
        raise RuntimeError("Python 3.10+ is required.")

    rows = load_dataset_rows(args.dataset_path, max_samples=args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    formatted_rows = format_chat_rows(rows, tokenizer)
    dataset = Dataset.from_dict({"text": formatted_rows})

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_batch, batched=True, remove_columns=["text"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.train()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = args.output_dir
    checkpoints_dir = output_dir / "checkpoints"
    adapter_dir = output_dir / "adapter"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=args.log_steps,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=0,
        seed=args.seed,
        use_cpu=True,
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    LOGGER.info("Starting CPU LoRA training...")
    result = trainer.train()
    LOGGER.info("Training complete. Metrics: %s", result.metrics)

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_model_card(
        adapter_dir,
        base_model=args.model_name,
        dataset_path=args.dataset_path,
        sample_count=len(rows),
        merged=False,
    )

    summary = {
        "base_model": args.model_name,
        "dataset_path": str(args.dataset_path),
        "sample_count": len(rows),
        "max_seq_length": args.max_seq_length,
        "num_epochs": args.num_epochs,
        "metrics": result.metrics,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if args.save_merged_model:
        LOGGER.info("Saving merged model artifact...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        merged_model = PeftModel.from_pretrained(
            base_model, str(adapter_dir)
        ).merge_and_unload()
        merged_dir = output_dir / "merged_model"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        write_model_card(
            merged_dir,
            base_model=args.model_name,
            dataset_path=args.dataset_path,
            sample_count=len(rows),
            merged=True,
        )

    LOGGER.info("Artifacts written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
