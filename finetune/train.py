from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen with Unsloth + TRL and export GGUF."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=repo_root / "ocr_pipeline" / "finetune_template.jsonl",
        help="Path to JSONL dataset in chat messages format.",
    )
    parser.add_argument(
        "--model-name",
        default="Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for checkpoints, adapters, and GGUF files.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024, safer for 16GB GPUs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4).",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2).",
    )
    parser.add_argument(
        "--max-memory-ratio",
        type=float,
        default=0.85,
        help="Max GPU memory usage ratio before reducing batch size (default: 0.85).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5).",
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=1.0,
        help="Number of train epochs (default: 1.0).",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.05,
        help="Evaluation split fraction in range [0, 0.5].",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of dataset rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed (default: 3407).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32, recommended 2x lora-r for style fine-tuning).",
    )
    parser.add_argument(
        "--log-steps",
        type=int,
        default=10,
        help="Log every N optimizer steps (default: 10).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps (default: 200).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Warmup steps (default: 20).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay (default: 0.05).",
    )
    parser.add_argument(
        "--optim",
        default="adamw_8bit",
        choices=["adamw_8bit", "adamw_torch"],
        help="Optimizer (default: adamw_8bit).",
    )
    parser.add_argument(
        "--report-to",
        default="none",
        choices=["none", "tensorboard"],
        help="Metrics backend (default: none).",
    )
    parser.add_argument(
        "--gguf-name",
        default="qwen3_5_4b_finance_q4_k_m",
        help="Output GGUF base name (without extension).",
    )
    parser.add_argument(
        "--save-merged-model",
        action="store_true",
        help="Also save a merged 16-bit Hugging Face model after training.",
    )
    parser.add_argument(
        "--push-adapter-repo-id",
        default=None,
        help="Optional Hugging Face repo ID for uploading the LoRA adapter.",
    )
    parser.add_argument(
        "--push-merged-repo-id",
        default=None,
        help="Optional Hugging Face repo ID for uploading the merged model.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create pushed Hugging Face repos as private.",
    )
    parser.add_argument(
        "--hub-token-env",
        default="HF_TOKEN",
        help="Environment variable name that stores the Hugging Face token.",
    )
    parser.add_argument(
        "--allow-thinking-template",
        action="store_true",
        help="Do not force enable_thinking=False when applying the chat template.",
    )
    parser.add_argument(
        "--disable-response-only-masking",
        action="store_true",
        help="Skip Unsloth response-only masking and train on the full formatted sequence.",
    )
    parser.add_argument(
        "--skip-gguf-export",
        action="store_true",
        help="Skip GGUF export after training.",
    )
    parser.add_argument(
        "--allow-empty-assistant",
        action="store_true",
        help="Allow examples where assistant message content is empty.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and format dataset only; do not initialize model/training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="True",
        default=None,
        help="Resume training from checkpoint. Pass a path to resume from that checkpoint, "
        "or 'True' to auto-resume from the latest checkpoint in output-dir.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def check_runtime_environment() -> None:
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        raise RuntimeError("Python 3.10+ is required for this training pipeline.")

    if version.minor >= 14:
        LOGGER.warning(
            "Detected Python %s.%s. Unsloth/bitsandbytes are usually validated on "
            "Python 3.10-3.12. Prefer Python 3.11 for stable installs.",
            version.major,
            version.minor,
        )


def resolve_hub_token(env_name: str) -> str | None:
    token = os.getenv(env_name)
    if token and token.strip():
        return token.strip()
    return None


def _has_nonempty_assistant(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True

    return False


def _is_valid_messages_list(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False

    roles: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            return False
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return False
        roles.add(role)

    return "user" in roles and "assistant" in roles


def load_and_validate_dataset(
    dataset_path: Path,
    max_samples: int | None,
    seed: int,
    allow_empty_assistant: bool,
):
    from datasets import load_dataset

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run ocr_pipeline/process_pdfs.py first."
        )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    total_rows = len(dataset)
    if total_rows == 0:
        raise RuntimeError("Dataset is empty.")

    dataset = dataset.filter(lambda row: _is_valid_messages_list(row.get("messages")))
    valid_rows = len(dataset)
    LOGGER.info("Dataset rows: total=%d, valid_messages=%d", total_rows, valid_rows)

    if valid_rows == 0:
        raise RuntimeError("No valid rows found with messages=[{role,content}, ...].")

    if not allow_empty_assistant:
        before = len(dataset)
        dataset = dataset.filter(
            lambda row: _has_nonempty_assistant(row.get("messages"))
        )
        dropped = before - len(dataset)
        if dropped > 0:
            LOGGER.info("Filtered out %d rows with empty assistant responses.", dropped)
        if len(dataset) == 0:
            raise RuntimeError(
                "All rows have empty assistant messages. Fill assistant completions "
                "in finetune_template.jsonl, or run with --allow-empty-assistant."
            )

    if max_samples is not None:
        max_samples = max(1, max_samples)
        if len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=seed).select(range(max_samples))
            LOGGER.info("Dataset limited to %d rows (--max-samples).", len(dataset))

    return dataset


def load_training_stack() -> dict[str, Any]:
    try:
        import torch
        from unsloth import FastLanguageModel
        from transformers import AutoConfig
        from transformers import DataCollatorForLanguageModeling
        from transformers import Trainer
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install with: pip install -r finetune/requirements.txt"
        ) from exc

    return {
        "torch": torch,
        "AutoConfig": AutoConfig,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "FastLanguageModel": FastLanguageModel,
    }


def resolve_optimizer(preferred: str) -> str:
    if preferred == "adamw_8bit" and importlib.util.find_spec("bitsandbytes") is None:
        LOGGER.warning(
            "bitsandbytes is unavailable. Falling back to adamw_torch optimizer."
        )
        return "adamw_torch"
    return preferred


def apply_chat_template(
    tokenizer, messages: list[dict[str, str]], allow_thinking: bool
) -> str:
    common_kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
    }

    if not allow_thinking:
        for extra_kwargs in (
            {"enable_thinking": False},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ):
            try:
                return tokenizer.apply_chat_template(
                    messages, **common_kwargs, **extra_kwargs
                )
            except TypeError:
                continue

    return tokenizer.apply_chat_template(messages, **common_kwargs)


def format_chat_examples(dataset, tokenizer, allow_thinking_template: bool):
    def apply_template(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        formatted = []
        for messages in examples["messages"]:
            text = apply_chat_template(
                tokenizer,
                messages,
                allow_thinking=allow_thinking_template,
            )
            formatted.append(text)
        return {"text": formatted}

    return dataset.map(
        apply_template,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Applying chat template",
    )


def tokenize_formatted_examples(dataset, tokenizer, max_seq_length: int):
    def tokenize_batch(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokens = tokenizer(
            text=examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        tokens["labels"] = [ids[:] for ids in tokens["input_ids"]]
        return tokens

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing formatted text",
    )


def split_dataset(dataset, eval_split: float, seed: int):
    if eval_split <= 0:
        return dataset, None
    if eval_split > 0.5:
        raise ValueError("--eval-split must be <= 0.5")
    if len(dataset) < 20:
        LOGGER.warning("Dataset is very small; disabling eval split.")
        return dataset, None

    split = dataset.train_test_split(test_size=eval_split, seed=seed)
    train_set = split["train"]
    eval_set = split["test"]
    LOGGER.info("Split dataset -> train=%d, eval=%d", len(train_set), len(eval_set))
    return train_set, eval_set


def maybe_mask_non_assistant_tokens(trainer, tokenizer, disable: bool):
    if disable:
        LOGGER.info("Skipping response-only masking (--disable-response-only-masking).")
        return trainer

    chat_template = getattr(tokenizer, "chat_template", "") or ""
    if "<|im_start|>assistant" not in chat_template:
        LOGGER.warning(
            "Chat template does not expose the expected assistant marker; skipping response-only masking."
        )
        return trainer

    try:
        from unsloth.chat_templates import train_on_responses_only

        return train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    except Exception as exc:  # noqa: PERF203
        LOGGER.warning(
            "Could not enable response-only masking; continuing without it: %s", exc
        )
        return trainer


def export_gguf(model, tokenizer, output_dir: Path, gguf_name: str) -> None:
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_target = gguf_dir / gguf_name

    LOGGER.info("Exporting GGUF (q4_k_m) to %s", gguf_target)
    model.save_pretrained_gguf(
        str(gguf_target),
        tokenizer,
        quantization_method="q4_k_m",
    )


def save_merged_model(
    model,
    tokenizer,
    output_dir: Path,
    *,
    hub_token: str | None,
    push_repo_id: str | None,
    hub_private: bool,
) -> Path:
    merged_dir = output_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving merged 16-bit model to %s", merged_dir)
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer=tokenizer,
        save_method="merged_16bit",
        maximum_memory_usage=0.6,
    )

    if push_repo_id:
        if not hub_token:
            raise RuntimeError(
                "Cannot push merged model without a Hugging Face token. "
                "Set the token in the configured hub token environment variable."
            )
        LOGGER.info("Uploading merged model to %s", push_repo_id)
        model.push_to_hub_merged(
            push_repo_id,
            tokenizer=tokenizer,
            save_method="merged_16bit",
            private=hub_private,
            token=hub_token,
        )

    return merged_dir


def maybe_push_adapter(
    model,
    tokenizer,
    repo_id: str | None,
    *,
    hub_private: bool,
    hub_token: str | None,
) -> None:
    if not repo_id:
        return
    if not hub_token:
        raise RuntimeError(
            "Cannot push adapter without a Hugging Face token. "
            "Set the token in the configured hub token environment variable."
        )

    LOGGER.info("Uploading LoRA adapter to %s", repo_id)
    model.push_to_hub_merged(
        repo_id,
        tokenizer=tokenizer,
        save_method="lora",
        private=hub_private,
        token=hub_token,
    )


def log_gpu_state(torch) -> None:
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_memory, total_visible = torch.cuda.mem_get_info()
    LOGGER.info(
        "Using GPU: %s | free=%.2f GiB | visible_total=%.2f GiB | physical_total=%.2f GiB",
        device_name,
        free_memory / 1024**3,
        total_visible / 1024**3,
        total_memory,
    )


def verify_model_support(model_name: str, hub_token: str | None, auto_config) -> None:
    try:
        config = auto_config.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hub_token,
        )
    except Exception as exc:  # noqa: PERF203
        raise RuntimeError(
            f"Could not load model config for {model_name}. Confirm transformers supports "
            "this model family and that Hugging Face access is available."
        ) from exc

    LOGGER.info(
        "Model config loaded: model_type=%s architectures=%s",
        getattr(config, "model_type", "unknown"),
        getattr(config, "architectures", []),
    )


def build_training_arguments(
    training_arguments_cls, *, has_eval: bool, report_to, **kwargs
):
    signature = inspect.signature(training_arguments_cls.__init__)
    supported = set(signature.parameters)

    training_kwargs = {
        "output_dir": kwargs["output_dir"],
        "per_device_train_batch_size": kwargs["batch_size"],
        "gradient_accumulation_steps": kwargs["gradient_accumulation"],
        "warmup_steps": kwargs["warmup_steps"],
        "num_train_epochs": kwargs["num_epochs"],
        "learning_rate": kwargs["learning_rate"],
        "fp16": kwargs["fp16"],
        "bf16": kwargs["bf16"],
        "logging_steps": kwargs["log_steps"],
        "optim": kwargs["optimizer"],
        "weight_decay": kwargs["weight_decay"],
        "lr_scheduler_type": "cosine",
        "seed": kwargs["seed"],
        "save_steps": kwargs["save_steps"],
        "save_total_limit": 3,
        "report_to": report_to,
        "remove_unused_columns": False,
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False if has_eval else None,
    }

    eval_key = (
        "evaluation_strategy" if "evaluation_strategy" in supported else "eval_strategy"
    )
    training_kwargs[eval_key] = "steps" if has_eval else "no"

    if has_eval:
        training_kwargs["eval_steps"] = max(kwargs["log_steps"] * 2, 20)

    if "save_strategy" in supported:
        training_kwargs["save_strategy"] = "steps"

    return training_arguments_cls(
        **{
            key: value
            for key, value in training_kwargs.items()
            if key in supported and value is not None
        }
    )


def write_training_summary(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    train_rows: int,
    eval_rows: int,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "base_model": args.model_name,
        "dataset_path": str(args.dataset_path),
        "output_dir": str(output_dir),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "save_merged_model": args.save_merged_model,
        "skip_gguf_export": args.skip_gguf_export,
        "metrics": metrics,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run_training(args: argparse.Namespace) -> int:
    check_runtime_environment()

    dataset = load_and_validate_dataset(
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        seed=args.seed,
        allow_empty_assistant=args.allow_empty_assistant,
    )

    LOGGER.info("Validated dataset at %s", args.dataset_path)
    LOGGER.info("Rows ready for training: %d", len(dataset))

    if args.dry_run:
        LOGGER.info("Dry run complete. Skipping model initialization and training.")
        return 0

    stack = load_training_stack()
    torch = stack["torch"]
    AutoConfig = stack["AutoConfig"]
    DataCollatorForLanguageModeling = stack["DataCollatorForLanguageModeling"]
    Trainer = stack["Trainer"]
    TrainingArguments = stack["TrainingArguments"]
    FastLanguageModel = stack["FastLanguageModel"]
    hub_token = resolve_hub_token(args.hub_token_env)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training script.")

    log_gpu_state(torch)
    verify_model_support(args.model_name, hub_token, AutoConfig)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=hub_token,
    )
    save_tokenizer = tokenizer

    free_mem, total_mem = torch.cuda.mem_get_info()
    max_allowed = total_mem * args.max_memory_ratio
    if free_mem < max_allowed * 0.3:
        LOGGER.warning(
            "GPU memory low! free=%.1f GiB, total=%.1f GiB. "
            "Consider reducing --batch-size.",
            free_mem / 1024**3,
            total_mem / 1024**3,
        )
    training_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if getattr(training_tokenizer, "pad_token", None) is None:
        training_tokenizer.pad_token = training_tokenizer.eos_token
    LOGGER.info(
        "Loaded tokenizer classes: save=%s train=%s",
        type(save_tokenizer).__name__,
        type(training_tokenizer).__name__,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    free_mem, total_mem = torch.cuda.mem_get_info()
    max_allowed = total_mem * args.max_memory_ratio
    LOGGER.info(
        "GPU memory after model load: free=%.1f GiB / %.1f GiB (%.0f%% free, threshold=%.0f%%)",
        free_mem / 1024**3,
        total_mem / 1024**3,
        (free_mem / total_mem) * 100,
        (1 - args.max_memory_ratio) * 100,
    )
    if free_mem < total_mem * (1 - args.max_memory_ratio):
        LOGGER.warning(
            "GPU memory usage exceeded --max-memory-ratio=%.0f%%! "
            "Reduce --batch-size or increase --gradient-accumulation.",
            args.max_memory_ratio * 100,
        )

    dataset = format_chat_examples(
        dataset,
        save_tokenizer,
        allow_thinking_template=args.allow_thinking_template,
    )
    train_dataset, eval_dataset = split_dataset(dataset, args.eval_split, args.seed)
    train_dataset = tokenize_formatted_examples(
        train_dataset,
        training_tokenizer,
        max_seq_length=args.max_seq_length,
    )
    if eval_dataset is not None:
        eval_dataset = tokenize_formatted_examples(
            eval_dataset,
            training_tokenizer,
            max_seq_length=args.max_seq_length,
        )

    output_dir = args.output_dir
    checkpoints_dir = output_dir / "checkpoints"
    adapter_dir = output_dir / "adapter"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    bf16 = bool(torch.cuda.is_bf16_supported())
    fp16 = not bf16
    optimizer = resolve_optimizer(args.optim)

    report_to = [] if args.report_to == "none" else [args.report_to]
    has_eval = eval_dataset is not None

    training_args = build_training_arguments(
        TrainingArguments,
        has_eval=has_eval,
        report_to=report_to,
        output_dir=str(checkpoints_dir),
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=fp16,
        bf16=bf16,
        log_steps=args.log_steps,
        optimizer=optimizer,
        weight_decay=args.weight_decay,
        seed=args.seed,
        save_steps=args.save_steps,
    )

    if not args.disable_response_only_masking:
        LOGGER.warning(
            "The current Qwen 3.5 training path uses tokenized full-sequence training. "
            "Response-only masking is not applied in this backend."
        )

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorForLanguageModeling(
            tokenizer=training_tokenizer,
            mlm=False,
        ),
        "args": training_args,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = training_tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = training_tokenizer

    trainer = Trainer(**trainer_kwargs)

    LOGGER.info("Starting training...")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    LOGGER.info("Training complete. Metrics: %s", result.metrics)
    write_training_summary(
        output_dir,
        args=args,
        train_rows=len(train_dataset),
        eval_rows=len(eval_dataset) if eval_dataset is not None else 0,
        metrics=result.metrics,
    )

    LOGGER.info("Saving LoRA adapter to %s", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    save_tokenizer.save_pretrained(str(adapter_dir))
    maybe_push_adapter(
        model,
        save_tokenizer,
        args.push_adapter_repo_id,
        hub_private=args.hub_private,
        hub_token=hub_token,
    )

    if args.save_merged_model or args.push_merged_repo_id:
        save_merged_model(
            model,
            save_tokenizer,
            output_dir,
            hub_token=hub_token,
            push_repo_id=args.push_merged_repo_id,
            hub_private=args.hub_private,
        )

    if not args.skip_gguf_export:
        export_gguf(model, save_tokenizer, output_dir, args.gguf_name)
    else:
        LOGGER.info("Skipping GGUF export (--skip-gguf-export).")

    LOGGER.info("Done.")
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        return run_training(args)
    except Exception:  # noqa: PERF203
        LOGGER.exception("Training failed:")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
