from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .audit_dataset import audit_dataset, audit_summary
except ImportError:
    from audit_dataset import audit_dataset, audit_summary


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
        default="unsloth/Qwen3.5-4B",
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
        default=1,
        help="Per-device batch size (default: 1 for a 16GB GPU).",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
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
        default=0.1,
        help="Evaluation document fraction in range [0, 0.5].",
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
        default=16,
        help="LoRA alpha (default: 16, matching the Qwen3.5 Unsloth recipe).",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_deterministic_seed(seed: int, torch=None) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_resume_from_checkpoint(value: str | None) -> str | bool | None:
    if value is None:
        return None
    if value.casefold() == "true":
        return True
    return value


def _review_status(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    status = metadata.get("review_status")
    return str(status).strip() if status is not None else None


def _metadata_id(row: dict[str, Any], name: str) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(name)
    return str(value).strip() if value is not None and str(value).strip() else None


def _document_id(row: dict[str, Any]) -> str | None:
    return _metadata_id(row, "doc_id")


def _document_family_id(row: dict[str, Any]) -> str | None:
    return _metadata_id(row, "document_family_id")


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


def select_complete_families(dataset, max_samples: int, seed: int):
    family_indices: dict[str, list[int]] = {}
    for index, row in enumerate(dataset):
        family_id = _document_family_id(row)
        if family_id is None:
            raise RuntimeError("Approved rows require metadata.document_family_id for family-aware sampling.")
        family_indices.setdefault(family_id, []).append(index)
    family_ids = sorted(family_indices)
    random.Random(seed).shuffle(family_ids)
    family_ids.sort(key=lambda family_id: len(family_indices[family_id]), reverse=True)
    selected: list[int] = []
    for family_id in family_ids:
        indices = family_indices[family_id]
        if len(selected) + len(indices) <= max_samples:
            selected.extend(indices)
    if not selected:
        smallest_family = min(family_ids, key=lambda family_id: (len(family_indices[family_id]), family_id))
        selected = family_indices[smallest_family]
    return dataset.select(sorted(selected))


def load_and_validate_dataset(
    dataset_path: Path,
    max_samples: int | None,
    seed: int,
    allow_empty_assistant: bool,
    required_review_status: str = "approved",
):
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run ocr_pipeline/process_pdfs.py first."
        )
    audit_report = None
    if required_review_status:
        audit_report = audit_dataset(dataset_path)
        if audit_report["error_count"]:
            raise RuntimeError(
                "Approved dataset audit failed: "
                f"errors={audit_report['error_count']}, warnings={audit_report['warning_count']}."
            )
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    total_rows = len(dataset)
    if total_rows == 0:
        raise RuntimeError("Dataset is empty.")
    if required_review_status:
        invalid_approved_rows = sum(
            _review_status(row) == required_review_status
            and not _is_valid_messages_list(row.get("messages"))
            for row in dataset
        )
        empty_approved_rows = sum(
            _review_status(row) == required_review_status
            and not _has_nonempty_assistant(row.get("messages"))
            for row in dataset
        )
        if invalid_approved_rows or empty_approved_rows:
            raise RuntimeError(
                "Approved rows must all contain valid messages and non-empty assistant targets: "
                f"invalid_messages={invalid_approved_rows}, empty_assistant={empty_approved_rows}."
            )

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

    if required_review_status:
        before = len(dataset)
        dataset = dataset.filter(
            lambda row: _review_status(row) == required_review_status
        )
        dropped = before - len(dataset)
        LOGGER.info(
            "Review gate '%s': kept=%d, dropped=%d",
            required_review_status,
            len(dataset),
            dropped,
        )
        if len(dataset) == 0:
            raise RuntimeError(
                "No rows passed the review gate. Set metadata.review_status to "
                f"'{required_review_status}' after human review."
            )

    missing_doc_ids = sum(_document_id(row) is None for row in dataset)
    missing_family_ids = sum(_document_family_id(row) is None for row in dataset)
    if required_review_status and (missing_doc_ids or missing_family_ids):
        raise RuntimeError(
            "Approved rows require metadata.doc_id and metadata.document_family_id; "
            f"missing_doc_ids={missing_doc_ids}, missing_family_ids={missing_family_ids}."
        )

    eligible_rows = len(dataset)
    sample_limit_applied = max_samples is not None and eligible_rows > max_samples
    if sample_limit_applied:
        if max_samples is None or max_samples < 1:
            raise ValueError("--max-samples must be positive.")
        if required_review_status:
            dataset = select_complete_families(dataset, max_samples, seed)
            LOGGER.info("Dataset limited to %d complete-family rows (--max-samples).", len(dataset))
        else:
            dataset = dataset.shuffle(seed=seed).select(range(max_samples))
            LOGGER.info("Parser dry-run dataset limited to %d rows (--max-samples).", len(dataset))

    return dataset, eligible_rows, sample_limit_applied, audit_summary(audit_report) if audit_report else None


def load_training_stack() -> dict[str, Any]:
    try:
        import torch
        from unsloth import FastLanguageModel
        from transformers import AutoConfig
        from transformers import Trainer
        from transformers import TrainingArguments
        from transformers import DataCollatorForTokenClassification
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install with: pip install -r finetune/requirements.txt"
        ) from exc

    return {
        "torch": torch,
        "AutoConfig": AutoConfig,
        "DataCollatorForTokenClassification": DataCollatorForTokenClassification,
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


def verify_formatted_examples_fit(dataset, tokenizer, max_seq_length: int) -> None:
    for index, row in enumerate(dataset):
        tokenized = tokenizer(row["text"], truncation=False, add_special_tokens=False)
        if len(tokenized["input_ids"]) > max_seq_length:
            raise RuntimeError(
                f"Rendered conversation at dataset index {index} exceeds --max-seq-length={max_seq_length}; refusing truncation."
            )


def tokenize_formatted_examples(dataset, tokenizer, max_seq_length: int):
    verify_formatted_examples_fit(dataset, tokenizer, max_seq_length)

    def tokenize_batch(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokens = tokenizer(
            text=examples["text"],
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )
        tokens["labels"] = [list(ids) for ids in tokens["input_ids"]]
        return tokens

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing formatted text",
    )


def choose_eval_document_ids(
    document_ids: list[str], eval_split: float, seed: int
) -> set[str]:
    if eval_split <= 0:
        return set()
    if eval_split > 0.5:
        raise ValueError("--eval-split must be <= 0.5")

    unique_ids = sorted(set(document_ids))
    if len(unique_ids) < 2:
        raise RuntimeError("Evaluation requires rows from at least two source documents.")

    random.Random(seed).shuffle(unique_ids)
    eval_count = min(len(unique_ids) - 1, max(1, round(len(unique_ids) * eval_split)))
    return set(unique_ids[:eval_count])


def dataset_metadata_ids(dataset, name: str) -> set[str]:
    return {value for row in dataset if (value := _metadata_id(row, name)) is not None}


def dataset_document_ids(dataset) -> set[str]:
    return dataset_metadata_ids(dataset, "doc_id")


def dataset_family_ids(dataset) -> set[str]:
    return dataset_metadata_ids(dataset, "document_family_id")


def choose_eval_family_ids(family_ids: list[str], eval_split: float, seed: int) -> set[str]:
    return choose_eval_document_ids(family_ids, eval_split, seed)


def split_dataset(dataset, eval_split: float, seed: int):
    if eval_split <= 0:
        return dataset, None

    eval_family_ids = choose_eval_family_ids(
        [_document_family_id(row) for row in dataset if _document_family_id(row) is not None],
        eval_split,
        seed,
    )
    train_set = dataset.filter(
        lambda row: _document_family_id(row) not in eval_family_ids,
        desc="Selecting training document families",
    )
    eval_set = dataset.filter(
        lambda row: _document_family_id(row) in eval_family_ids,
        desc="Selecting evaluation document families",
    )
    train_families = dataset_family_ids(train_set)
    eval_families = dataset_family_ids(eval_set)
    if not train_families or not eval_families or train_families & eval_families:
        raise RuntimeError("Document-family split leaked or omitted document families across train and eval.")
    LOGGER.info(
        "Document-family split -> train=%d rows/%d families, eval=%d rows/%d families",
        len(train_set),
        len(train_families),
        len(eval_set),
        len(eval_families),
    )
    return train_set, eval_set


def require_response_only_masking(trainer, tokenizer):
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    if "<|im_start|>assistant" not in chat_template:
        raise RuntimeError(
            "Assistant-only loss requires a ChatML template with the expected assistant marker."
        )

    try:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    except Exception as exc:  # noqa: PERF203
        raise RuntimeError("Could not enable required assistant-only loss masking.") from exc

    LOGGER.info("Assistant-only loss masking ENABLED.")
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


def save_merged_model(model, tokenizer, output_dir: Path) -> Path:
    merged_dir = output_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving merged 16-bit model to %s", merged_dir)
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer=tokenizer,
        save_method="merged_16bit",
        maximum_memory_usage=0.6,
    )
    return merged_dir


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


def verify_model_support(
    model_name: str, hub_token: str | None, auto_config
) -> str | None:
    from huggingface_hub import HfApi

    revision = None
    try:
        revision = HfApi(token=hub_token).model_info(model_name).sha
    except Exception as exc:  # noqa: PERF203
        LOGGER.warning("Could not resolve the base model commit: %s", exc)

    try:
        config = auto_config.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            token=hub_token,
        )
        LOGGER.info(
            "Model config loaded: model_type=%s architectures=%s",
            getattr(config, "model_type", "unknown"),
            getattr(config, "architectures", []),
        )
    except Exception as exc:  # noqa: PERF203
        LOGGER.warning(
            "Could not load model config for %s via AutoConfig (transformers may not "
            "yet support this architecture). Unsloth may still be able to load it directly. Error: %s",
            model_name,
            exc,
        )
    if revision:
        LOGGER.info("Base model commit: %s", revision)
    return revision


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
        "data_seed": kwargs["seed"],
        "full_determinism": True,
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


def current_git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def current_git_dirty(exclude_path: Path | None = None) -> bool | None:
    command = ["git", "status", "--porcelain", "--", "."]
    if exclude_path is not None:
        try:
            repo_root = Path(
                subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
            relative_path = exclude_path.resolve().relative_to(repo_root.resolve())
            command.extend(
                [
                    f":(exclude){relative_path.as_posix()}",
                    f":(exclude){relative_path.as_posix()}/**",
                ]
            )
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    try:
        return bool(
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_training_summary(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    dataset_sha256: str,
    train_rows: int,
    eval_rows: int,
    train_document_ids: set[str],
    eval_document_ids: set[str],
    train_family_ids: set[str],
    eval_family_ids: set[str],
    audit_report: dict[str, Any] | None,
    metrics: dict[str, Any],
    eligible_rows: int,
    sample_limit_applied: bool,
    git_commit: str,
    git_dirty: bool,
    base_model_revision: str | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "base_model": args.model_name,
        "base_model_revision": base_model_revision,
        "dataset_path": str(args.dataset_path),
        "dataset_sha256": dataset_sha256,
        "eligible_rows": eligible_rows,
        "max_samples": args.max_samples,
        "sample_limit_applied": sample_limit_applied,
        "required_review_status": "approved",
        "output_dir": str(output_dir),
        "seed": args.seed,
        "split_strategy": "document_family_id",
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "train_document_count": len(train_document_ids),
        "eval_document_count": len(eval_document_ids),
        "train_document_ids": sorted(train_document_ids),
        "eval_document_ids": sorted(eval_document_ids),
        "train_family_count": len(train_family_ids),
        "eval_family_count": len(eval_family_ids),
        "train_family_ids": sorted(train_family_ids),
        "eval_family_ids": sorted(eval_family_ids),
        "audit": audit_report,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "precision": "bfloat16",
        "assistant_only_loss": True,
        "save_merged_model": args.save_merged_model,
        "skip_gguf_export": args.skip_gguf_export,
        "resume_from_checkpoint": normalize_resume_from_checkpoint(
            args.resume_from_checkpoint
        ),
        "packages": package_versions(
            ["torch", "unsloth", "transformers", "datasets", "accelerate", "peft"]
        ),
        "metrics": metrics,
    }
    serialized = json.dumps(payload, indent=2, default=str)
    (output_dir / "run_manifest.json").write_text(serialized, encoding="utf-8")
    (output_dir / "training_summary.json").write_text(serialized, encoding="utf-8")


def run_training(args: argparse.Namespace) -> int:
    check_runtime_environment()
    set_deterministic_seed(args.seed)
    if args.allow_empty_assistant and not args.dry_run:
        raise RuntimeError("--allow-empty-assistant is permitted only with --dry-run.")
    if not args.dry_run and args.eval_split <= 0:
        raise RuntimeError("Production training requires --eval-split greater than zero.")
    if not args.dry_run and not args.skip_gguf_export:
        raise RuntimeError(
            "Production training requires --skip-gguf-export; run export_gguf.py from the saved merged model."
        )
    if not args.dry_run and not args.save_merged_model:
        raise RuntimeError("Production training requires --save-merged-model for release provenance.")
    training_git_commit = current_git_commit() if not args.dry_run else None
    training_git_dirty = (
        current_git_dirty(args.output_dir) if not args.dry_run else None
    )
    if not args.dry_run and (
        not training_git_commit or training_git_dirty is not False
    ):
        raise RuntimeError("Production training requires a clean committed worktree.")
    required_review_status = "" if args.allow_empty_assistant else "approved"
    dataset_sha256 = sha256_file(args.dataset_path)

    dataset, eligible_rows, sample_limit_applied, audit_report = load_and_validate_dataset(
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        seed=args.seed,
        allow_empty_assistant=args.allow_empty_assistant,
        required_review_status=required_review_status,
    )
    if sha256_file(args.dataset_path) != dataset_sha256:
        raise RuntimeError("Dataset changed while loading; refusing to train.")
    LOGGER.info("Validated dataset at %s", args.dataset_path)
    LOGGER.info("Rows ready for training: %d", len(dataset))

    if args.dry_run and args.allow_empty_assistant:
        train_dataset, eval_dataset = dataset, None
    else:
        train_dataset, eval_dataset = split_dataset(dataset, args.eval_split, args.seed)
    train_document_ids = dataset_document_ids(train_dataset)
    eval_document_ids = dataset_document_ids(eval_dataset) if eval_dataset is not None else set()
    train_family_ids = dataset_family_ids(train_dataset)
    eval_family_ids = dataset_family_ids(eval_dataset) if eval_dataset is not None else set()

    if args.dry_run:
        LOGGER.info("Dry run complete. Skipping model initialization and training.")
        return 0
    assert training_git_commit is not None and training_git_dirty is False

    stack = load_training_stack()
    torch = stack["torch"]
    AutoConfig = stack["AutoConfig"]
    DataCollatorForTokenClassification = stack["DataCollatorForTokenClassification"]
    Trainer = stack["Trainer"]
    TrainingArguments = stack["TrainingArguments"]
    FastLanguageModel = stack["FastLanguageModel"]
    hub_token = resolve_hub_token(args.hub_token_env)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training script.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This Qwen3.5 workflow requires a bf16-capable GPU. Do not silently fall back to fp16."
        )

    set_deterministic_seed(args.seed, torch)
    log_gpu_state(torch)
    base_model_revision = verify_model_support(args.model_name, hub_token, AutoConfig)
    if not base_model_revision:
        raise RuntimeError("Could not resolve an immutable base model commit.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
        token=hub_token,
        revision=base_model_revision,
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
        max_seq_length=args.max_seq_length,
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

    train_dataset = format_chat_examples(
        train_dataset,
        save_tokenizer,
        allow_thinking_template=args.allow_thinking_template,
    )
    train_dataset = tokenize_formatted_examples(
        train_dataset,
        training_tokenizer,
        max_seq_length=args.max_seq_length,
    )
    if eval_dataset is not None:
        eval_dataset = format_chat_examples(
            eval_dataset,
            save_tokenizer,
            allow_thinking_template=args.allow_thinking_template,
        )
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

    bf16 = True
    fp16 = False
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

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorForTokenClassification(
            training_tokenizer,
            label_pad_token_id=-100,
        ),
        "args": training_args,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = training_tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = training_tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer = require_response_only_masking(trainer, training_tokenizer)

    metrics: dict[str, Any] = {}
    if eval_dataset is not None:
        baseline_metrics = trainer.evaluate(metric_key_prefix="baseline")
        metrics.update(baseline_metrics)
        LOGGER.info("Baseline metrics: %s", baseline_metrics)

    LOGGER.info("Starting training...")
    result = trainer.train(
        resume_from_checkpoint=normalize_resume_from_checkpoint(
            args.resume_from_checkpoint
        )
    )
    metrics.update(result.metrics)
    if eval_dataset is not None:
        final_metrics = trainer.evaluate(metric_key_prefix="final")
        metrics.update(final_metrics)
        LOGGER.info("Final evaluation metrics: %s", final_metrics)
    LOGGER.info("Training complete. Metrics: %s", metrics)
    if sha256_file(args.dataset_path) != dataset_sha256:
        raise RuntimeError("Dataset changed during training; refusing to write a run manifest.")
    if (
        current_git_commit() != training_git_commit
        or current_git_dirty(args.output_dir) is not False
    ):
        raise RuntimeError("Repository changed during training; refusing to write a run manifest.")
    write_training_summary(
        output_dir,
        args=args,
        dataset_sha256=dataset_sha256,
        train_rows=len(train_dataset),
        eval_rows=len(eval_dataset) if eval_dataset is not None else 0,
        train_document_ids=train_document_ids,
        eval_document_ids=eval_document_ids,
        train_family_ids=train_family_ids,
        eval_family_ids=eval_family_ids,
        audit_report=audit_report,
        metrics=metrics,
        eligible_rows=eligible_rows,
        sample_limit_applied=sample_limit_applied,
        git_commit=training_git_commit,
        git_dirty=training_git_dirty,
        base_model_revision=base_model_revision,
    )

    LOGGER.info("Saving LoRA adapter to %s", adapter_dir)
    for peft_config in getattr(model, "peft_config", {}).values():
        peft_config.base_model_name_or_path = args.model_name
        peft_config.revision = base_model_revision
    model.save_pretrained(str(adapter_dir))
    save_tokenizer.save_pretrained(str(adapter_dir))
    if args.save_merged_model:
        save_merged_model(model, save_tokenizer, output_dir)

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
