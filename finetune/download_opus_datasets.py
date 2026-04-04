"""
download_opus_datasets.py
========================
Downloads and converts 4 Opus reasoning datasets to a unified JSONL format
compatible with the Jackrong training pipeline.

Datasets:
  1. nohurry/Opus-4.6-Reasoning-3000x-filtered  (~2,330 rows) — flat format, needs conversion
  2. Roman1111111/claude-opus-4.6-10000x        (~9,633 rows) — messages format
  3. TeichAI/claude-4.5-opus-high-reasoning-250x (~250 rows)  — messages format
  4. Jackrong/Qwen3.5-reasoning-700x            (~633 rows)   — conversation format, needs conversion

Output: finetune/outputs/datasets/opus_reasoning.jsonl
  One JSON object per line: {"messages": [{"role":..., "content":...}, ...]}

Usage:
    python finetune/download_opus_datasets.py --output finetune/outputs/datasets/opus_reasoning.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Please reason step by step before answering."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert Opus reasoning datasets to unified JSONL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "datasets"
        / "opus_reasoning.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on total rows (for quick testing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def convert_flat_reasoning(example: dict[str, Any]) -> dict[str, Any] | None:
    problem = example.get("problem", "").strip()
    thinking = example.get("thinking", "").strip()
    solution = example.get("solution", "").strip()
    if not problem or not solution:
        return None
    internal = thinking if thinking else ""
    assistant_content = (
        f"<think>\n{internal}\n</think>\n\n{solution}" if internal else solution
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def convert_conversation(example: dict[str, Any]) -> dict[str, Any] | None:
    user_content = example.get("input", "").strip()
    assistant_content = example.get("output", "").strip()
    if not user_content or not assistant_content:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def convert_messages(example: dict[str, Any]) -> dict[str, Any] | None:
    messages = example.get("messages", [])
    if not messages or len(messages) < 2:
        return None
    filtered = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        filtered.append({"role": role, "content": content})
    if len(filtered) < 3:
        return None
    return {"messages": filtered}


def should_include_roman(example: dict[str, Any]) -> bool:
    category = example.get("metadata", {}).get("category", "")
    allowed = {"simple logic and math", "math"}
    if category and category not in allowed:
        return False
    return True


def download_nohurry(max_rows: int | None, seed: int) -> list[dict]:
    LOGGER.info("Downloading nohurry/Opus-4.6-Reasoning-3000x-filtered ...")
    try:
        from datasets import load_dataset
    except ImportError:
        LOGGER.error("datasets package required: pip install datasets")
        raise SystemExit(1)
    rows = []
    ds = load_dataset("nohurry/Opus-4.6-Reasoning-3000x-filtered", split="train")
    for example in ds:
        row = convert_flat_reasoning(example)
        if row:
            rows.append(row)
    LOGGER.info("  -> %d rows converted", len(rows))
    return rows


def download_roman(max_rows: int | None, seed: int) -> list[dict]:
    LOGGER.info("Downloading Roman1111111/claude-opus-4.6-10000x ...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(1)
    rows = []
    ds = load_dataset(
        "Roman1111111/claude-opus-4.6-10000x",
        split="train",
        streaming=True,
    )
    for example in ds:
        if not should_include_roman(example):
            continue
        row = convert_messages(example)
        if row:
            rows.append(row)
    LOGGER.info("  -> %d rows converted", len(rows))
    return rows


def download_teich(max_rows: int | None, seed: int) -> list[dict]:
    LOGGER.info("Downloading TeichAI/claude-4.5-opus-high-reasoning-250x ...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(1)
    rows = []
    ds = load_dataset("TeichAI/claude-4.5-opus-high-reasoning-250x", split="train")
    for example in ds:
        row = convert_messages(example)
        if row:
            rows.append(row)
    LOGGER.info("  -> %d rows converted", len(rows))
    return rows


def download_jackrong(max_rows: int | None, seed: int) -> list[dict]:
    LOGGER.info("Downloading Jackrong/Qwen3.5-reasoning-700x ...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(1)
    rows = []
    ds = load_dataset("Jackrong/Qwen3.5-reasoning-700x", split="train")
    for example in ds:
        row = convert_conversation(example)
        if row:
            rows.append(row)
    LOGGER.info("  -> %d rows converted", len(rows))
    return rows


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(download_nohurry(args.max_rows, args.seed))
    all_rows.extend(download_roman(args.max_rows, args.seed))
    all_rows.extend(download_teich(args.max_rows, args.seed))
    all_rows.extend(download_jackrong(args.max_rows, args.seed))

    total = len(all_rows)
    LOGGER.info("Total rows before sampling: %d", total)

    if args.max_rows and total > args.max_rows:
        random.seed(args.seed)
        random.shuffle(all_rows)
        all_rows = all_rows[: args.max_rows]
        LOGGER.info("Sampled to %d rows", len(all_rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    LOGGER.info("Wrote %d rows to %s", len(all_rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
