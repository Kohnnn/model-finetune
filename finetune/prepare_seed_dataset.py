from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path


LOGGER = logging.getLogger(__name__)

DEFAULT_TASK_PROMPT = (
    "Task: Deliver expert equity research commentary and strategic evaluation based "
    "on the context above. Prioritize deep analysis over factual reporting."
)

DISCLAIMER_MARKERS = [
    "analyst certification",
    "disclaimer",
    "all rights reserved",
    "for investment advice",
    "important disclosures",
    "major us institutional investors",
    "u.k. and european economic area",
    "this report is provided",
    "contacts",
    "copyright",
    "xac nhan cua chuyen vien phan tich",
    "bao cao nay duoc viet va phat hanh",
    "phuong phap dinh gia",
    "phong giao dich chung khoan",
    "phong nghien cuu va phan tich",
    "cong ty co phan chung khoan ban viet",
]

ANALYTICAL_MARKERS = [
    "buy",
    "outperform",
    "underperform",
    "market perform",
    "target price",
    "valuation",
    "upside",
    "downside",
    "margin",
    "earnings",
    "profit",
    "credit",
    "nim",
    "roe",
    "npl",
    "forecast",
    "khuyen nghi",
    "gia muc tieu",
    "dinh gia",
    "bien loi nhuan",
    "tang truong",
    "loi nhuan",
    "du bao",
    "rui ro",
    "nim",
    "roe",
    "no xau",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create a hybrid-review SFT draft dataset with synthetic analyst completions."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=repo_root / "ocr_pipeline" / "finetune_template.jsonl",
        help="Input template dataset produced by the OCR pipeline.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "datasets"
        / "qwen35_hybrid_review_512.jsonl",
        help="Output JSONL path for the hybrid-review draft dataset.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=512,
        help="Maximum number of draft rows to write.",
    )
    parser.add_argument(
        "--min-assistant-chars",
        type=int,
        default=180,
        help="Minimum assistant completion length.",
    )
    parser.add_argument(
        "--max-context-words",
        type=int,
        default=450,
        help="Trim the context passed to the user message to this many words.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for deterministic sampling.",
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


def normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text)
    return compact.strip()


def extract_context(user_content: str) -> str:
    context = user_content
    if "Context:" in context:
        context = context.split("Context:", 1)[1]
    if "\n\nTask:" in context:
        context = context.split("\n\nTask:", 1)[0]
    elif "\nTask:" in context:
        context = context.split("\nTask:", 1)[0]
    return normalize_text(context)


def extract_task_prompt(user_content: str) -> str:
    if "Task:" not in user_content:
        return DEFAULT_TASK_PROMPT
    return normalize_text(f"Task:{user_content.split('Task:', 1)[1]}")


def truncate_context_words(context: str, max_context_words: int) -> str:
    words = context.split()
    if max_context_words <= 0 or len(words) <= max_context_words:
        return context
    return " ".join(words[:max_context_words]).strip()


def is_disclaimer_context(context: str) -> bool:
    lowered = context.lower()
    hits = sum(marker in lowered for marker in DISCLAIMER_MARKERS)
    return hits >= 2 or lowered.startswith("analyst certification")


def split_sentences(context: str) -> list[str]:
    prepared = re.sub(r"\s+-\s+", ". ", context)
    prepared = prepared.replace("\u2022", ". ")
    raw_parts = re.split(r"(?<=[.!?])\s+|\s{2,}|\n+|;\s+", prepared)
    sentences = [normalize_text(part) for part in raw_parts if normalize_text(part)]
    if len(sentences) <= 1:
        sentences = [
            normalize_text(part)
            for part in re.split(r",\s+(?=[A-Z0-9À-ỹ])", prepared)
            if normalize_text(part)
        ]
    return sentences


def sentence_score(sentence: str) -> int:
    lowered = sentence.lower()
    score = 0
    if any(marker in lowered for marker in ANALYTICAL_MARKERS):
        score += 4
    if re.search(r"\b\d+(?:[.,]\d+)?%?\b", sentence):
        score += 2
    if 60 <= len(sentence) <= 320:
        score += 2
    if len(sentence) > 360:
        score -= 2
    if any(marker in lowered for marker in DISCLAIMER_MARKERS):
        score -= 6
    if lowered.count("source:") or lowered.count("nguon:"):
        score -= 2
    return score


def build_assistant_completion(context: str, min_chars: int) -> str | None:
    sentences = split_sentences(context)
    if not sentences:
        return None

    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        score = sentence_score(sentence)
        if score > 0:
            scored.append((score, index, sentence))

    if not scored:
        fallback = [sentence for sentence in sentences if 60 <= len(sentence) <= 320]
        selected = fallback[:3]
    else:
        top_sentences = sorted(scored, key=lambda item: (-item[0], item[1]))[:5]
        selected = [
            sentence
            for _, _, sentence in sorted(top_sentences, key=lambda item: item[1])
        ]

    assistant = normalize_text(" ".join(selected[:3]))
    if len(assistant) < min_chars:
        for sentence in sentences:
            if sentence in selected:
                continue
            candidate = normalize_text(f"{assistant} {sentence}".strip())
            if len(candidate) > len(assistant):
                assistant = candidate
            if len(assistant) >= min_chars:
                break

    if len(assistant) < min_chars:
        return None
    return assistant


def iter_seed_rows(input_path: Path, min_chars: int, max_context_words: int):
    with input_path.open("r", encoding="utf-8") as input_file:
        for row_index, line in enumerate(input_file):
            payload = json.loads(line)
            messages = payload.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                continue

            user_message = next(
                (message for message in messages if message.get("role") == "user"),
                None,
            )
            assistant_message = next(
                (message for message in messages if message.get("role") == "assistant"),
                None,
            )
            if not user_message or not assistant_message:
                continue

            context = extract_context(str(user_message.get("content", "")))
            task_prompt = extract_task_prompt(str(user_message.get("content", "")))
            if not context or is_disclaimer_context(context):
                continue

            truncated_context = truncate_context_words(context, max_context_words)

            assistant = build_assistant_completion(
                truncated_context,
                min_chars=min_chars,
            )
            if not assistant:
                continue

            updated_messages = []
            for message in messages:
                copied = dict(message)
                if copied.get("role") == "user":
                    copied["content"] = (
                        f"Context:\n{truncated_context}\n\n{task_prompt}"
                    )
                if copied.get("role") == "assistant":
                    copied["content"] = assistant
                updated_messages.append(copied)

            yield {
                "messages": updated_messages,
                "metadata": {
                    **(
                        payload.get("metadata")
                        if isinstance(payload.get("metadata"), dict)
                        else {}
                    ),
                    "seed_row_index": row_index,
                    "assistant_char_count": len(assistant),
                    "source_context_word_count": len(context.split()),
                    "truncated_context_word_count": len(truncated_context.split()),
                    "draft_method": "heuristic_extract",
                    "review_status": "draft",
                },
            }


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input_path}")

    rows = list(
        iter_seed_rows(
            args.input_path,
            min_chars=args.min_assistant_chars,
            max_context_words=args.max_context_words,
        )
    )
    if not rows:
        raise RuntimeError("No usable seed rows were generated.")

    random.Random(args.seed).shuffle(rows)
    rows = rows[: max(1, args.max_rows)]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    LOGGER.info("Seed dataset written to %s", args.output_path)
    LOGGER.info("Rows written: %d", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
