"""
generate_sft_dataset.py
=====================
Generates SFT responses by calling Ollama CLI directly (bypasses API streaming issues).
Each output row: {"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = (
    "You are a senior equity research analyst. Based ONLY on the provided context, "
    "deliver concise, evidence-based analytical commentary. "
    "Cite specific data points inline as [S1], [S2] when using evidence from the context. "
    "Do NOT copy full sentences from the context. "
    "Paraphrase and synthesize. If the context is insufficient, say so."
)

GENERATION_TEMPLATE = """Context:
{context}

Task: Deliver expert equity research commentary based on the context above.
Prioritize synthesis and strategic insight over factual reporting."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SFT rows via Ollama CLI.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "ocr_pipeline"
        / "finetune_template.jsonl",
        help="Input JSONL with existing dataset rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "datasets"
        / "vietcap_sft_generated.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=512, help="Max rows to process."
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:9b",
        help="Ollama model to use (default: qwen3.5:9b).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed (fixed for reproducibility).",
    )
    parser.add_argument(
        "--max-context-words",
        type=int,
        default=600,
        help="Truncate context to N words.",
    )
    parser.add_argument(
        "--max-response-tokens", type=int, default=400, help="Max tokens in response."
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Retry failed generations N times.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per row in seconds (default: 600).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip rows already in output file (by doc_id + chunk_index).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index in shuffled/input row list (0-based). Use with --end.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index in shuffled/input row list (exclusive). Use with --start.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def extract_context(user_content: str, max_words: int) -> str:
    ctx = user_content
    if "Context:" in ctx:
        ctx = ctx.split("Context:", 1)[1]
    for marker in ("\n\nTask:", "\nTask:"):
        if marker in ctx:
            ctx = ctx.split(marker, 1)[0]
    words = ctx.split()
    return " ".join(words[:max_words])


def iter_input_rows(input_path: Path, max_rows: int, seed: int):
    rows = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed line %d: %s", line_num, exc)
                continue
    random.seed(seed)
    random.shuffle(rows)
    return rows[:max_rows]


def load_existing_outputs(output_path: Path) -> set[str]:
    """Return set of row keys already generated (doc_id + chunk_index)."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                meta = row.get("metadata", {})
                doc_id = meta.get("doc_id", "")
                chunk_idx = meta.get("chunk_index", "")
                done.add(f"{doc_id}_{chunk_idx}")
            except json.JSONDecodeError:
                continue
    LOGGER.info("Resume: found %d existing rows in output", len(done))
    return done


def build_user_message(context: str) -> str:
    return GENERATION_TEMPLATE.format(context=context)


def generate_response_via_cli(
    model: str,
    context: str,
    max_tokens: int,
    temperature: float = 0.3,
    timeout: int = 600,
) -> str | None:
    user_msg = build_user_message(context)
    prompt = (
        f"<|im_start|>system\n{ANALYST_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else ""
            if stderr:
                LOGGER.warning("Ollama stderr: %s", stderr[:200])
        stdout = result.stdout
        if not stdout:
            return None
        answer = _extract_answer(stdout)
        return answer
    except subprocess.TimeoutExpired:
        LOGGER.warning("Ollama timed out after 600s")
        return None
    except Exception as exc:
        LOGGER.warning("Ollama run failed: %s", exc)
        return None


def _extract_answer(stdout: str) -> str | None:
    marker = "...done thinking."
    idx = stdout.rfind(marker)
    if idx != -1:
        answer = stdout[idx + len(marker) :].strip()
    else:
        lines = stdout.split("\n")
        answer_lines = []
        skip_patterns = (
            "thinking",
            "process",
            "analyze",
            "determine",
            "check",
            "draft",
            "review",
            "select",
            "final",
            "constraint",
            "step ",
            "* ",
            "- ",
            "1. ",
            "2. ",
            "3. ",
            "4. ",
            "5. ",
            "...done",
            "cw",
            "\x1b[",
        )
        for line in lines:
            lower = line.lower().strip()
            if any(p in lower for p in skip_patterns):
                continue
            stripped = line.strip()
            if stripped and not stripped.startswith("["):
                answer_lines.append(stripped)
        answer = " ".join(answer_lines[-10:])
    answer = answer.strip()
    if not answer:
        return None
    answer = " ".join(answer.split())
    return answer if answer else None


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.input.exists():
        LOGGER.error("Input file not found: %s", args.input)
        return 1

    if args.start is not None or args.end is not None:
        if args.start is None or args.end is None:
            LOGGER.error("Use both --start and --end together.")
            return 1
        if args.start >= args.end:
            LOGGER.error("--start (%d) must be < --end (%d).", args.start, args.end)
            return 1

    rows = iter_input_rows(args.input, args.max_rows, args.seed)
    LOGGER.info("Loaded %d total rows, max_rows=%d", len(rows), args.max_rows)

    done_keys: set[str] = set()
    if args.resume:
        done_keys = load_existing_outputs(args.output)

    slice_start = args.start if args.start is not None else 0
    slice_end = args.end if args.end is not None else len(rows)
    rows_to_process = rows[slice_start:slice_end]
    LOGGER.info(
        "Processing slice [%d:%d] = %d rows (resume=%s, existing=%d done)",
        slice_start,
        slice_end,
        len(rows_to_process),
        args.resume,
        len(done_keys),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    success_count = 0
    skip_count = 0
    skip_already_done = 0

    with args.output.open(mode, encoding="utf-8") as out_fh:
        for i, row in enumerate(rows_to_process, slice_start + 1):
            meta = row.get("metadata", {})
            doc_id = meta.get("doc_id", "")
            chunk_idx = meta.get("chunk_index", "")
            row_key = f"{doc_id}_{chunk_idx}"

            if args.resume and row_key in done_keys:
                skip_already_done += 1
                continue

            messages = row.get("messages", [])
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if not user_msg:
                skip_count += 1
                continue

            raw_content = str(user_msg.get("content", ""))
            context = extract_context(raw_content, args.max_context_words)
            if not context or len(context.split()) < 50:
                skip_count += 1
                LOGGER.debug("Skipping row %d — context too short", i)
                continue

            assistant_content = None
            for attempt in range(args.retry_attempts):
                assistant_content = generate_response_via_cli(
                    args.model,
                    context,
                    args.max_response_tokens,
                    temperature=0.3,
                    timeout=args.timeout,
                )
                if assistant_content:
                    break
                wait = 2**attempt
                LOGGER.warning(
                    "Retry %d/%d for row %d after %ds",
                    attempt + 1,
                    args.retry_attempts,
                    i,
                    wait,
                )
                time.sleep(wait)

            if not assistant_content:
                skip_count += 1
                LOGGER.debug("Skipping row %d — generation failed", i)
                continue

            new_row = {
                "metadata": meta,
                "messages": [
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(context)},
                    {"role": "assistant", "content": assistant_content},
                ],
            }
            out_fh.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            success_count += 1

            if i % 10 == 0 or i == slice_end:
                LOGGER.info(
                    "Progress: rows=%d success=%d skip=%d skip_already_done=%d",
                    i,
                    success_count,
                    skip_count,
                    skip_already_done,
                )

    LOGGER.info(
        "Done. total=%d success=%d skip=%d skip_already_done=%d",
        len(rows_to_process),
        success_count,
        skip_count,
        skip_already_done,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
