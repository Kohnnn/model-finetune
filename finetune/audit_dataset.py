from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


HEX_64 = re.compile(r"^[0-9a-fA-F]{64}$")
NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")
ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'`])(?:[A-Za-z]:[\\/]|/home/|/Users/|/var/|/tmp/|\\\\)")
SECRET = re.compile(
    r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|(?:api[_-]?key|secret|password|token)\s*[:=]|(?:sk|pk)_[A-Za-z0-9_-]{16,}",
    re.IGNORECASE,
)
CONTEXT_HEADER = re.compile(
    r"(?:^|\n)Context\s*:\s*(.*?)(?=\n\s*\nTask\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def normalize_context_section(user_content: str) -> str | None:
    match = CONTEXT_HEADER.search(user_content.replace("\r\n", "\n").replace("\r", "\n"))
    if not match:
        return None
    return "\n".join(line.strip() for line in match.group(1).strip().splitlines())


def context_sha256(user_content: str) -> str | None:
    context = normalize_context_section(user_content)
    if context is None:
        return None
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def numeric_tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in NUMBER.finditer(text)}


def token_shingles(text: str, width: int = 8) -> set[tuple[str, ...]]:
    tokens = [token.casefold() for token in re.findall(r"\w+|[^\w\s]", text)]
    return {
        tuple(tokens[index : index + width])
        for index in range(max(0, len(tokens) - width + 1))
    }


def assistant_context_copy_ratio(context: str, assistant: str, width: int = 8) -> float:
    assistant_shingles = token_shingles(assistant, width)
    if not assistant_shingles:
        return 0.0
    return len(assistant_shingles & token_shingles(context, width)) / len(assistant_shingles)


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _string_values(nested)


def _issue(errors: list[dict[str, Any]], row_number: int, code: str) -> None:
    errors.append({"row": row_number, "code": code})


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def _valid_spans(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for span in value:
        if not isinstance(span, dict):
            return False
        start_page = span.get("start_page")
        end_page = span.get("end_page")
        if (
            not isinstance(start_page, int)
            or isinstance(start_page, bool)
            or not isinstance(end_page, int)
            or isinstance(end_page, bool)
            or start_page <= 0
            or end_page <= 0
            or end_page < start_page
        ):
            return False
    return True


def _messages(row: dict[str, Any]) -> tuple[str, str] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) not in {2, 3}:
        return None
    expected_roles = ["user", "assistant"] if len(messages) == 2 else ["system", "user", "assistant"]
    if [message.get("role") if isinstance(message, dict) else None for message in messages] != expected_roles:
        return None
    if not all(isinstance(message.get("content"), str) for message in messages):
        return None
    return messages[-2]["content"], messages[-1]["content"]


def _metadata_errors(metadata: Any, user_content: str) -> list[str]:
    if not isinstance(metadata, dict):
        return ["metadata_missing"]
    required_text = [
        "doc_id",
        "document_family_id",
        "task_type",
        "reviewed_by",
        "approval_checklist_version",
    ]
    errors = [f"metadata_{name}_missing" for name in required_text if not isinstance(metadata.get(name), str) or not metadata[name].strip()]
    if metadata.get("review_status") != "approved":
        errors.append("review_status_not_approved")
    if metadata.get("language") not in {"en", "vi"}:
        errors.append("language_invalid")
    if not HEX_64.fullmatch(str(metadata.get("source_file_sha256", ""))):
        errors.append("source_file_sha256_invalid")
    actual_context_hash = context_sha256(user_content)
    if actual_context_hash is None or metadata.get("context_sha256") != actual_context_hash:
        errors.append("context_sha256_invalid")
    if not _valid_spans(metadata.get("source_spans")):
        errors.append("source_spans_invalid")
    if not _valid_timestamp(metadata.get("reviewed_at")):
        errors.append("reviewed_at_invalid")
    verified = metadata.get("verified_external_numbers", [])
    if not isinstance(verified, list) or not all(isinstance(value, (str, int, float)) and not isinstance(value, bool) for value in verified):
        errors.append("verified_external_numbers_invalid")
    return errors


def audit_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    distributions: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_targets: dict[str, set[str]] = defaultdict(set)
    exact_pairs: Counter[tuple[str, str]] = Counter()
    shingles: dict[tuple[str, ...], set[int]] = defaultdict(set)
    copy_ratios: list[float] = []
    total_words = 0
    total_tokens = 0
    approved_rows = 0
    total_rows = 0

    for row_number, row in enumerate(rows, start=1):
        total_rows += 1
        if not isinstance(row, dict):
            _issue(errors, row_number, "row_not_object")
            continue
        if "_audit_error" in row:
            _issue(errors, row_number, "jsonl_invalid")
            continue
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("review_status") != "approved":
            distributions["review_status"][str(metadata.get("review_status") if isinstance(metadata, dict) else "missing")] += 1
            continue
        approved_rows += 1
        parsed_messages = _messages(row)
        if parsed_messages is None:
            _issue(errors, row_number, "messages_invalid")
            continue
        user_content, assistant_content = parsed_messages
        if not assistant_content.strip():
            _issue(errors, row_number, "assistant_target_empty")
        for code in _metadata_errors(metadata, user_content):
            _issue(errors, row_number, code)
        for value in _string_values(row):
            if ABSOLUTE_PATH.search(value):
                _issue(errors, row_number, "absolute_local_path_detected")
                break
        for value in _string_values(row):
            if SECRET.search(value):
                _issue(errors, row_number, "secret_or_private_key_token_detected")
                break
        context = normalize_context_section(user_content)
        if context is not None:
            supported_numbers = numeric_tokens(context)
            verified_numbers = {str(value).casefold() for value in metadata.get("verified_external_numbers", []) if isinstance(value, (str, int, float)) and not isinstance(value, bool)}
            unsupported = numeric_tokens(assistant_content) - supported_numbers - verified_numbers
            if unsupported:
                _issue(errors, row_number, "assistant_number_unsupported")
            row_copy_ratio = assistant_context_copy_ratio(context, assistant_content)
            copy_ratios.append(row_copy_ratio)
            if row_copy_ratio >= 0.8:
                warnings.append(
                    {
                        "row": row_number,
                        "code": "assistant_context_copy_ratio_high",
                        "ratio": row_copy_ratio,
                    }
                )
        prompt_targets[user_content].add(assistant_content)
        exact_pairs[(user_content, assistant_content)] += 1
        words = re.findall(r"\S+", user_content + " " + assistant_content)
        total_words += len(words)
        total_tokens += len(re.findall(r"\w+|[^\w\s]", user_content + " " + assistant_content))
        distributions["language"][str(metadata.get("language"))] += 1
        distributions["task_type"][str(metadata.get("task_type"))] += 1
        distributions["document_family_id"][str(metadata.get("document_family_id"))] += 1
        if len(words) >= 8:
            for index in range(len(words) - 7):
                shingles[tuple(word.casefold() for word in words[index : index + 8])].add(row_number)

    duplicate_pairs = sum(count - 1 for count in exact_pairs.values() if count > 1)
    conflicting_prompts = sum(1 for targets in prompt_targets.values() if len(targets) > 1)
    duplicate_shingles = sum(1 for rows_with_shingle in shingles.values() if len(rows_with_shingle) > 1)
    cross_row_shingle_ratio = duplicate_shingles / max(1, len(shingles))
    mean_assistant_context_copy_ratio = sum(copy_ratios) / max(1, len(copy_ratios))
    if duplicate_pairs:
        errors.append({"row": None, "code": "duplicate_exact_prompt_target", "count": duplicate_pairs})
    if conflicting_prompts:
        errors.append({"row": None, "code": "conflicting_prompt_targets", "count": conflicting_prompts})
    if duplicate_shingles:
        warnings.append(
            {
                "code": "cross_row_shared_8_token_shingles",
                "count": duplicate_shingles,
                "ratio": cross_row_shingle_ratio,
            }
        )
    if approved_rows == 0:
        errors.append({"row": None, "code": "no_approved_rows"})
    return {
        "schema_version": 1,
        "rows_total": total_rows,
        "rows_approved": approved_rows,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "distributions": {name: dict(sorted(values.items())) for name, values in sorted(distributions.items())},
        "approximations": {"words": total_words, "tokens": total_tokens},
        "duplicates": {
            "exact_prompt_target_rows": duplicate_pairs,
            "conflicting_prompts": conflicting_prompts,
            "shared_8_token_shingles": duplicate_shingles,
            "cross_row_shared_8_token_shingle_ratio": cross_row_shingle_ratio,
            "mean_assistant_context_copy_ratio": mean_assistant_context_copy_ratio,
        },
    }


def stream_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                yield {"_audit_error": f"jsonl_blank_line_{row_number}"}
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                yield {"_audit_error": f"jsonl_invalid_{row_number}"}
                continue
            yield value


def audit_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    report = audit_rows(stream_jsonl(path))
    return report


def audit_summary(report: dict[str, Any]) -> dict[str, Any]:
    distributions = report.get("distributions") or {}
    return {
        "schema_version": report["schema_version"],
        "rows_total": report["rows_total"],
        "rows_approved": report["rows_approved"],
        "error_count": report["error_count"],
        "warning_count": report["warning_count"],
        "duplicates": report["duplicates"],
        "approximations": report["approximations"],
        "language_distribution": distributions.get("language", {}),
        "task_type_distribution": distributions.get("task_type", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit approved JSONL training data without printing row content.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_dataset(args.dataset_path)
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Audit complete: errors={report['error_count']} warnings={report['warning_count']}")
        return 1 if report["error_count"] else 0
    except Exception as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
