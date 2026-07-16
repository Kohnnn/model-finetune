from __future__ import annotations

import hashlib

import pytest

from finetune.audit_dataset import audit_rows
from finetune.train import (
    apply_chat_template,
    choose_eval_document_ids,
    normalize_resume_from_checkpoint,
    resolve_hub_token,
    select_complete_families,
    verify_formatted_examples_fit,
)


def _approved_row(answer: str = "Revenue was 10.") -> dict:
    context = "Revenue was 10."
    return {
        "messages": [
            {"role": "user", "content": f"Question: What was revenue?\nContext:\n{context}"},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "review_status": "approved",
            "doc_id": "doc-a",
            "document_family_id": "family-a",
            "source_file_sha256": "a" * 64,
            "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "task_type": "qa",
            "language": "en",
            "source_spans": [{"start_page": 1, "end_page": 1}],
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-07-16T00:00:00+00:00",
            "approval_checklist_version": "1",
            "verified_external_numbers": [],
        },
    }


class _Rows(list):
    def select(self, indices):
        return _Rows(self[index] for index in indices)


class _LengthTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


class _DummyTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        if "enable_thinking" in kwargs:
            return "disabled-thinking"
        return "default-template"


def test_apply_chat_template_disables_thinking_when_possible() -> None:
    tokenizer = _DummyTokenizer()

    result = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": "Hello"}],
        allow_thinking=False,
    )

    assert result == "disabled-thinking"
    assert tokenizer.calls[0]["enable_thinking"] is False


def test_resolve_hub_token_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN_TEST", "secret-value")

    assert resolve_hub_token("HF_TOKEN_TEST") == "secret-value"


def test_document_split_is_deterministic_and_disjoint() -> None:
    document_ids = ["doc-c", "doc-a", "doc-b", "doc-a"]

    first = choose_eval_document_ids(document_ids, eval_split=0.34, seed=3407)
    second = choose_eval_document_ids(document_ids, eval_split=0.34, seed=3407)

    assert first == second
    assert first
    assert first < set(document_ids)


def test_approved_audit_rejects_unsupported_numbers_and_duplicates() -> None:
    report = audit_rows([_approved_row("Revenue was 11."), _approved_row("Revenue was 11.")])

    assert report["error_count"] >= 2
    assert {error["code"] for error in report["errors"]} >= {
        "assistant_number_unsupported",
        "duplicate_exact_prompt_target",
    }


def test_family_sampling_keeps_complete_families_under_cap() -> None:
    dataset = _Rows(
        [
            {"metadata": {"document_family_id": "a"}},
            {"metadata": {"document_family_id": "a"}},
            {"metadata": {"document_family_id": "b"}},
        ]
    )

    selected = select_complete_families(dataset, max_samples=2, seed=3407)

    assert len(selected) == 2
    assert {row["metadata"]["document_family_id"] for row in selected} == {"a"}


def test_length_gate_refuses_truncation() -> None:
    with pytest.raises(RuntimeError, match="exceeds"):
        verify_formatted_examples_fit(_Rows([{"text": "one two three"}]), _LengthTokenizer(), 2)


def test_resume_true_is_normalized_for_transformers() -> None:
    assert normalize_resume_from_checkpoint("True") is True
    assert normalize_resume_from_checkpoint("true") is True
    assert normalize_resume_from_checkpoint("checkpoint-20") == "checkpoint-20"
    assert normalize_resume_from_checkpoint(None) is None
