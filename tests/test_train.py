from __future__ import annotations

from finetune.train import (
    apply_chat_template,
    choose_eval_document_ids,
    normalize_resume_from_checkpoint,
    resolve_hub_token,
)


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


def test_resume_true_is_normalized_for_transformers() -> None:
    assert normalize_resume_from_checkpoint("True") is True
    assert normalize_resume_from_checkpoint("true") is True
    assert normalize_resume_from_checkpoint("checkpoint-20") == "checkpoint-20"
    assert normalize_resume_from_checkpoint(None) is None
