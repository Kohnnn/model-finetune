from __future__ import annotations

from finetune.train import apply_chat_template, resolve_hub_token


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
