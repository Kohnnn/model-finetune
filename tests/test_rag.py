from __future__ import annotations

from prompts import build_query_messages
from rag import (
    RetrievedChunk,
    answer_is_grounded,
    build_context_block,
    build_fallback_answer,
    build_source_records,
)


def test_build_context_block_respects_limit() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            text="Revenue growth accelerated while margins remained stable.",
            metadata={"relative_source": "reports/a.docx", "title": "Report A"},
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            text="Leverage increased materially because expansion capex stayed elevated.",
            metadata={"relative_source": "reports/b.docx", "title": "Report B"},
        ),
    ]

    context = build_context_block(chunks, max_context_chars=110)

    assert "[S1] reports/a.docx" in context
    assert "[S2] reports/b.docx" not in context


def test_build_source_records_formats_excerpt_and_labels() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        text=" ".join(["evidence"] * 80),
        metadata={
            "relative_source": "reports/a.docx",
            "title": "Report A",
            "doc_id": "report_a",
            "chunk_index": 3,
        },
        distance=0.123,
    )

    records = build_source_records([chunk])

    assert records[0]["source_label"] == "S1"
    assert records[0]["doc_id"] == "report_a"
    assert records[0]["chunk_index"] == 3
    assert records[0]["excerpt"].endswith("...")


def test_build_query_messages_enforces_grounding() -> None:
    messages = build_query_messages("What changed?", "[S1] reports/a.docx\nEvidence")

    assert messages[0]["role"] == "system"
    assert "Use only the retrieved context" in messages[1]["content"]
    assert "[S1]" in messages[1]["content"]


def test_answer_is_grounded_requires_citations() -> None:
    assert answer_is_grounded("Margins improved [S1].") is True
    assert answer_is_grounded("Margins improved without citation.") is False
    assert answer_is_grounded("import torch\nfrom x import y [S1]") is False


def test_build_fallback_answer_uses_evidence_excerpts() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            text="Target price was raised while downside risk remained tied to funding costs.",
            metadata={"relative_source": "reports/a.docx", "title": "Report A"},
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            text="Margin pressure persisted because raw material costs stayed volatile.",
            metadata={"relative_source": "reports/b.docx", "title": "Report B"},
        ),
    ]

    answer = build_fallback_answer(chunks)

    assert "Top evidence:" in answer
    assert "[S1]" in answer
    assert "funding costs" in answer
