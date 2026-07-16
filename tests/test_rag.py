from __future__ import annotations

import hashlib

import pytest

from prompts import build_query_messages
from rag import (
    RetrievedChunk,
    answer_is_grounded,
    answer_is_refusal,
    build_context_block,
    build_fallback_answer,
    build_source_records,
    model_is_available,
)
from settings import (
    Settings,
    acquire_ingestion_lock,
    build_evaluation_target,
    collection_is_servable,
    evaluation_access_is_valid,
    evaluation_token_matches,
    hash_collection_snapshot,
    release_ingestion_lock,
    sha256_file,
    sign_evaluation_payload,
    snapshot_input,
)


def test_evaluation_token_fails_closed() -> None:
    assert evaluation_token_matches("", "supplied") is False
    assert evaluation_token_matches("configured", None) is False
    assert evaluation_token_matches("configured", "wrong") is False
    assert evaluation_token_matches("configured", "configured") is True


def evaluation_settings(tmp_path, token: str, key: str) -> Settings:
    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    index_path = tmp_path / "index.jsonl"
    for path in (model_path, mmproj_path, index_path):
        path.write_text("test", encoding="utf-8")
    checksums_path = tmp_path / "SHA256SUMS"
    checksums_path.write_text(
        f"{hashlib.sha256(model_path.read_bytes()).hexdigest()}  {model_path.name}\n"
        f"{hashlib.sha256(mmproj_path.read_bytes()).hexdigest()}  {mmproj_path.name}\n",
        encoding="utf-8",
    )
    return Settings(
        host="127.0.0.1",
        port=8000,
        llama_api_url="http://llama-server:8080/v1",
        chroma_api_url="http://chromadb:8000",
        chroma_auth_token="chroma",
        evaluation_api_token=token,
        evaluation_attestation_key=key,
        evaluation_model_path=str(model_path),
        evaluation_mmproj_path=str(mmproj_path),
        evaluation_index_path=str(index_path),
        evaluation_checksums_path=str(checksums_path),
        ingestion_lock_path=str(tmp_path / ".ingest.lock"),
        chroma_collection_name="test",
        embedding_model_name="test-embedding",
        llama_cpp_image=f"ghcr.io/ggerganov/llama.cpp@sha256:{'c' * 64}",
        llama_ctx_size=8192,
        llama_gpu_layers=32,
        llama_threads=4,
        llm_model_name="test-model",
        retrieval_top_k=4,
        llm_temperature=0.1,
        llm_max_tokens=100,
        llm_request_timeout_seconds=10.0,
        max_context_chars=1000,
    )


def test_evaluation_access_requires_distinct_strong_secrets_and_local_gguf(
    tmp_path,
) -> None:
    valid = evaluation_settings(tmp_path, "a" * 32, "b" * 32)
    assert evaluation_access_is_valid(valid, "a" * 32) is True
    assert evaluation_access_is_valid(valid, "wrong") is False
    assert evaluation_access_is_valid(
        evaluation_settings(tmp_path, "short", "b" * 32),
        "short",
    ) is False
    assert evaluation_access_is_valid(
        evaluation_settings(tmp_path, "a" * 32, "a" * 32),
        "a" * 32,
    ) is False
    assert evaluation_access_is_valid(
        Settings(**{**valid.__dict__, "llama_api_url": "http://ollama:11434/v1"}),
        "a" * 32,
    ) is False
    assert evaluation_access_is_valid(
        Settings(
            **{
                **valid.__dict__,
                "llama_cpp_image": "ghcr.io/ggerganov/llama.cpp:server",
            }
        ),
        "a" * 32,
    ) is False


def test_evaluation_access_rejects_llama_url_userinfo_bypass(tmp_path) -> None:
    valid = evaluation_settings(tmp_path, "a" * 32, "b" * 32)
    bypass = Settings(
        **{
            **valid.__dict__,
            "llama_api_url": "http://llama-server:8080@external-host/v1",
        }
    )
    assert evaluation_access_is_valid(bypass, "a" * 32) is False


def test_evaluation_access_rechecks_model_checksum(tmp_path) -> None:
    settings = evaluation_settings(tmp_path, "a" * 32, "b" * 32)
    assert evaluation_access_is_valid(settings, "a" * 32) is True
    model_path = tmp_path / "model.gguf"
    model_path.write_text("changed", encoding="utf-8")
    assert evaluation_access_is_valid(settings, "a" * 32) is False


def test_sha256_file_rehashes_changed_file(tmp_path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first")
    first = sha256_file(str(path))
    path.write_bytes(b"second")
    assert sha256_file(str(path)) != first


def test_snapshot_input_is_immutable(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"id":"first"}\n')
    snapshot, digest = snapshot_input(source)
    try:
        source.write_bytes(b'{"id":"second"}\n')
        assert snapshot.read_bytes() == b'{"id":"first"}\n'
        assert digest == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    finally:
        snapshot.unlink(missing_ok=True)


def test_ingestion_lock_is_exclusive_and_owner_released(tmp_path) -> None:
    lock_path = tmp_path / ".ingest.lock"
    token = acquire_ingestion_lock(lock_path)
    with pytest.raises(RuntimeError, match="Ingestion lock exists"):
        acquire_ingestion_lock(lock_path)
    release_ingestion_lock(lock_path, "wrong")
    assert lock_path.exists()
    release_ingestion_lock(lock_path, token)
    assert not lock_path.exists()


def test_collection_serving_rejects_ingesting_and_mismatched_inventory(
    tmp_path,
) -> None:
    class Collection:
        def __init__(self, state: str, rows: int):
            self.metadata = {
                "index_state": state,
                "index_generation": "c" * 32,
                "embedding_model": "test-embedding",
                "index_rows": rows,
            }

        def count(self):
            return 2

    lock_path = tmp_path / ".ingest.lock"
    assert collection_is_servable(Collection("complete", 2), "test-embedding") is True
    assert collection_is_servable(Collection("pilot", 2), "test-embedding") is True
    assert collection_is_servable(Collection("ingesting", 2), "test-embedding") is False
    assert collection_is_servable(Collection("complete", 1), "test-embedding") is False
    lock_path.write_text("active", encoding="utf-8")
    assert (
        collection_is_servable(
            Collection("complete", 2),
            "test-embedding",
            str(lock_path),
        )
        is False
    )


def test_evaluation_target_binds_runtime_and_generation_config(tmp_path) -> None:
    settings = evaluation_settings(tmp_path, "a" * 32, "b" * 32)
    target = build_evaluation_target(settings, "d" * 64, "e" * 32)
    runtime_changed = build_evaluation_target(
        Settings(**{**settings.__dict__, "llama_threads": 8}),
        "d" * 64,
        "e" * 32,
    )
    generation_changed = build_evaluation_target(
        Settings(**{**settings.__dict__, "llm_temperature": 0.2}),
        "d" * 64,
        "e" * 32,
    )
    assert target["runtime_sha256"] != runtime_changed["runtime_sha256"]
    assert (
        target["generation_config_sha256"]
        != generation_changed["generation_config_sha256"]
    )


def test_collection_snapshot_hash_binds_content_and_order() -> None:
    class Collection:
        def __init__(self, rows):
            self.rows = rows

        def count(self):
            return len(self.rows)

        def get(self, limit, offset, include):
            rows = self.rows[offset : offset + limit]
            return {
                "ids": [row[0] for row in rows],
                "documents": [row[1] for row in rows],
                "metadatas": [row[2] for row in rows],
                "embeddings": [row[3] for row in rows],
            }

    rows = [
        ("b", "second", {"doc_id": "2"}, [0.2]),
        ("a", "first", {"doc_id": "1"}, [0.1]),
    ]
    digest = hash_collection_snapshot(Collection(rows))
    assert digest == hash_collection_snapshot(Collection(list(reversed(rows))))
    changed = [("b", "changed", {"doc_id": "2"}, [0.2]), rows[1]]
    assert digest != hash_collection_snapshot(Collection(changed))


def test_evaluation_attestation_binds_payload() -> None:
    payload = {"answer": "grounded", "query_sha256": "a" * 64}
    signature = sign_evaluation_payload(payload, "k" * 32)
    assert signature == sign_evaluation_payload(payload, "k" * 32)
    assert signature != sign_evaluation_payload({**payload, "answer": "forged"}, "k" * 32)


def test_inference_health_requires_configured_model() -> None:
    assert model_is_available({"other-model"}, "private-model") is False
    assert model_is_available({"private-model:latest"}, "private-model") is True


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


def test_answer_is_grounded_requires_valid_citations() -> None:
    assert answer_is_grounded("Margins improved [S1].", source_count=1) is True
    assert answer_is_grounded("Margins improved [S2].", source_count=1) is False
    assert answer_is_grounded("Margins improved without citation.", source_count=1) is False
    assert answer_is_grounded("import torch\nfrom x import y [S1]", source_count=1) is False


def test_answer_is_refusal_detects_insufficient_evidence() -> None:
    assert answer_is_refusal("There is insufficient evidence in the context.") is True
    assert answer_is_refusal("Margins improved [S1].") is False


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
