from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from functools import lru_cache
from urllib.parse import urlparse

import chromadb
from fastapi import FastAPI, Header, HTTPException, Response
from openai import OpenAI

from embeddings import get_embedding_model
from prompts import build_query_messages
from rag import (
    answer_is_grounded,
    answer_is_refusal,
    build_context,
    build_fallback_answer,
    build_source_records,
    model_is_available,
    parse_chroma_results,
    RetrievedChunk,
)
from schemas import (
    GenerateWithEvidenceRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrieveResponse,
)
from settings import (
    Settings,
    build_evaluation_target,
    collection_is_servable,
    evaluation_access_is_valid,
    get_settings,
    hash_collection_snapshot,
    sign_evaluation_payload,
)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO"), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I could not find sufficient evidence in the indexed research corpus to "
    "answer this question confidently."
)


def create_chroma_client(settings: Settings) -> chromadb.HttpClient:
    parsed = urlparse(settings.chroma_api_url)
    host = parsed.hostname or "chromadb"
    port = parsed.port or (443 if parsed.scheme == "https" else 8000)
    headers: dict[str, str] = {}
    if settings.chroma_auth_token:
        headers["X-Chroma-Token"] = settings.chroma_auth_token

    return chromadb.HttpClient(
        host=host,
        port=port,
        ssl=parsed.scheme == "https",
        headers=headers,
    )


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm_client = OpenAI(
            base_url=settings.llama_api_url,
            api_key="sk-no-key-required",
            timeout=settings.llm_request_timeout_seconds,
        )
        self.chroma_client = create_chroma_client(settings)
        self.embedding_model = get_embedding_model(settings.embedding_model_name)

    def has_collection(self) -> bool:
        try:
            collection = self.chroma_client.get_collection(
                name=self.settings.chroma_collection_name
            )
            if not collection_is_servable(
                collection,
                self.settings.embedding_model_name,
                self.settings.ingestion_lock_path,
            ):
                return False
            embedding = self.embedding_model.encode_query("health check")
            collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["distances"],
            )
        except Exception:  # noqa: PERF203
            return False
        return True

    def has_inference(self) -> bool:
        try:
            models = self.llm_client.models.list()
            model_ids = {str(model.id) for model in models.data}
        except Exception:  # noqa: PERF203
            return False
        return model_is_available(model_ids, self.settings.llm_model_name)

    def get_collection(self):
        try:
            return self.chroma_client.get_collection(
                name=self.settings.chroma_collection_name
            )
        except Exception as exc:  # noqa: PERF203
            raise RuntimeError(
                "Chroma collection is unavailable. Run deployment/app/ingest.py first."
            ) from exc

    def collection_generation(self) -> str:
        collection = self.get_collection()
        if not collection_is_servable(
            collection,
            self.settings.embedding_model_name,
            self.settings.ingestion_lock_path,
        ):
            raise RuntimeError("Chroma index is incomplete. Run deployment/app/ingest.py first.")
        return str((collection.metadata or {}).get("index_generation", ""))

    def require_collection_generation(self, expected: str) -> None:
        if self.collection_generation() != expected:
            raise RuntimeError("Chroma index changed during the request. Retry the request.")

    def retrieve_chunks(self, query: str, top_k: int) -> tuple[list, str]:
        collection = self.get_collection()
        if not collection_is_servable(
            collection,
            self.settings.embedding_model_name,
            self.settings.ingestion_lock_path,
        ):
            raise RuntimeError("Chroma index is incomplete. Run deployment/app/ingest.py first.")
        index_generation = str((collection.metadata or {}).get("index_generation", ""))
        query_embedding = self.embedding_model.encode_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        self.require_collection_generation(index_generation)
        return parse_chroma_results(results), index_generation

    def generate_answer(self, query: str, chunks: list) -> tuple[str, str, int]:
        context_block, context_used = build_context(
            chunks,
            max_context_chars=self.settings.max_context_chars,
        )
        if not context_block:
            return INSUFFICIENT_EVIDENCE_ANSWER, "insufficient_evidence", 0

        response = self.llm_client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=build_query_messages(query, context_block),
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        message = response.choices[0].message.content
        if not message:
            return INSUFFICIENT_EVIDENCE_ANSWER, "insufficient_evidence", context_used
        answer = message.strip()
        if answer_is_refusal(answer):
            return INSUFFICIENT_EVIDENCE_ANSWER, "insufficient_evidence", context_used
        if not answer_is_grounded(answer, context_used):
            LOGGER.warning("Model returned an ungrounded answer; returning evidence excerpts")
            return (
                build_fallback_answer(chunks[:context_used]),
                "evidence_fallback",
                context_used,
            )
        return answer, "model", context_used


@lru_cache
def get_service() -> RAGService:
    return RAGService(get_settings())


settings = get_settings()
app = FastAPI(title="Private Analyst RAG API", version="0.1.0")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    service = get_service()
    collection_available = service.has_collection()
    inference_available = service.has_inference()
    return HealthResponse(
        status="ok" if collection_available and inference_available else "degraded",
        collection_name=settings.chroma_collection_name,
        collection_available=collection_available,
        inference_available=inference_available,
        embedding_model_name=settings.embedding_model_name,
        llm_model_name=settings.llm_model_name,
    )


def retrieve_or_raise(query_text: str, top_k: int) -> tuple[list, str]:
    try:
        return get_service().retrieve_chunks(query_text, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: PERF203
        LOGGER.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed") from exc


def generate_or_raise(query_text: str, chunks: list) -> tuple[str, str, int]:
    try:
        return get_service().generate_answer(query_text, chunks)
    except Exception as exc:  # noqa: PERF203
        LOGGER.exception("Generation failed")
        raise HTTPException(status_code=502, detail="Generation failed") from exc


@lru_cache(maxsize=8)
def get_verified_evaluation_target(
    index_sha256: str,
    corpus_sha256: str,
    index_rows: int,
    index_generation: str,
    target_json: str,
) -> dict[str, str]:
    collection = get_service().get_collection()
    metadata = collection.metadata or {}
    target = json.loads(target_json)
    if (
        not collection_is_servable(
            collection,
            settings.embedding_model_name,
            settings.ingestion_lock_path,
        )
        or metadata.get("index_state") != "complete"
        or metadata.get("embedding_model") != settings.embedding_model_name
        or metadata.get("corpus_sha256") != corpus_sha256
        or metadata.get("corpus_sha256") != target["corpus_sha256"]
        or metadata.get("index_sha256") != index_sha256
        or metadata.get("index_generation") != index_generation
        or metadata.get("index_sha256") != hash_collection_snapshot(collection)
        or int(metadata.get("index_rows", -1)) != index_rows
        or index_rows != collection.count()
    ):
        raise RuntimeError
    return target


def evaluation_target() -> dict[str, str]:
    collection = get_service().get_collection()
    metadata = collection.metadata or {}
    if not collection_is_servable(
        collection,
        settings.embedding_model_name,
        settings.ingestion_lock_path,
    ) or metadata.get("index_state") != "complete":
        raise RuntimeError
    index_sha256 = str(metadata.get("index_sha256", ""))
    index_generation = str(metadata.get("index_generation", ""))
    target = build_evaluation_target(settings, index_sha256, index_generation)
    return get_verified_evaluation_target(
        index_sha256,
        str(metadata.get("corpus_sha256", "")),
        int(metadata.get("index_rows", -1)),
        index_generation,
        json.dumps(target, sort_keys=True, separators=(",", ":")),
    )


def require_evaluation_token(supplied_token: str | None) -> dict[str, str]:
    if not evaluation_access_is_valid(settings, supplied_token):
        raise HTTPException(status_code=404, detail="Not found")
    try:
        return evaluation_target()
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc


def require_unchanged_evaluation_target(target: dict[str, str]) -> None:
    try:
        if evaluation_target() != target:
            raise RuntimeError
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Evaluation target changed") from exc


def evaluation_attestation_payload(
    endpoint: str,
    query_text: str,
    top_k: int | None,
    evidence_sha256: str | None,
    answer: str,
    answer_mode: str,
    source_doc_ids: list[str],
    elapsed_seconds: float,
    target: dict[str, str],
) -> dict:
    return {
        "endpoint": endpoint,
        "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
        "top_k": top_k,
        "evidence_sha256": evidence_sha256,
        "answer": answer,
        "answer_mode": answer_mode,
        "source_doc_ids": source_doc_ids,
        "elapsed_seconds": elapsed_seconds,
        "evaluation_target": target,
    }


def set_evaluation_attestation(
    response: Response,
    payload: dict,
) -> None:
    response.headers["X-Evaluation-Elapsed-Seconds"] = str(
        payload["elapsed_seconds"]
    )
    response.headers["X-Evaluation-Target"] = json.dumps(
        payload["evaluation_target"],
        sort_keys=True,
        separators=(",", ":"),
    )
    response.headers["X-Evaluation-Attestation"] = sign_evaluation_payload(
        payload,
        settings.evaluation_attestation_key,
    )


def evidence_sha256(sources: list[dict]) -> str:
    canonical = json.dumps(
        sources,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    request: QueryRequest,
    response: Response,
    x_evaluation_token: str | None = Header(default=None),
) -> RetrieveResponse:
    start = time.perf_counter()
    target = require_evaluation_token(x_evaluation_token)
    top_k = request.top_k or settings.retrieval_top_k
    chunks, _ = retrieve_or_raise(request.query, top_k)
    result = RetrieveResponse(
        sources=build_source_records(chunks),
        collection_name=settings.chroma_collection_name,
    )
    require_unchanged_evaluation_target(target)
    set_evaluation_attestation(
        response,
        evaluation_attestation_payload(
            "/retrieve",
            request.query,
            top_k,
            None,
            "",
            "unknown",
            [source.doc_id for source in result.sources if source.doc_id],
            time.perf_counter() - start,
            target,
        ),
    )
    return result


@app.post("/generate-with-evidence", response_model=QueryResponse)
def generate_with_evidence(
    request: GenerateWithEvidenceRequest,
    response: Response,
    x_evaluation_token: str | None = Header(default=None),
) -> QueryResponse:
    start = time.perf_counter()
    target = require_evaluation_token(x_evaluation_token)
    chunks = [
        RetrievedChunk(
            chunk_id=f"frozen-{index}",
            text=source.excerpt,
            metadata={
                "relative_source": source.relative_source,
                "title": source.title,
                "doc_id": source.doc_id,
                "chunk_index": source.chunk_index,
            },
        )
        for index, source in enumerate(request.sources, start=1)
    ]
    answer, answer_mode, context_used = generate_or_raise(request.query, chunks)
    result = QueryResponse(
        answer=answer,
        answer_mode=answer_mode,
        sources=build_source_records(chunks[:context_used]),
        context_used=context_used,
        collection_name=settings.chroma_collection_name,
    )
    require_unchanged_evaluation_target(target)
    set_evaluation_attestation(
        response,
        evaluation_attestation_payload(
            "/generate-with-evidence",
            request.query,
            None,
            evidence_sha256(
                [
                    source.model_dump(mode="json", exclude_none=True)
                    for source in request.sources
                ]
            ),
            result.answer,
            result.answer_mode,
            [source.doc_id for source in result.sources if source.doc_id],
            time.perf_counter() - start,
            target,
        ),
    )
    return result


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    response: Response,
    x_evaluation_token: str | None = Header(default=None),
) -> QueryResponse:
    start = time.perf_counter()
    target = None
    if x_evaluation_token is not None:
        target = require_evaluation_token(x_evaluation_token)
    top_k = request.top_k or settings.retrieval_top_k
    chunks, index_generation = retrieve_or_raise(request.query, top_k)
    if not chunks:
        result = QueryResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            answer_mode="insufficient_evidence",
            sources=[],
            context_used=0,
            collection_name=settings.chroma_collection_name,
        )
    else:
        answer, answer_mode, context_used = generate_or_raise(request.query, chunks)
        result = QueryResponse(
            answer=answer,
            answer_mode=answer_mode,
            sources=build_source_records(chunks[:context_used]),
            context_used=context_used,
            collection_name=settings.chroma_collection_name,
        )
    try:
        get_service().require_collection_generation(index_generation)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if x_evaluation_token is not None and target is not None:
        require_unchanged_evaluation_target(target)
        set_evaluation_attestation(
            response,
            evaluation_attestation_payload(
                "/query",
                request.query,
                top_k,
                None,
                result.answer,
                result.answer_mode,
                [source.doc_id for source in result.sources if source.doc_id],
                time.perf_counter() - start,
                target,
            ),
        )
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
