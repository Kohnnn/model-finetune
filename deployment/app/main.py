from __future__ import annotations

import logging
import os
from functools import lru_cache
from urllib.parse import urlparse

import chromadb
from fastapi import FastAPI, HTTPException
from openai import OpenAI

from embeddings import get_embedding_model
from prompts import build_query_messages
from rag import (
    answer_is_grounded,
    build_context_block,
    build_fallback_answer,
    build_source_records,
    parse_chroma_results,
)
from schemas import HealthResponse, QueryRequest, QueryResponse
from settings import Settings, get_settings

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
            self.chroma_client.get_collection(name=self.settings.chroma_collection_name)
        except Exception:  # noqa: PERF203
            return False
        return True

    def get_collection(self):
        try:
            return self.chroma_client.get_collection(
                name=self.settings.chroma_collection_name
            )
        except Exception as exc:  # noqa: PERF203
            raise RuntimeError(
                "Chroma collection is unavailable. Run deployment/app/ingest.py first."
            ) from exc

    def retrieve_chunks(self, query: str, top_k: int) -> list:
        collection = self.get_collection()
        query_embedding = self.embedding_model.encode_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return parse_chroma_results(results)

    def generate_answer(self, query: str, chunks: list) -> str:
        context_block = build_context_block(
            chunks,
            max_context_chars=self.settings.max_context_chars,
        )
        if not context_block:
            return INSUFFICIENT_EVIDENCE_ANSWER

        response = self.llm_client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=build_query_messages(query, context_block),
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        message = response.choices[0].message.content
        if not message:
            return INSUFFICIENT_EVIDENCE_ANSWER
        answer = message.strip()
        if not answer_is_grounded(answer):
            LOGGER.warning("Model returned ungrounded answer; using fallback evidence")
            return build_fallback_answer(chunks)
        return answer


@lru_cache
def get_service() -> RAGService:
    return RAGService(get_settings())


settings = get_settings()
app = FastAPI(title="Private Analyst RAG API", version="0.1.0")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    service = get_service()
    return HealthResponse(
        status="ok",
        collection_name=settings.chroma_collection_name,
        collection_available=service.has_collection(),
        embedding_model_name=settings.embedding_model_name,
        llm_model_name=settings.llm_model_name,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    service = get_service()
    top_k = request.top_k or settings.retrieval_top_k

    try:
        chunks = service.retrieve_chunks(request.query, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: PERF203
        LOGGER.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed") from exc

    if not chunks:
        return QueryResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            sources=[],
            context_used=0,
            collection_name=settings.chroma_collection_name,
        )

    try:
        answer = service.generate_answer(request.query, chunks)
    except Exception as exc:  # noqa: PERF203
        LOGGER.exception("Generation failed")
        raise HTTPException(status_code=502, detail="Generation failed") from exc

    return QueryResponse(
        answer=answer,
        sources=build_source_records(chunks),
        context_used=len(chunks),
        collection_name=settings.chroma_collection_name,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
