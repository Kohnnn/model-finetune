from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    llama_api_url: str
    chroma_api_url: str
    chroma_auth_token: str
    chroma_collection_name: str
    embedding_model_name: str
    llm_model_name: str
    retrieval_top_k: int
    llm_temperature: float
    llm_max_tokens: int
    llm_request_timeout_seconds: float
    max_context_chars: int


@lru_cache
def get_settings() -> Settings:
    return Settings(
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=_read_int("APP_PORT", 8000),
        llama_api_url=os.getenv("LLAMA_API_URL", "http://llama:8080/v1"),
        chroma_api_url=os.getenv("CHROMA_API_URL", "http://chromadb:8000"),
        chroma_auth_token=os.getenv("CHROMA_AUTH_TOKEN", ""),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME", "research_chunks_v1"
        ),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "qwen"),
        retrieval_top_k=_read_int("RETRIEVAL_TOP_K", 4),
        llm_temperature=_read_float("LLM_TEMPERATURE", 0.1),
        llm_max_tokens=_read_int("LLM_MAX_TOKENS", 900),
        llm_request_timeout_seconds=_read_float("LLM_TIMEOUT_SECONDS", 120.0),
        max_context_chars=_read_int("MAX_CONTEXT_CHARS", 12000),
    )
