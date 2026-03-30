from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3, description="Natural-language analyst query.")
    top_k: int | None = Field(default=None, ge=1, le=8)


class SourceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_label: str
    relative_source: str
    title: str | None = None
    doc_id: str | None = None
    chunk_index: int | None = None
    distance: float | None = None
    excerpt: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[SourceChunk]
    context_used: int
    collection_name: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    collection_name: str
    collection_available: bool
    embedding_model_name: str
    llm_model_name: str
