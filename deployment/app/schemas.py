from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=3, max_length=2000, description="Natural-language analyst query.")
    top_k: int | None = Field(default=None, ge=1, le=8)


class SourceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_label: str = Field(min_length=2, max_length=16)
    relative_source: str = Field(min_length=1, max_length=1000)
    title: str | None = Field(default=None, max_length=1000)
    doc_id: str | None = Field(default=None, min_length=1, max_length=256)
    chunk_index: int | None = Field(default=None, ge=0)
    distance: float | None = None
    excerpt: str = Field(min_length=1, max_length=12000)


class EvidenceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    relative_source: str = Field(min_length=1, max_length=1000)
    title: str | None = Field(default=None, max_length=1000)
    doc_id: str | None = Field(default=None, min_length=1, max_length=256)
    chunk_index: int | None = Field(default=None, ge=0)
    excerpt: str = Field(min_length=1, max_length=12000)


class GenerateWithEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=3, max_length=2000)
    sources: list[EvidenceChunk] = Field(min_length=1, max_length=8)


class RetrieveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sources: list[SourceChunk]
    collection_name: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str
    answer_mode: str
    sources: list[SourceChunk]
    context_used: int
    collection_name: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: str
    collection_name: str
    collection_available: bool
    inference_available: bool
    embedding_model_name: str
    llm_model_name: str
