from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


CITATION_PATTERN = re.compile(r"\[S\d+\]")
CODE_DUMP_PATTERN = re.compile(
    r"(?:^|\n)(?:import\s+\w+|from\s+\w+\s+import|/[\w.-]+/[\w./-]+)"
)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None


def prepare_passage_text(text: str) -> str:
    return f"passage: {' '.join(text.split())}"


def prepare_query_text(query: str) -> str:
    return f"query: {' '.join(query.split())}"


def summarize_excerpt(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    truncated = compact[: limit - 3].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def parse_chroma_results(results: dict[str, list[list[Any]]]) -> list[RetrievedChunk]:
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for index, chunk_id in enumerate(ids):
        document = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None

        chunks.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                text=str(document or ""),
                metadata=metadata if isinstance(metadata, dict) else {},
                distance=float(distance) if distance is not None else None,
            )
        )
    return chunks


def build_context_block(chunks: list[RetrievedChunk], max_context_chars: int) -> str:
    if max_context_chars <= 0:
        return ""

    sections: list[str] = []
    current_length = 0

    for index, chunk in enumerate(chunks, start=1):
        relative_source = str(chunk.metadata.get("relative_source", "unknown_source"))
        title = str(chunk.metadata.get("title", "") or "")
        header = f"[S{index}] {relative_source}"
        if title and title != relative_source:
            header = f"{header} | {title}"

        body = " ".join(chunk.text.split())
        section = f"{header}\n{body}".strip()
        proposed_length = current_length + len(section) + 2

        if sections and proposed_length > max_context_chars:
            break

        if not sections and len(section) > max_context_chars:
            section = section[:max_context_chars].rsplit(" ", 1)[0].rstrip()

        sections.append(section)
        current_length += len(section) + 2

    return "\n\n".join(sections)


def build_source_records(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        sources.append(
            {
                "source_label": f"S{index}",
                "relative_source": str(
                    chunk.metadata.get("relative_source", "unknown_source")
                ),
                "title": chunk.metadata.get("title"),
                "doc_id": chunk.metadata.get("doc_id"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "distance": chunk.distance,
                "excerpt": summarize_excerpt(chunk.text),
            }
        )
    return sources


def answer_is_grounded(answer: str) -> bool:
    normalized = answer.strip()
    if not normalized:
        return False
    if not CITATION_PATTERN.search(normalized):
        return False
    if CODE_DUMP_PATTERN.search(normalized[:1200]):
        return False
    return True


def build_fallback_answer(chunks: list[RetrievedChunk], max_sources: int = 3) -> str:
    if not chunks:
        return "I could not find relevant evidence in the indexed research corpus."

    lines = [
        "I found relevant evidence in the indexed research corpus, but the local model did not return a grounded cited answer.",
        "Top evidence:",
    ]

    for index, chunk in enumerate(chunks[:max_sources], start=1):
        lines.append(f"[S{index}] {summarize_excerpt(chunk.text, limit=320)}")

    lines.append("Please refine the question if you want a narrower answer.")
    return "\n".join(lines)
