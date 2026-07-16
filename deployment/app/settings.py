from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse


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


def evaluation_token_matches(configured: str, supplied: str | None) -> bool:
    return bool(
        configured
        and supplied
        and hmac.compare_digest(configured, supplied)
    )


def sign_evaluation_payload(payload: dict, token: str) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(token.encode("utf-8"), canonical, hashlib.sha256).hexdigest()


def _stat_signature(stat) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def snapshot_input(path: Path) -> tuple[Path, str]:
    digest = hashlib.sha256()
    snapshot_path: Path | None = None
    try:
        path_signature = _file_signature(path)
        with path.open("rb") as source, tempfile.NamedTemporaryFile(
            prefix="chroma_chunks_",
            suffix=".jsonl",
            delete=False,
        ) as snapshot:
            snapshot_path = Path(snapshot.name)
            source_signature = _stat_signature(os.fstat(source.fileno()))
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                snapshot.write(block)
            if (
                _stat_signature(os.fstat(source.fileno())) != source_signature
                or _file_signature(path) != path_signature
            ):
                raise RuntimeError(f"Input changed while snapshotting: {path}")
        return snapshot_path, digest.hexdigest()
    except Exception:
        if snapshot_path is not None:
            snapshot_path.unlink(missing_ok=True)
        raise


def acquire_ingestion_lock(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = os.urandom(16).hex()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Ingestion lock exists: {path}. Ensure no ingestion is active before removing it."
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
        lock_file.write(token)
    return token


def release_ingestion_lock(path: Path, token: str) -> None:
    try:
        if hmac.compare_digest(path.read_text(encoding="utf-8"), token):
            path.unlink()
    except FileNotFoundError:
        pass


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    return _stat_signature(path.stat())


@lru_cache(maxsize=32)
def _sha256_file_version(
    path: str,
    signature: tuple[int, int, int, int, int],
) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str) -> str:
    resolved = Path(path).resolve(strict=True)
    for _ in range(2):
        signature = _file_signature(resolved)
        digest = _sha256_file_version(str(resolved), signature)
        if _file_signature(resolved) == signature:
            return digest
    raise RuntimeError(f"File changed while hashing: {resolved}")


def hash_app_files(path: Path) -> str:
    files = sorted(path.glob("*.py"))
    signatures = {file_path: _file_signature(file_path) for file_path in files}
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(str(file_path)).encode("ascii"))
        digest.update(b"\n")
    if (
        sorted(path.glob("*.py")) != files
        or any(_file_signature(file_path) != signature for file_path, signature in signatures.items())
    ):
        raise RuntimeError("App files changed while hashing.")
    return digest.hexdigest()


def hash_collection_snapshot(collection, batch_size: int = 256) -> str:
    count = collection.count()
    records: list[tuple[str, str]] = []
    for offset in range(0, count, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = batch.get("ids") or []
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []
        embeddings = batch.get("embeddings")
        embeddings = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings or []
        if not len(ids) == len(documents) == len(metadatas) == len(embeddings):
            raise RuntimeError("Chroma index snapshot is incomplete.")
        for chunk_id, document, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embeddings,
        ):
            canonical = json.dumps(
                {
                    "id": str(chunk_id),
                    "document": str(document),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "embedding": [float(value) for value in embedding],
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            records.append((str(chunk_id), hashlib.sha256(canonical).hexdigest()))
    if len(records) != count or len({chunk_id for chunk_id, _ in records}) != count:
        raise RuntimeError("Chroma index snapshot inventory is invalid.")
    digest = hashlib.sha256()
    for chunk_id, record_hash in sorted(records):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def collection_is_servable(
    collection,
    embedding_model_name: str,
    ingestion_lock_path: str = "",
) -> bool:
    metadata = collection.metadata or {}
    index_rows = metadata.get("index_rows")
    return bool(
        not ingestion_lock_path
        or not Path(ingestion_lock_path).exists()
    ) and bool(
        metadata.get("index_state") in {"pilot", "complete"}
        and re.fullmatch(r"[0-9a-f]{32}", str(metadata.get("index_generation", "")))
        and metadata.get("embedding_model") == embedding_model_name
        and isinstance(index_rows, int)
        and not isinstance(index_rows, bool)
        and index_rows == collection.count()
    )


def _llama_api_is_local(url: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme == "http"
        and parsed.hostname == "llama-server"
        and port == 8080
        and parsed.username is None
        and parsed.password is None
        and parsed.path.rstrip("/") == "/v1"
        and not parsed.query
        and not parsed.fragment
    )


def _read_model_checksums(settings: Settings) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for raw_line in Path(settings.evaluation_checksums_path).read_text(
        encoding="utf-8"
    ).splitlines():
        parts = raw_line.strip().split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            continue
        filename = Path(parts[1].lstrip("* ")).name
        if filename in checksums:
            raise ValueError("Duplicate model checksum entry.")
        checksums[filename] = parts[0]
    return checksums


def _model_checksums_are_valid(settings: Settings) -> bool:
    try:
        checksums = _read_model_checksums(settings)
        return all(
            checksums.get(path.name) == sha256_file(str(path))
            for path in (
                Path(settings.evaluation_model_path),
                Path(settings.evaluation_mmproj_path),
            )
        )
    except (OSError, RuntimeError, UnicodeError, ValueError):
        return False


def evaluation_access_is_valid(settings: Settings, supplied_token: str | None) -> bool:
    return bool(
        len(settings.evaluation_api_token) >= 32
        and len(settings.evaluation_attestation_key) >= 32
        and not hmac.compare_digest(
            settings.evaluation_api_token,
            settings.evaluation_attestation_key,
        )
        and evaluation_token_matches(settings.evaluation_api_token, supplied_token)
        and _llama_api_is_local(settings.llama_api_url)
        and re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", settings.llama_cpp_image)
        and all(
            Path(path).is_file()
            for path in (
                settings.evaluation_model_path,
                settings.evaluation_mmproj_path,
                settings.evaluation_index_path,
            )
        )
        and _model_checksums_are_valid(settings)
    )


def _canonical_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_evaluation_target(
    settings: Settings,
    index_sha256: str,
    index_generation: str,
) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{64}", index_sha256) or not re.fullmatch(
        r"[0-9a-f]{32}", index_generation
    ):
        raise ValueError("Invalid evaluation index identity.")
    checksums = _read_model_checksums(settings)
    model_sha256 = sha256_file(settings.evaluation_model_path)
    mmproj_sha256 = sha256_file(settings.evaluation_mmproj_path)
    if (
        checksums.get(Path(settings.evaluation_model_path).name) != model_sha256
        or checksums.get(Path(settings.evaluation_mmproj_path).name) != mmproj_sha256
    ):
        raise ValueError("Evaluation model checksum mismatch.")
    return {
        "model_sha256": model_sha256,
        "mmproj_sha256": mmproj_sha256,
        "corpus_sha256": sha256_file(settings.evaluation_index_path),
        "index_sha256": index_sha256,
        "index_generation": index_generation,
        "app_sha256": hash_app_files(Path(__file__).resolve().parent),
        "runtime_sha256": _canonical_sha256(
            {
                "llama_cpp_image": settings.llama_cpp_image,
                "llama_api_url": settings.llama_api_url,
                "ctx_size": settings.llama_ctx_size,
                "n_gpu_layers": settings.llama_gpu_layers,
                "threads": settings.llama_threads,
            }
        ),
        "generation_config_sha256": _canonical_sha256(
            {
                "llm_model_name": settings.llm_model_name,
                "retrieval_top_k": settings.retrieval_top_k,
                "llm_temperature": settings.llm_temperature,
                "llm_max_tokens": settings.llm_max_tokens,
                "llm_request_timeout_seconds": settings.llm_request_timeout_seconds,
                "max_context_chars": settings.max_context_chars,
            }
        ),
        "collection_name": settings.chroma_collection_name,
        "embedding_model": settings.embedding_model_name,
    }


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    llama_api_url: str
    chroma_api_url: str
    chroma_auth_token: str
    evaluation_api_token: str
    evaluation_attestation_key: str
    evaluation_model_path: str
    evaluation_mmproj_path: str
    evaluation_index_path: str
    evaluation_checksums_path: str
    ingestion_lock_path: str
    chroma_collection_name: str
    embedding_model_name: str
    llama_cpp_image: str
    llama_ctx_size: int
    llama_gpu_layers: int
    llama_threads: int
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
        llama_api_url=os.getenv("LLAMA_API_URL", "http://llama-server:8080/v1"),
        chroma_api_url=os.getenv("CHROMA_API_URL", "http://chromadb:8000"),
        chroma_auth_token=os.getenv("CHROMA_AUTH_TOKEN", ""),
        evaluation_api_token=os.getenv("EVALUATION_API_TOKEN", ""),
        evaluation_attestation_key=os.getenv("EVALUATION_ATTESTATION_KEY", ""),
        evaluation_model_path=os.getenv("EVALUATION_MODEL_PATH", ""),
        evaluation_mmproj_path=os.getenv("EVALUATION_MMPROJ_PATH", ""),
        evaluation_index_path=os.getenv("EVALUATION_INDEX_PATH", ""),
        evaluation_checksums_path=os.getenv("EVALUATION_CHECKSUMS_PATH", ""),
        ingestion_lock_path=os.getenv("INGESTION_LOCK_PATH", ""),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME", "research_chunks_v1"
        ),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        llama_cpp_image=os.getenv(
            "LLAMA_CPP_IMAGE",
            "ghcr.io/ggerganov/llama.cpp:server",
        ),
        llama_ctx_size=_read_int("LLAMA_CTX_SIZE", 8192),
        llama_gpu_layers=_read_int("LLAMA_N_GPU_LAYERS", 32),
        llama_threads=_read_int("LLAMA_THREADS", 4),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "qwen3.5-private-analyst"),
        retrieval_top_k=_read_int("RETRIEVAL_TOP_K", 4),
        llm_temperature=_read_float("LLM_TEMPERATURE", 0.1),
        llm_max_tokens=_read_int("LLM_MAX_TOKENS", 900),
        llm_request_timeout_seconds=_read_float("LLM_TIMEOUT_SECONDS", 120.0),
        max_context_chars=_read_int("MAX_CONTEXT_CHARS", 12000),
    )
