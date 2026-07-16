from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = ROOT / "deployment" / "bootstrap_local.py"


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("bootstrap_local", BOOTSTRAP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = _load_bootstrap()


def _write_models_and_dataset(tmp_path: Path, monkeypatch, model_name: str) -> None:
    """Point bootstrap's resolved paths at a temp deployment layout."""
    deployment_dir = tmp_path / "deployment"
    models_dir = deployment_dir / "models"
    models_dir.mkdir(parents=True)
    model_path = models_dir / model_name
    mmproj_path = models_dir / "Qwen3.5-4B.BF16-mmproj.gguf"
    model_path.write_text("gguf", encoding="utf-8")
    mmproj_path.write_text("mmproj", encoding="utf-8")
    (models_dir / "SHA256SUMS").write_text(
        f"{bootstrap.sha256_file(model_path)}  {model_path.name}\n"
        f"{bootstrap.sha256_file(mmproj_path)}  {mmproj_path.name}\n",
        encoding="utf-8",
    )

    ocr_dir = tmp_path / "ocr_pipeline"
    ocr_dir.mkdir(parents=True)
    (ocr_dir / "chroma_chunks.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "DEPLOYMENT_DIR", deployment_dir)
    monkeypatch.setattr(bootstrap, "REPO_ROOT", tmp_path)


def test_localgguf_validation_passes_with_matching_model(tmp_path, monkeypatch) -> None:
    _write_models_and_dataset(tmp_path, monkeypatch, "custom-model.gguf")
    env = {"CHROMA_AUTH_TOKEN": "a-real-token", "LLM_MODEL": "custom-model.gguf"}
    # Should not raise.
    bootstrap.validate_inputs(env, with_proxy=False, inference="localgguf")


def test_localgguf_validation_rejects_placeholder_token(tmp_path, monkeypatch) -> None:
    _write_models_and_dataset(tmp_path, monkeypatch, "custom-model.gguf")
    env = {"CHROMA_AUTH_TOKEN": "change-me", "LLM_MODEL": "custom-model.gguf"}
    with pytest.raises(RuntimeError, match="CHROMA_AUTH_TOKEN"):
        bootstrap.validate_inputs(env, with_proxy=False, inference="localgguf")


def test_localgguf_validation_requires_existing_model(tmp_path, monkeypatch) -> None:
    _write_models_and_dataset(tmp_path, monkeypatch, "present.gguf")
    env = {"CHROMA_AUTH_TOKEN": "a-real-token", "LLM_MODEL": "missing.gguf"}
    with pytest.raises(RuntimeError, match="Model file not found"):
        bootstrap.validate_inputs(env, with_proxy=False, inference="localgguf")


def test_localgguf_validation_rejects_checksum_mismatch(tmp_path, monkeypatch) -> None:
    _write_models_and_dataset(tmp_path, monkeypatch, "custom-model.gguf")
    model_path = tmp_path / "deployment" / "models" / "custom-model.gguf"
    model_path.write_text("changed", encoding="utf-8")
    env = {"CHROMA_AUTH_TOKEN": "a-real-token", "LLM_MODEL": "custom-model.gguf"}

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        bootstrap.validate_inputs(env, with_proxy=False, inference="localgguf")


def test_ollama_validation_skips_local_model_check(tmp_path, monkeypatch) -> None:
    # No GGUF is required because Ollama uses its persistent model volume.
    deployment_dir = tmp_path / "deployment"
    (deployment_dir / "models").mkdir(parents=True)
    ocr_dir = tmp_path / "ocr_pipeline"
    ocr_dir.mkdir(parents=True)
    (ocr_dir / "chroma_chunks.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "DEPLOYMENT_DIR", deployment_dir)
    monkeypatch.setattr(bootstrap, "REPO_ROOT", tmp_path)

    env = {
        "CHROMA_AUTH_TOKEN": "a-real-token",
        "OLLAMA_MODEL": "private-analyst-qwen35",
    }
    bootstrap.validate_inputs(env, with_proxy=False, inference="ollama")


def test_ollama_validation_requires_model_name(tmp_path, monkeypatch) -> None:
    deployment_dir = tmp_path / "deployment"
    (deployment_dir / "models").mkdir(parents=True)
    ocr_dir = tmp_path / "ocr_pipeline"
    ocr_dir.mkdir(parents=True)
    (ocr_dir / "chroma_chunks.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "DEPLOYMENT_DIR", deployment_dir)
    monkeypatch.setattr(bootstrap, "REPO_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="OLLAMA_MODEL"):
        bootstrap.validate_inputs(
            {"CHROMA_AUTH_TOKEN": "a-real-token"},
            with_proxy=False,
            inference="ollama",
        )


def test_inference_default_is_localgguf() -> None:
    import sys as _sys

    old_argv = _sys.argv
    try:
        _sys.argv = ["bootstrap_local.py"]
        args = bootstrap.parse_args()
    finally:
        _sys.argv = old_argv
    assert args.inference == "localgguf"
