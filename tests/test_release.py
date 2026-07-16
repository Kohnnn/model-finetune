from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune.export_gguf import validate_export_inputs
from finetune.validate_release import (
    hash_directory,
    sha256_file,
    validate_release,
    verify_checksums,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_release_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    merged_dir = run_dir / "merged_model"
    gguf_dir = run_dir / "gguf"
    merged_dir.mkdir(parents=True)
    gguf_dir.mkdir()
    (merged_dir / "model.safetensors").write_bytes(b"weights")
    model_path = gguf_dir / "Qwen3.5-4B.Q4_K_M.gguf"
    mmproj_path = gguf_dir / "Qwen3.5-4B.BF16-mmproj.gguf"
    model_path.write_bytes(b"gguf")
    mmproj_path.write_bytes(b"mmproj")
    (gguf_dir / "SHA256SUMS").write_text(
        f"{sha256_file(model_path)}  {model_path.name}\n"
        f"{sha256_file(mmproj_path)}  {mmproj_path.name}\n",
        encoding="utf-8",
    )

    run_manifest_path = run_dir / "run_manifest.json"
    _write_json(
        run_manifest_path,
        {
            "base_model": "unsloth/Qwen3.5-4B",
            "base_model_revision": "a" * 40,
            "dataset_sha256": "e" * 64,
            "git_commit": "d" * 40,
            "git_dirty": False,
            "required_review_status": "approved",
            "eligible_rows": 10,
            "max_samples": None,
            "sample_limit_applied": False,
            "precision": "bfloat16",
            "assistant_only_loss": True,
            "split_strategy": "document_id",
            "train_rows": 8,
            "eval_rows": 2,
            "train_document_count": 2,
            "eval_document_count": 1,
            "train_document_ids": ["doc-a", "doc-b"],
            "eval_document_ids": ["doc-c"],
            "metrics": {
                "train_loss": 0.9,
                "baseline_loss": 1.5,
                "final_loss": 1.0,
            },
        },
    )
    _write_json(
        gguf_dir / "export_manifest.json",
        {
            "schema_version": 1,
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "base_model_revision": "a" * 40,
            "source_model": "merged_model",
            "source_model_sha256": hash_directory(merged_dir),
            "gguf_files": {
                model_path.name: sha256_file(model_path),
                mmproj_path.name: sha256_file(mmproj_path),
            },
        },
    )
    benchmark_path = tmp_path / "benchmark.json"
    _write_json(
        benchmark_path,
        {
            "summary": {"passed": 4, "total": 5, "fallback_safe": True},
            "comparison": {
                "baseline_passed": 4,
                "candidate_passed": 4,
                "pass_delta": 0,
                "question_set_matches": True,
                "regressed_ids": [],
                "regressed": False,
            },
        },
    )
    return run_dir, benchmark_path


def test_export_requires_merged_model_from_same_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    merged_dir = run_dir / "merged_model"
    adapter_dir = run_dir / "adapter"
    merged_dir.mkdir(parents=True)
    adapter_dir.mkdir()
    (merged_dir / "model.safetensors").write_bytes(b"merged")
    manifest_path = run_dir / "run_manifest.json"
    _write_json(manifest_path, {"base_model_revision": "abc123"})

    expected_manifest, manifest, source_hash = validate_export_inputs(
        merged_dir, run_dir, manifest_path
    )

    assert expected_manifest == manifest_path.resolve()
    assert manifest["base_model_revision"] == "abc123"
    assert source_hash == hash_directory(merged_dir)
    with pytest.raises(RuntimeError, match="merged_model"):
        validate_export_inputs(adapter_dir, run_dir, manifest_path)


def test_validate_release_writes_cards_manifest_and_hashes(tmp_path: Path) -> None:
    run_dir, benchmark_path = _build_release_fixture(tmp_path)

    manifest = validate_release(run_dir, benchmark_path)

    assert manifest["passed"] is True
    assert (run_dir / "merged_model" / "README.md").exists()
    assert (run_dir / "dataset_card.md").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "benchmark_summary.json").exists()
    assert not (run_dir / "benchmark.json").exists()
    assert (run_dir / "SHA256SUMS").exists()
    assert "README.md" in manifest["artifacts"]
    assert "run_summary.json" in manifest["artifacts"]
    verify_checksums(run_dir, manifest["artifacts"])


def test_validate_release_accepts_non_truncating_max_samples_and_crlf(tmp_path: Path) -> None:
    run_dir, benchmark_path = _build_release_fixture(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["max_samples"] = 100
    manifest_path.write_bytes(json.dumps(payload, indent=2).replace("\n", "\r\n").encode())
    export_manifest_path = run_dir / "gguf" / "export_manifest.json"
    export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    export_manifest["run_manifest_sha256"] = sha256_file(manifest_path)
    _write_json(export_manifest_path, export_manifest)

    manifest = validate_release(run_dir, benchmark_path)

    assert manifest["run_manifest_sha256"] == sha256_file(manifest_path)


def test_validate_release_rejects_stale_gguf_checksum(tmp_path: Path) -> None:
    run_dir, benchmark_path = _build_release_fixture(tmp_path)
    (run_dir / "gguf" / "Qwen3.5-4B.BF16-mmproj.gguf").write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="gguf_bundle"):
        validate_release(run_dir, benchmark_path)


def test_validate_release_rejects_unapproved_data(tmp_path: Path) -> None:
    run_dir, benchmark_path = _build_release_fixture(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["required_review_status"] = "draft"
    _write_json(manifest_path, payload)

    with pytest.raises(RuntimeError, match="approved_data_only"):
        validate_release(run_dir, benchmark_path)
