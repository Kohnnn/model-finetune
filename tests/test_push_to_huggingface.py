from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUSH_PATH = ROOT / "finetune" / "push_to_huggingface.py"


def _load_push_module():
    spec = importlib.util.spec_from_file_location("push_to_huggingface", PUSH_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


push = _load_push_module()


def _release_manifest(tmp_path: Path, artifacts: dict[str, str]) -> dict:
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text("{}", encoding="utf-8")
    return {
        "run_manifest_sha256": push.sha256_file(run_manifest),
        "artifacts": artifacts,
    }


def test_verify_local_release_accepts_matching_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "merged_model" / "model.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"model")
    manifest = _release_manifest(
        tmp_path, {"merged_model/model.bin": push.sha256_file(artifact)}
    )
    (tmp_path / "SHA256SUMS").write_text(
        f"{push.sha256_file(artifact)}  merged_model/model.bin\n",
        encoding="utf-8",
    )

    push.verify_local_release(tmp_path, manifest)


def test_verify_local_release_rejects_path_outside_run_dir(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"outside")
    manifest = _release_manifest(
        tmp_path, {"../outside.bin": push.sha256_file(outside)}
    )

    with pytest.raises(RuntimeError, match="escapes run directory"):
        push.verify_local_release(tmp_path, manifest)


def test_verify_local_release_rejects_mismatched_checksum_inventory(tmp_path: Path) -> None:
    artifact = tmp_path / "merged_model" / "model.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"model")
    manifest = _release_manifest(
        tmp_path, {"merged_model/model.bin": push.sha256_file(artifact)}
    )
    (tmp_path / "SHA256SUMS").write_text("bad  other.bin\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA256SUMS"):
        push.verify_local_release(tmp_path, manifest)


def test_verify_remote_inventory_rejects_unhashed_files() -> None:
    with pytest.raises(RuntimeError, match="unhashed files"):
        push.verify_remote_inventory(
            {"README.md", "stale.bin"},
            {"README.md"},
        )


def test_verify_local_release_rejects_changed_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "merged_model" / "model.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"model")
    manifest = _release_manifest(
        tmp_path, {"merged_model/model.bin": push.sha256_file(artifact)}
    )
    artifact.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        push.verify_local_release(tmp_path, manifest)
