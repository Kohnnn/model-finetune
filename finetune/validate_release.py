from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IGNORED_RELEASE_FILES = {"release_manifest.json", "SHA256SUMS"}
ROOT_RELEASE_FILES = [
    "README.md",
    "dataset_card.md",
    "run_summary.json",
    "benchmark_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce private-model release gates and build release metadata."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--max-train-loss", type=float, default=1.2)
    parser.add_argument("--min-benchmark-passes", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_hex_digest(value: Any, minimum_length: int, maximum_length: int) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(rf"[0-9a-fA-F]{{{minimum_length},{maximum_length}}}", value)
    )


def metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def collect_artifact_files(run_dir: Path) -> list[Path]:
    roots = [run_dir / "adapter", run_dir / "merged_model", run_dir / "gguf"]
    files = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.name not in IGNORED_RELEASE_FILES
    ]
    files.extend(
        run_dir / name for name in ROOT_RELEASE_FILES if (run_dir / name).is_file()
    )
    return sorted(set(files))


def build_checksums(run_dir: Path, files: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in files
    }


def hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(file for file in path.rglob("*") if file.is_file()):
        if file_path.name in {"README.md", "release_manifest.json", "SHA256SUMS"}:
            continue
        relative_path = file_path.relative_to(path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_checksum_names(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            checksums[parts[1].lstrip("* ").replace("\\", "/")] = parts[0]
    return checksums


def verify_checksums(run_dir: Path, checksums: dict[str, str]) -> None:
    for relative_path, expected in checksums.items():
        path = run_dir / relative_path
        if not path.exists():
            raise RuntimeError(f"Release artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Release artifact checksum mismatch: {path}")


def evaluate_gates(
    run_dir: Path,
    run_manifest: dict[str, Any],
    benchmark: dict[str, Any],
    artifact_files: list[Path],
    *,
    max_train_loss: float,
    min_benchmark_passes: int,
) -> list[dict[str, Any]]:
    metrics = run_manifest.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    train_docs = set(run_manifest.get("train_document_ids") or [])
    eval_docs = set(run_manifest.get("eval_document_ids") or [])
    summary = benchmark.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    comparison = benchmark.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    train_loss = metric(metrics, "train_loss")
    baseline_loss = metric(metrics, "baseline_loss", "baseline_eval_loss")
    final_loss = metric(metrics, "final_loss", "final_eval_loss")
    gguf_dir = run_dir / "gguf"
    gguf_files = [path for path in artifact_files if path.suffix.casefold() == ".gguf"]
    model_ggufs = [path for path in gguf_files if "mmproj" not in path.name.casefold()]
    mmproj_ggufs = [path for path in gguf_files if "mmproj" in path.name.casefold()]
    current_gguf_hashes = {
        path.relative_to(gguf_dir).as_posix(): sha256_file(path) for path in gguf_files
    }
    export_checksums = read_checksum_names(gguf_dir / "SHA256SUMS")
    export_manifest_path = gguf_dir / "export_manifest.json"
    try:
        export_manifest = read_json(export_manifest_path)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError):
        export_manifest = {}
    source_model = str(export_manifest.get("source_model", ""))
    try:
        source_model_path = (run_dir / source_model).resolve()
        source_model_relative = source_model_path.relative_to(run_dir.resolve())
        source_model_valid = (
            source_model_relative.as_posix() == "merged_model"
            and source_model_path.is_dir()
        )
    except ValueError:
        source_model_path = run_dir
        source_model_valid = False
    gguf_checksums_match = (
        len(model_ggufs) == 1
        and len(mmproj_ggufs) == 1
        and export_checksums == current_gguf_hashes
    )
    gguf_provenance_matches = (
        gguf_checksums_match
        and export_manifest.get("run_manifest_sha256")
        == sha256_file(run_dir / "run_manifest.json")
        and export_manifest.get("base_model_revision")
        == run_manifest.get("base_model_revision")
        and export_manifest.get("gguf_files") == current_gguf_hashes
        and source_model_valid
        and export_manifest.get("source_model_sha256")
        == hash_directory(source_model_path)
    )
    merged_weights = [
        path
        for path in artifact_files
        if "merged_model" in path.parts
        and path.suffix.casefold() in {".safetensors", ".bin"}
    ]

    return [
        {
            "name": "approved_data_only",
            "passed": run_manifest.get("required_review_status") == "approved",
            "detail": run_manifest.get("required_review_status"),
        },
        {
            "name": "full_approved_dataset",
            "passed": run_manifest.get("sample_limit_applied") is False
            and int(run_manifest.get("eligible_rows", 0))
            == int(run_manifest.get("train_rows", 0))
            + int(run_manifest.get("eval_rows", 0)),
            "detail": {
                "eligible_rows": run_manifest.get("eligible_rows"),
                "used_rows": int(run_manifest.get("train_rows", 0))
                + int(run_manifest.get("eval_rows", 0)),
                "sample_limit_applied": run_manifest.get("sample_limit_applied"),
            },
        },
        {
            "name": "document_split",
            "passed": bool(train_docs) and bool(eval_docs) and not (train_docs & eval_docs),
            "detail": {
                "train_documents": len(train_docs),
                "eval_documents": len(eval_docs),
                "overlap": sorted(train_docs & eval_docs),
            },
        },
        {
            "name": "assistant_only_loss",
            "passed": run_manifest.get("assistant_only_loss") is True,
            "detail": run_manifest.get("assistant_only_loss"),
        },
        {
            "name": "bf16_lora",
            "passed": run_manifest.get("precision") == "bfloat16",
            "detail": run_manifest.get("precision"),
        },
        {
            "name": "immutable_base_revision",
            "passed": is_hex_digest(run_manifest.get("base_model_revision"), 40, 64),
            "detail": run_manifest.get("base_model_revision"),
        },
        {
            "name": "dataset_fingerprint",
            "passed": is_hex_digest(run_manifest.get("dataset_sha256"), 64, 64),
            "detail": run_manifest.get("dataset_sha256"),
        },
        {
            "name": "committed_training_code",
            "passed": is_hex_digest(run_manifest.get("git_commit"), 40, 64)
            and run_manifest.get("git_dirty") is False,
            "detail": {
                "git_commit": run_manifest.get("git_commit"),
                "git_dirty": run_manifest.get("git_dirty"),
            },
        },
        {
            "name": "training_loss",
            "passed": train_loss is not None and train_loss < min(1.2, max_train_loss),
            "detail": {"actual": train_loss, "maximum": min(1.2, max_train_loss)},
        },
        {
            "name": "eval_improvement",
            "passed": (
                baseline_loss is not None
                and final_loss is not None
                and final_loss < baseline_loss
            ),
            "detail": {"baseline": baseline_loss, "final": final_loss},
        },
        {
            "name": "benchmark",
            "passed": (
                int(summary.get("passed", 0)) >= max(4, min_benchmark_passes)
                and int(summary.get("total", 0)) == 5
            ),
            "detail": {
                "passed": summary.get("passed", 0),
                "total": summary.get("total", 0),
                "minimum": max(4, min_benchmark_passes),
            },
        },
        {
            "name": "baseline_comparison",
            "passed": bool(comparison)
            and comparison.get("regressed") is False
            and comparison.get("question_set_matches") is True
            and not comparison.get("regressed_ids")
            and int(comparison.get("pass_delta", -1)) >= 0,
            "detail": comparison,
        },
        {
            "name": "fallback_safety",
            "passed": summary.get("fallback_safe") is True,
            "detail": summary.get("fallback_safe"),
        },
        {
            "name": "gguf_bundle",
            "passed": gguf_provenance_matches,
            "detail": {
                "models": [path.name for path in model_ggufs],
                "mmproj": [path.name for path in mmproj_ggufs],
                "export_checksums_match": gguf_checksums_match,
                "export_provenance_matches": gguf_provenance_matches,
                "source_model": source_model,
            },
        },
        {
            "name": "merged_model",
            "passed": bool(merged_weights),
            "detail": [path.name for path in merged_weights],
        },
    ]


def render_model_card(run_manifest: dict[str, Any], benchmark: dict[str, Any]) -> str:
    metrics = run_manifest.get("metrics") or {}
    summary = benchmark.get("summary") or {}
    return f"""---
library_name: transformers
base_model: {run_manifest.get('base_model')}
tags:
- finance
- rag
- lora
- private
---

# Private Analyst Qwen3.5-4B

Private internal model for grounded financial-research writing. The source corpus, prompts, and completions are proprietary and are not included.

## Lineage

- Base model: `{run_manifest.get('base_model')}`
- Base revision: `{run_manifest.get('base_model_revision')}`
- Dataset SHA-256: `{run_manifest.get('dataset_sha256')}`
- Training code commit: `{run_manifest.get('git_commit')}`
- Training worktree dirty: `{run_manifest.get('git_dirty')}`
- Precision: `{run_manifest.get('precision')}` LoRA
- Assistant-only loss: `{run_manifest.get('assistant_only_loss')}`

## Evaluation

- Train loss: `{metric(metrics, 'train_loss')}`
- Baseline eval loss: `{metric(metrics, 'baseline_loss', 'baseline_eval_loss')}`
- Final eval loss: `{metric(metrics, 'final_loss', 'final_eval_loss')}`
- Live benchmark: `{summary.get('passed', 0)}/{summary.get('total', 0)}` passed
- Fallback safety: `{summary.get('fallback_safe')}`

## Intended use

Use behind the repository's retrieval layer. The application rejects uncited answers and returns evidence excerpts or an insufficient-evidence response.

## Limitations

This model is not investment advice. It can reflect errors or omissions in the private training corpus and must not be used without retrieval, citations, and human review.
"""


def render_dataset_card(run_manifest: dict[str, Any]) -> str:
    return f"""---
pretty_name: Private Analyst Reviewed SFT Metadata
license: other
task_categories:
- text-generation
language:
- en
- vi
---

# Private Analyst Reviewed SFT Dataset

Metadata-only card for a proprietary supervised fine-tuning dataset. No source text, prompts, completions, filenames, or document identifiers are published.

## Provenance

- Dataset SHA-256: `{run_manifest.get('dataset_sha256')}`
- Approved rows used: `{run_manifest.get('train_rows', 0) + run_manifest.get('eval_rows', 0)}`
- Training documents: `{run_manifest.get('train_document_count')}`
- Evaluation documents: `{run_manifest.get('eval_document_count')}`
- Split strategy: `{run_manifest.get('split_strategy')}`
- Review status required: `{run_manifest.get('required_review_status')}`

## Privacy

The dataset contains proprietary financial research. Access must remain private and governed by the source-document licenses and organizational policy.
"""


def write_publishable_metadata(
    run_dir: Path,
    run_manifest: dict[str, Any],
    benchmark: dict[str, Any],
) -> None:
    merged_dir = run_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model_card = render_model_card(run_manifest, benchmark)
    (run_dir / "README.md").write_text(model_card, encoding="utf-8")
    (merged_dir / "README.md").write_text(model_card, encoding="utf-8")
    (run_dir / "dataset_card.md").write_text(
        render_dataset_card(run_manifest), encoding="utf-8"
    )
    run_summary_keys = [
        "schema_version",
        "created_at",
        "git_commit",
        "git_dirty",
        "base_model",
        "base_model_revision",
        "dataset_sha256",
        "eligible_rows",
        "required_review_status",
        "seed",
        "split_strategy",
        "train_rows",
        "eval_rows",
        "train_document_count",
        "eval_document_count",
        "max_seq_length",
        "batch_size",
        "gradient_accumulation",
        "learning_rate",
        "num_epochs",
        "lora_r",
        "lora_alpha",
        "precision",
        "assistant_only_loss",
        "packages",
        "metrics",
    ]
    run_summary = {key: run_manifest.get(key) for key in run_summary_keys}
    (run_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    benchmark_summary = {
        "schema_version": 1,
        "label": benchmark.get("label"),
        "summary": benchmark.get("summary"),
        "comparison": benchmark.get("comparison"),
    }
    (run_dir / "benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2), encoding="utf-8"
    )


def write_release_files(run_dir: Path, release_manifest: dict[str, Any]) -> None:
    manifest_text = json.dumps(release_manifest, indent=2)
    (run_dir / "release_manifest.json").write_text(manifest_text, encoding="utf-8")
    checksum_lines = [
        f"{digest}  {relative_path}"
        for relative_path, digest in sorted(release_manifest["artifacts"].items())
    ]
    (run_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )


def validate_release(
    run_dir: Path,
    benchmark_path: Path,
    *,
    max_train_loss: float = 1.2,
    min_benchmark_passes: int = 4,
) -> dict[str, Any]:
    run_manifest = read_json(run_dir / "run_manifest.json")
    benchmark = read_json(benchmark_path)
    artifact_files = collect_artifact_files(run_dir)
    gates = evaluate_gates(
        run_dir,
        run_manifest,
        benchmark,
        artifact_files,
        max_train_loss=max_train_loss,
        min_benchmark_passes=min_benchmark_passes,
    )
    passed = all(gate["passed"] for gate in gates)
    if not passed:
        failed_manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "passed": False,
            "visibility": "private",
            "gates": gates,
            "artifacts": {},
        }
        (run_dir / "release_manifest.json").write_text(
            json.dumps(failed_manifest, indent=2), encoding="utf-8"
        )
        failed = ", ".join(gate["name"] for gate in gates if not gate["passed"])
        raise RuntimeError(f"Release gates failed: {failed}")

    write_publishable_metadata(run_dir, run_manifest, benchmark)
    artifact_files = collect_artifact_files(run_dir)
    checksums = build_checksums(run_dir, artifact_files)
    release_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "visibility": "private",
        "run_summary": "run_summary.json",
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "benchmark_summary": "benchmark_summary.json",
        "benchmark_input_sha256": sha256_file(benchmark_path),
        "gates": gates,
        "artifacts": checksums,
    }
    write_release_files(run_dir, release_manifest)
    verify_checksums(run_dir, checksums)
    return release_manifest


def main() -> int:
    args = parse_args()
    try:
        manifest = validate_release(
            args.run_dir,
            args.benchmark_json,
            max_train_loss=args.max_train_loss,
            min_benchmark_passes=args.min_benchmark_passes,
        )
        print(f"Release gates passed with {len(manifest['artifacts'])} hashed artifacts.")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
