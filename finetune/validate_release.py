from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from deployment.evaluate_claim_ledger import (
        compare_baseline as compare_claim_ledger_baseline,
        load_pack as load_claim_ledger_pack,
        rescore_report as rescore_claim_ledger_report,
        target_identity as claim_ledger_target_identity,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deployment.evaluate_claim_ledger import (
        compare_baseline as compare_claim_ledger_baseline,
        load_pack as load_claim_ledger_pack,
        rescore_report as rescore_claim_ledger_report,
        target_identity as claim_ledger_target_identity,
    )


IGNORED_RELEASE_FILES = {"release_manifest.json", "SHA256SUMS"}
ROOT_RELEASE_FILES = [
    "README.md",
    "dataset_card.md",
    "run_summary.json",
    "benchmark_summary.json",
    "claim_ledger_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce private-model release gates and build release metadata."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--claim-ledger-pack", type=Path, required=True)
    parser.add_argument("--claim-ledger-baseline-json", type=Path, required=True)
    parser.add_argument("--claim-ledger-json", type=Path, required=True)
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
    claim_ledger: dict[str, Any],
    artifact_files: list[Path],
    *,
    max_train_loss: float,
    min_benchmark_passes: int,
) -> list[dict[str, Any]]:
    metrics = run_manifest.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    audit = run_manifest.get("audit")
    audit = audit if isinstance(audit, dict) else {}
    train_docs = set(run_manifest.get("train_document_ids") or [])
    eval_docs = set(run_manifest.get("eval_document_ids") or [])
    train_families = set(run_manifest.get("train_family_ids") or [])
    eval_families = set(run_manifest.get("eval_family_ids") or [])
    summary = benchmark.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    comparison = benchmark.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    claim_summary = claim_ledger.get("summary")
    claim_summary = claim_summary if isinstance(claim_summary, dict) else {}
    claim_comparison = claim_ledger.get("baseline")
    claim_comparison = claim_comparison if isinstance(claim_comparison, dict) else {}
    claim_lanes = set(claim_ledger.get("lanes") or [])
    claim_comparison_detail = {
        "identity_matches": claim_comparison.get("identity_matches"),
        "result_set_matches": claim_comparison.get("result_set_matches"),
        "wins": claim_comparison.get("win"),
        "ties": claim_comparison.get("tie"),
        "losses": claim_comparison.get("loss"),
        "regression_count": len(claim_comparison.get("regressed_ids") or []),
        "critical_regression_count": len(
            claim_comparison.get("critical_regressions") or []
        ),
    }
    train_loss = metric(metrics, "train_loss")
    baseline_loss = metric(metrics, "baseline_loss", "baseline_eval_loss")
    final_loss = metric(metrics, "final_loss", "final_eval_loss")
    claim_accuracy = metric(claim_summary, "claim_accuracy")
    numeric_accuracy = metric(claim_summary, "numeric_accuracy")
    citation_precision = metric(claim_summary, "citation_precision")
    citation_completeness = metric(claim_summary, "citation_completeness")
    retrieval_recall = metric(claim_summary, "retrieval_recall_at_k")
    refusal_correctness = metric(claim_summary, "refusal_correctness")
    p95_latency = metric(claim_summary, "p95_latency_seconds")
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
            "name": "approved_dataset_audit",
            "passed": audit.get("error_count") == 0
            and int(audit.get("rows_approved", 0))
            == int(run_manifest.get("eligible_rows", 0)),
            "detail": {
                "approved_rows": audit.get("rows_approved"),
                "eligible_rows": run_manifest.get("eligible_rows"),
                "errors": audit.get("error_count"),
                "warnings": audit.get("warning_count"),
            },
        },
        {
            "name": "document_family_split",
            "passed": (
                run_manifest.get("split_strategy") == "document_family_id"
                and bool(train_families)
                and bool(eval_families)
                and not (train_families & eval_families)
            ),
            "detail": {
                "split_strategy": run_manifest.get("split_strategy"),
                "train_families": len(train_families),
                "eval_families": len(eval_families),
                "overlap": sorted(train_families & eval_families),
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
            "name": "claim_ledger_quality",
            "passed": (
                claim_ledger.get("schema_version") == 1
                and is_hex_digest(claim_ledger.get("pack_sha256"), 64, 64)
                and int(claim_ledger.get("case_count", 0)) >= 100
                and claim_lanes == {"frozen", "retrieval", "live"}
                and int(claim_summary.get("total", 0))
                == int(claim_ledger.get("case_count", 0)) * 3
                and int(claim_summary.get("numeric_expected", 0)) >= 1
                and int(claim_summary.get("model_results", 0)) >= 1
                and int(claim_summary.get("refusal_results", 0)) >= 1
                and int(claim_summary.get("retrieval_results", 0)) >= 1
                and claim_accuracy is not None
                and claim_accuracy >= 0.75
                and numeric_accuracy is not None
                and numeric_accuracy >= 0.75
                and citation_precision is not None
                and citation_precision >= 0.75
                and citation_completeness is not None
                and citation_completeness >= 0.75
                and retrieval_recall is not None
                and retrieval_recall >= 0.75
                and refusal_correctness is not None
                and refusal_correctness == 1.0
                and int(claim_summary.get("false_refusals", 1)) == 0
                and p95_latency is not None
                and p95_latency <= 180.0
            ),
            "detail": {
                "pack_sha256": claim_ledger.get("pack_sha256"),
                "case_count": claim_ledger.get("case_count"),
                "lanes": sorted(claim_lanes),
                "scored_results": claim_summary.get("total"),
                "numeric_expected": claim_summary.get("numeric_expected"),
                "model_results": claim_summary.get("model_results"),
                "refusal_results": claim_summary.get("refusal_results"),
                "retrieval_results": claim_summary.get("retrieval_results"),
                "claim_accuracy": claim_summary.get("claim_accuracy"),
                "numeric_accuracy": claim_summary.get("numeric_accuracy"),
                "citation_precision": claim_summary.get("citation_precision"),
                "citation_completeness": claim_summary.get("citation_completeness"),
                "retrieval_recall_at_k": claim_summary.get("retrieval_recall_at_k"),
                "refusal_correctness": claim_summary.get("refusal_correctness"),
                "false_refusals": claim_summary.get("false_refusals"),
                "p95_latency_seconds": claim_summary.get("p95_latency_seconds"),
            },
        },
        {
            "name": "claim_ledger_comparison",
            "passed": (
                claim_comparison.get("identity_matches") is True
                and claim_comparison.get("result_set_matches") is True
                and not claim_comparison.get("critical_regressions")
            ),
            "detail": claim_comparison_detail,
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


def render_model_card(
    run_manifest: dict[str, Any],
    benchmark: dict[str, Any],
    claim_ledger: dict[str, Any],
) -> str:
    metrics = run_manifest.get("metrics") or {}
    summary = benchmark.get("summary") or {}
    claim_summary = claim_ledger.get("summary") or {}
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
- Fixed live smoke: `{summary.get('passed', 0)}/{summary.get('total', 0)}` passed
- Fallback safety: `{summary.get('fallback_safe')}`
- Private claim-ledger cases: `{claim_ledger.get('case_count')}` across `{', '.join(claim_ledger.get('lanes') or [])}`
- Claim accuracy: `{claim_summary.get('claim_accuracy')}`
- Numeric accuracy: `{claim_summary.get('numeric_accuracy')}`
- Citation precision: `{claim_summary.get('citation_precision')}`
- Citation completeness: `{claim_summary.get('citation_completeness')}`
- Retrieval recall@k: `{claim_summary.get('retrieval_recall_at_k')}`
- False refusals: `{claim_summary.get('false_refusals')}`

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
- Training document families: `{run_manifest.get('train_family_count')}`
- Evaluation document families: `{run_manifest.get('eval_family_count')}`
- Split strategy: `{run_manifest.get('split_strategy')}`
- Review status required: `{run_manifest.get('required_review_status')}`
- Dataset audit errors: `{(run_manifest.get('audit') or {}).get('error_count')}`
- Dataset audit warnings: `{(run_manifest.get('audit') or {}).get('warning_count')}`

## Privacy

The dataset contains proprietary financial research. Access must remain private and governed by the source-document licenses and organizational policy.
"""


def write_publishable_metadata(
    run_dir: Path,
    run_manifest: dict[str, Any],
    benchmark: dict[str, Any],
    claim_ledger: dict[str, Any],
) -> None:
    merged_dir = run_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model_card = render_model_card(run_manifest, benchmark, claim_ledger)
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
        "train_family_count",
        "eval_family_count",
        "audit",
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
    claim_comparison = claim_ledger.get("baseline")
    claim_comparison = claim_comparison if isinstance(claim_comparison, dict) else {}
    claim_ledger_summary = {
        "schema_version": 1,
        "pack_id": claim_ledger.get("pack_id"),
        "pack_sha256": claim_ledger.get("pack_sha256"),
        "case_count": claim_ledger.get("case_count"),
        "lanes": claim_ledger.get("lanes"),
        "evaluation_target_sha256": claim_ledger.get("evaluation_target_sha256"),
        "summary": claim_ledger.get("summary"),
        "baseline": {
            "identity_matches": claim_comparison.get("identity_matches"),
            "result_set_matches": claim_comparison.get("result_set_matches"),
            "wins": claim_comparison.get("win"),
            "ties": claim_comparison.get("tie"),
            "losses": claim_comparison.get("loss"),
            "regression_count": len(claim_comparison.get("regressed_ids") or []),
            "critical_regression_count": len(
                claim_comparison.get("critical_regressions") or []
            ),
        },
    }
    (run_dir / "claim_ledger_summary.json").write_text(
        json.dumps(claim_ledger_summary, indent=2), encoding="utf-8"
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
    claim_ledger_pack_path: Path,
    claim_ledger_baseline_path: Path,
    claim_ledger_path: Path,
    *,
    max_train_loss: float = 1.2,
    min_benchmark_passes: int = 4,
) -> dict[str, Any]:
    run_manifest = read_json(run_dir / "run_manifest.json")
    benchmark = read_json(benchmark_path)
    claim_ledger_pack = load_claim_ledger_pack(claim_ledger_pack_path)
    expected_pack_hash = os.getenv("EXPECTED_CLAIM_LEDGER_PACK_SHA256", "")
    actual_pack_hash = hashlib.sha256(
        json.dumps(
            claim_ledger_pack,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(expected_pack_hash, actual_pack_hash):
        raise RuntimeError(
            "Claim-ledger pack does not match EXPECTED_CLAIM_LEDGER_PACK_SHA256."
        )
    attestation_key = os.getenv("EVALUATION_ATTESTATION_KEY", "")
    if len(attestation_key) < 32:
        raise RuntimeError(
            "EVALUATION_ATTESTATION_KEY must be set to verify evaluation reports."
        )
    claim_ledger_baseline_input = read_json(claim_ledger_baseline_path)
    claim_ledger_input = read_json(claim_ledger_path)
    claim_ledger_baseline = rescore_claim_ledger_report(
        claim_ledger_pack,
        claim_ledger_baseline_input,
        attestation_key,
    )
    expected_baseline_target_hash = os.getenv(
        "EXPECTED_BASELINE_EVALUATION_TARGET_SHA256",
        "",
    )
    actual_baseline_target_hash = claim_ledger_target_identity(
        claim_ledger_baseline["evaluation_target"]
    )
    if not hmac.compare_digest(
        expected_baseline_target_hash,
        actual_baseline_target_hash,
    ):
        raise RuntimeError(
            "Baseline report does not match EXPECTED_BASELINE_EVALUATION_TARGET_SHA256."
        )
    claim_ledger = rescore_claim_ledger_report(
        claim_ledger_pack,
        claim_ledger_input,
        attestation_key,
    )
    export_manifest = read_json(run_dir / "gguf" / "export_manifest.json")
    gguf_files = export_manifest.get("gguf_files")
    if not isinstance(gguf_files, dict):
        raise RuntimeError("GGUF export manifest is invalid.")
    model_hashes = [
        digest
        for name, digest in gguf_files.items()
        if "mmproj" not in str(name).casefold()
    ]
    mmproj_hashes = [
        digest
        for name, digest in gguf_files.items()
        if "mmproj" in str(name).casefold()
    ]
    candidate_target = claim_ledger["evaluation_target"]
    if (
        len(model_hashes) != 1
        or len(mmproj_hashes) != 1
        or candidate_target["model_sha256"] != model_hashes[0]
        or candidate_target["mmproj_sha256"] != mmproj_hashes[0]
    ):
        raise RuntimeError("Candidate evaluation target does not match the GGUF export.")
    supplied_comparison = claim_ledger_input.get("baseline")
    computed_comparison = compare_claim_ledger_baseline(
        claim_ledger_baseline,
        claim_ledger,
    )
    if supplied_comparison != computed_comparison:
        raise RuntimeError("Claim-ledger comparison does not match recomputed results.")
    claim_ledger["baseline"] = computed_comparison
    artifact_files = collect_artifact_files(run_dir)
    gates = evaluate_gates(
        run_dir,
        run_manifest,
        benchmark,
        claim_ledger,
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

    write_publishable_metadata(run_dir, run_manifest, benchmark, claim_ledger)
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
        "claim_ledger_summary": "claim_ledger_summary.json",
        "claim_ledger_pack_sha256": actual_pack_hash,
        "baseline_evaluation_target_sha256": actual_baseline_target_hash,
        "candidate_evaluation_target_sha256": claim_ledger_target_identity(
            candidate_target
        ),
        "claim_ledger_pack_input_sha256": sha256_file(claim_ledger_pack_path),
        "claim_ledger_baseline_input_sha256": sha256_file(
            claim_ledger_baseline_path
        ),
        "claim_ledger_input_sha256": sha256_file(claim_ledger_path),
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
            args.claim_ledger_pack,
            args.claim_ledger_baseline_json,
            args.claim_ledger_json,
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
