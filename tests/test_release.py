from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.evaluate_claim_ledger import (
    compare_baseline,
    evaluation_attestation,
    evidence_identity,
    pack_identity,
    question_identity,
    score_answer,
    score_retrieval,
    summarize_results,
    target_identity,
)
from finetune.export_gguf import validate_export_inputs
from finetune.validate_release import (
    hash_directory,
    sha256_file,
    validate_release,
    verify_checksums,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


ATTESTATION_KEY = "a" * 32
BASELINE_TARGET = {
    "model_sha256": "1" * 64,
    "mmproj_sha256": "2" * 64,
    "corpus_sha256": "3" * 64,
    "index_sha256": "5" * 64,
    "index_generation": "8" * 32,
    "app_sha256": "4" * 64,
    "runtime_sha256": "6" * 64,
    "generation_config_sha256": "7" * 64,
    "collection_name": "synthetic-collection",
    "embedding_model": "synthetic-embedding",
}


def _claim_report(pack: dict, target: dict) -> dict:
    results = []
    for lane in ("frozen", "retrieval", "live"):
        for case in pack["cases"]:
            if lane == "retrieval":
                evaluation_input = {
                    "answer": "",
                    "answer_mode": "unknown",
                    "source_doc_ids": ["synthetic-doc"],
                }
                response = {
                    "sources": [{"doc_id": "synthetic-doc"}],
                }
                result = score_retrieval(case, response, 1.0)
            elif case["expected_mode"] == "refusal":
                evaluation_input = {
                    "answer": "Insufficient evidence.",
                    "answer_mode": "insufficient_evidence",
                    "source_doc_ids": ["synthetic-doc"],
                }
                response = {
                    "answer": evaluation_input["answer"],
                    "answer_mode": evaluation_input["answer_mode"],
                    "sources": [{"doc_id": "synthetic-doc"}],
                }
                result = score_answer(case, response, 1.0)
            else:
                evaluation_input = {
                    "answer": "The metric increased to 42 [S1].",
                    "answer_mode": "model",
                    "source_doc_ids": ["synthetic-doc"],
                }
                response = {
                    "answer": evaluation_input["answer"],
                    "answer_mode": evaluation_input["answer_mode"],
                    "sources": [{"doc_id": "synthetic-doc"}],
                }
                result = score_answer(case, response, 1.0)
            evaluation_input.update(
                {
                    "endpoint": {
                        "frozen": "/generate-with-evidence",
                        "retrieval": "/retrieve",
                        "live": "/query",
                    }[lane],
                    "query_sha256": question_identity(case["question"]),
                    "top_k": None if lane == "frozen" else 4,
                    "evidence_sha256": evidence_identity(case["frozen_sources"])
                    if lane == "frozen"
                    else None,
                    "elapsed_seconds": 1.0,
                    "evaluation_target": target,
                }
            )
            evaluation_input["attestation"] = evaluation_attestation(
                evaluation_input,
                ATTESTATION_KEY,
            )
            result["lane"] = lane
            result["evaluation_input"] = evaluation_input
            results.append(result)
    return {
        "schema_version": 1,
        "pack_id": pack["pack_id"],
        "pack_sha256": pack_identity(pack),
        "case_count": len(pack["cases"]),
        "lanes": ["frozen", "retrieval", "live"],
        "top_k": 4,
        "question_identities": {
            case["id"]: question_identity(case["question"])
            for case in pack["cases"]
        },
        "evaluation_target": target,
        "evaluation_target_sha256": target_identity(target),
        "results": results,
        "summary": summarize_results(results),
    }


@pytest.fixture(autouse=True)
def _attestation_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALUATION_ATTESTATION_KEY", ATTESTATION_KEY)


def _build_release_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, Path]:
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
            "split_strategy": "document_family_id",
            "train_rows": 8,
            "eval_rows": 2,
            "train_document_count": 2,
            "eval_document_count": 1,
            "train_document_ids": ["doc-a", "doc-b"],
            "eval_document_ids": ["doc-c"],
            "train_family_ids": ["family-a", "family-b"],
            "eval_family_ids": ["family-c"],
            "audit": {
                "rows_approved": 10,
                "error_count": 0,
                "warning_count": 0,
            },
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
    cases = []
    for index in range(100):
        refusal = index == 99
        cases.append(
            {
                "id": f"synthetic-case-{index:03d}",
                "question": f"What is synthetic metric {index}?",
                "expected_mode": "refusal" if refusal else "model",
                "language": "en" if index % 2 == 0 else "vi",
                "task_type": "unsupported" if refusal else "fact_lookup",
                "critical": True,
                "claims": []
                if refusal
                else [
                    {
                        "id": "metric",
                        "required_terms": ["metric"],
                        "any_terms": ["increased"],
                        "numeric_values": [42],
                        "supporting_doc_ids": ["synthetic-doc"],
                        "prohibited_terms": ["guaranteed"],
                    }
                ],
                "frozen_sources": [
                    {
                        "relative_source": "synthetic/report.txt",
                        "doc_id": "synthetic-doc",
                        "excerpt": "The synthetic metric increased to 42.",
                    }
                ],
            }
        )
    pack = {"pack_id": "synthetic-release-pack", "cases": cases}
    monkeypatch.setenv(
        "EXPECTED_CLAIM_LEDGER_PACK_SHA256",
        pack_identity(pack),
    )
    claim_pack_path = tmp_path / "claim-ledger-pack.json"
    _write_json(claim_pack_path, pack)
    candidate_target = {
        **BASELINE_TARGET,
        "model_sha256": sha256_file(model_path),
        "mmproj_sha256": sha256_file(mmproj_path),
    }
    monkeypatch.setenv(
        "EXPECTED_BASELINE_EVALUATION_TARGET_SHA256",
        target_identity(BASELINE_TARGET),
    )
    baseline = _claim_report(pack, BASELINE_TARGET)
    claim_baseline_path = tmp_path / "claim-ledger-baseline.json"
    _write_json(claim_baseline_path, baseline)
    candidate = _claim_report(pack, candidate_target)
    candidate["baseline"] = compare_baseline(baseline, candidate)
    claim_ledger_path = tmp_path / "claim-ledger-candidate.json"
    _write_json(claim_ledger_path, candidate)
    return (
        run_dir,
        benchmark_path,
        claim_pack_path,
        claim_baseline_path,
        claim_ledger_path,
    )


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


def test_validate_release_writes_cards_manifest_and_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, benchmark_path, claim_pack_path, claim_baseline_path, claim_ledger_path = _build_release_fixture(tmp_path, monkeypatch)

    manifest = validate_release(
            run_dir,
            benchmark_path,
            claim_pack_path,
            claim_baseline_path,
            claim_ledger_path,
        )

    assert manifest["passed"] is True
    assert (run_dir / "merged_model" / "README.md").exists()
    assert (run_dir / "dataset_card.md").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "benchmark_summary.json").exists()
    claim_summary_path = run_dir / "claim_ledger_summary.json"
    assert claim_summary_path.exists()
    summary_text = claim_summary_path.read_text(encoding="utf-8")
    assert "synthetic-case-000" not in summary_text
    assert "synthetic-doc" not in summary_text
    assert "synthetic/report.txt" not in summary_text
    assert "The metric increased to 42" not in summary_text
    assert not (run_dir / "benchmark.json").exists()
    assert (run_dir / "SHA256SUMS").exists()
    assert "README.md" in manifest["artifacts"]
    assert "run_summary.json" in manifest["artifacts"]
    assert "claim_ledger_summary.json" in manifest["artifacts"]
    verify_checksums(run_dir, manifest["artifacts"])


def test_validate_release_accepts_non_truncating_max_samples_and_crlf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, benchmark_path, claim_pack_path, claim_baseline_path, claim_ledger_path = _build_release_fixture(tmp_path, monkeypatch)
    manifest_path = run_dir / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["max_samples"] = 100
    manifest_path.write_bytes(json.dumps(payload, indent=2).replace("\n", "\r\n").encode())
    export_manifest_path = run_dir / "gguf" / "export_manifest.json"
    export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    export_manifest["run_manifest_sha256"] = sha256_file(manifest_path)
    _write_json(export_manifest_path, export_manifest)

    manifest = validate_release(
            run_dir,
            benchmark_path,
            claim_pack_path,
            claim_baseline_path,
            claim_ledger_path,
        )

    assert manifest["run_manifest_sha256"] == sha256_file(manifest_path)


@pytest.mark.parametrize("pin", [None, "f" * 64])
def test_validate_release_requires_pinned_claim_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pin: str | None,
) -> None:
    run_dir, benchmark_path, claim_pack_path, claim_baseline_path, claim_ledger_path = _build_release_fixture(tmp_path, monkeypatch)
    if pin is None:
        monkeypatch.delenv("EXPECTED_CLAIM_LEDGER_PACK_SHA256")
    else:
        monkeypatch.setenv("EXPECTED_CLAIM_LEDGER_PACK_SHA256", pin)

    with pytest.raises(RuntimeError, match="EXPECTED_CLAIM_LEDGER_PACK_SHA256"):
        validate_release(
            run_dir,
            benchmark_path,
            claim_pack_path,
            claim_baseline_path,
            claim_ledger_path,
        )


def test_validate_release_requires_pinned_baseline_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, benchmark_path, claim_pack_path, claim_baseline_path, claim_ledger_path = _build_release_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("EXPECTED_BASELINE_EVALUATION_TARGET_SHA256", "f" * 64)

    with pytest.raises(RuntimeError, match="EXPECTED_BASELINE_EVALUATION_TARGET_SHA256"):
        validate_release(
            run_dir,
            benchmark_path,
            claim_pack_path,
            claim_baseline_path,
            claim_ledger_path,
        )


def test_validate_release_rejects_candidate_target_outside_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, benchmark_path, claim_pack_path, claim_baseline_path, claim_ledger_path = _build_release_fixture(tmp_path, monkeypatch)
    payload = json.loads(claim_ledger_path.read_text(encoding="utf-8"))
    target = {**payload["evaluation_target"], "model_sha256": "f" * 64}
    payload["evaluation_target"] = target
    payload["evaluation_target_sha256"] = target_identity(target)
    for result in payload["results"]:
        result["evaluation_input"]["evaluation_target"] = target
        result["evaluation_input"]["attestation"] = evaluation_attestation(
            result["evaluation_input"],
            ATTESTATION_KEY,
        )
    _write_json(claim_ledger_path, payload)

    with pytest.raises(RuntimeError, match="Candidate evaluation target"):
        validate_release(
            run_dir,
            benchmark_path,
            claim_pack_path,
            claim_baseline_path,
            claim_ledger_path,
        )


def test_validate_release_rejects_stale_gguf_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, benchmark_path, claim_pack_path, claim_baseline_path, claim_ledger_path = _build_release_fixture(tmp_path, monkeypatch)
    (run_dir / "gguf" / "Qwen3.5-4B.BF16-mmproj.gguf").write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="gguf_bundle"):
        validate_release(
            run_dir,
            benchmark_path,
            claim_pack_path,
            claim_baseline_path,
            claim_ledger_path,
        )


def test_validate_release_rejects_overlapping_or_missing_family_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, benchmark_path, claim_pack_path, claim_baseline_path, claim_ledger_path = _build_release_fixture(tmp_path, monkeypatch)
    manifest_path = run_dir / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["eval_family_ids"] = ["family-a"]
    _write_json(manifest_path, payload)

    with pytest.raises(RuntimeError, match="document_family_split"):
        validate_release(
            run_dir,
            benchmark_path,
            claim_pack_path,
            claim_baseline_path,
            claim_ledger_path,
        )


def test_validate_release_rejects_unapproved_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, benchmark_path, claim_pack_path, claim_baseline_path, claim_ledger_path = _build_release_fixture(tmp_path, monkeypatch)
    manifest_path = run_dir / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["required_review_status"] = "draft"
    _write_json(manifest_path, payload)

    with pytest.raises(RuntimeError, match="approved_data_only"):
        validate_release(
            run_dir,
            benchmark_path,
            claim_pack_path,
            claim_baseline_path,
            claim_ledger_path,
        )


def test_validate_release_rejects_incomplete_claim_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, benchmark_path, claim_pack_path, claim_baseline_path, claim_ledger_path = _build_release_fixture(tmp_path, monkeypatch)
    payload = json.loads(claim_ledger_path.read_text(encoding="utf-8"))
    payload["summary"]["total"] = 299
    _write_json(claim_ledger_path, payload)

    with pytest.raises(ValueError, match="Invalid claim-ledger pack"):
        validate_release(
            run_dir,
            benchmark_path,
            claim_pack_path,
            claim_baseline_path,
            claim_ledger_path,
        )


def test_validate_release_rejects_critical_claim_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, benchmark_path, claim_pack_path, claim_baseline_path, claim_ledger_path = _build_release_fixture(tmp_path, monkeypatch)
    pack = json.loads(claim_pack_path.read_text(encoding="utf-8"))
    baseline = json.loads(claim_baseline_path.read_text(encoding="utf-8"))
    payload = json.loads(claim_ledger_path.read_text(encoding="utf-8"))
    degraded_input = {
        "answer": "Insufficient evidence.",
        "answer_mode": "insufficient_evidence",
        "source_doc_ids": ["synthetic-doc"],
        "endpoint": "/query",
        "query_sha256": question_identity(pack["cases"][0]["question"]),
        "top_k": 4,
        "evidence_sha256": None,
        "elapsed_seconds": 1.0,
        "evaluation_target": payload["evaluation_target"],
    }
    degraded_input["attestation"] = evaluation_attestation(
        degraded_input,
        ATTESTATION_KEY,
    )
    degraded = score_answer(
        pack["cases"][0],
        {
            "answer": degraded_input["answer"],
            "answer_mode": degraded_input["answer_mode"],
            "sources": [{"doc_id": "synthetic-doc"}],
        },
        1.0,
    )
    degraded["lane"] = "live"
    degraded["evaluation_input"] = degraded_input
    payload["results"] = [
        degraded
        if result["lane"] == "live" and result["id"] == "synthetic-case-000"
        else result
        for result in payload["results"]
    ]
    payload["summary"] = summarize_results(payload["results"])
    payload["baseline"] = compare_baseline(baseline, payload)
    _write_json(claim_ledger_path, payload)

    with pytest.raises(RuntimeError, match="claim_ledger_comparison"):
        validate_release(
            run_dir,
            benchmark_path,
            claim_pack_path,
            claim_baseline_path,
            claim_ledger_path,
        )
