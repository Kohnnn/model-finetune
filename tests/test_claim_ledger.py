from __future__ import annotations

import pytest

from deployment.evaluate_claim_ledger import (
    PackValidationError,
    compare_baseline,
    evaluate_pack,
    evaluation_attestation,
    nearest_rank,
    pack_identity,
    question_identity,
    rescore_report,
    retrieval_recall,
    score_answer,
    summarize_results,
    target_identity,
    validate_pack,
)


EVALUATION_TARGET = {
    "model_sha256": "1" * 64,
    "mmproj_sha256": "2" * 64,
    "corpus_sha256": "3" * 64,
    "index_sha256": "4" * 64,
    "index_generation": "8" * 32,
    "app_sha256": "5" * 64,
    "runtime_sha256": "6" * 64,
    "generation_config_sha256": "7" * 64,
    "collection_name": "test-collection",
    "embedding_model": "test-embedding",
}


def model_case() -> dict:
    return {
        "id": "case-1",
        "question": "What is the synthetic metric?",
        "expected_mode": "model",
        "language": "en",
        "task_type": "fact",
        "critical": True,
        "claims": [
            {
                "id": "claim-1",
                "required_terms": ["metric"],
                "any_terms": ["increased", "rose"],
                "numeric_values": [42],
                "supporting_doc_ids": ["doc-1"],
                "prohibited_terms": ["guaranteed"],
            }
        ],
    }


def test_validate_pack_rejects_unknown_fields_without_content() -> None:
    pack = {"pack_id": "test", "cases": [model_case()]}
    assert validate_pack(pack) == pack
    pack["cases"][0]["private_detail"] = "never expose"
    with pytest.raises(PackValidationError, match="Invalid claim-ledger pack\\."):
        validate_pack(pack)


def test_score_claims_numbers_and_citations() -> None:
    result = score_answer(
        model_case(),
        {
            "answer": "The metric increased to 42 [S1].",
            "answer_mode": "model",
            "sources": [{"doc_id": "doc-1"}],
        },
        0.25,
    )
    assert result["passed"] is True
    assert result["claim_accuracy"] == 1.0
    assert result["numeric_accuracy"] == 1.0
    assert result["valid_citation_syntax"] is True
    assert result["citation_precision"] == 1.0
    assert result["citation_completeness"] == 1.0


def test_score_accepts_signed_numeric_values() -> None:
    case = model_case()
    case["claims"][0]["numeric_values"] = [-42]
    result = score_answer(
        case,
        {
            "answer": "The metric increased by -42 [S1].",
            "answer_mode": "model",
            "sources": [{"doc_id": "doc-1"}],
        },
        0.25,
    )
    assert result["numeric_accuracy"] == 1.0


def test_score_accepts_locale_specific_numeric_strings() -> None:
    case = model_case()
    case["claims"][0]["numeric_values"] = ["1,5%", "1.500"]
    result = score_answer(
        case,
        {
            "answer": "The metric increased to 1,5%, or 1.500 units [S1].",
            "answer_mode": "model",
            "sources": [{"doc_id": "doc-1"}],
        },
        0.25,
    )
    assert result["numeric_accuracy"] == 1.0


def test_locale_specific_numeric_strings_reject_adjacent_digits() -> None:
    case = model_case()
    case["claims"][0]["numeric_values"] = ["1,5%"]
    result = score_answer(
        case,
        {
            "answer": "The metric increased to 11,5% [S1].",
            "answer_mode": "model",
            "sources": [{"doc_id": "doc-1"}],
        },
        0.25,
    )
    assert result["numeric_accuracy"] == 0.0


def test_score_rejects_invalid_citation_and_prohibited_term() -> None:
    result = score_answer(
        model_case(),
        {
            "answer": "The metric increased to 42 and is guaranteed [S2].",
            "answer_mode": "model",
            "sources": [{"doc_id": "doc-1"}],
        },
        0.25,
    )
    assert result["passed"] is False
    assert result["valid_citation_syntax"] is False
    assert result["claim_accuracy"] == 0.0


def test_all_lanes_require_frozen_sources() -> None:
    pack = {"pack_id": "test", "cases": [model_case()]}
    with pytest.raises(PackValidationError, match="Invalid claim-ledger pack\\."):
        evaluate_pack(pack, ["frozen", "retrieval", "live"], "http://localhost", 4, 1)


def test_rescore_report_rejects_forged_summary_and_duplicate_inventory() -> None:
    case = model_case()
    pack = {"pack_id": "test", "cases": [case]}
    result = score_answer(
        case,
        {
            "answer": "The metric increased to 42 [S1].",
            "answer_mode": "model",
            "sources": [{"doc_id": "doc-1"}],
        },
        0.25,
    )
    result["lane"] = "live"
    result["evaluation_input"] = {
        "endpoint": "/query",
        "query_sha256": question_identity(case["question"]),
        "top_k": 4,
        "evidence_sha256": None,
        "answer": "The metric increased to 42 [S1].",
        "answer_mode": "model",
        "source_doc_ids": ["doc-1"],
        "elapsed_seconds": 0.25,
        "evaluation_target": EVALUATION_TARGET,
    }
    result["evaluation_input"]["attestation"] = evaluation_attestation(
        result["evaluation_input"],
        "k" * 32,
    )
    report = {
        "schema_version": 1,
        "pack_id": "test",
        "pack_sha256": pack_identity(pack),
        "case_count": 1,
        "lanes": ["live"],
        "top_k": 4,
        "question_identities": {"case-1": question_identity(case["question"])},
        "evaluation_target": EVALUATION_TARGET,
        "evaluation_target_sha256": target_identity(EVALUATION_TARGET),
        "results": [result],
        "summary": summarize_results([result]),
    }
    assert rescore_report(pack, report, "k" * 32)["summary"] == report["summary"]

    report["results"][0]["evaluation_input"]["answer"] = "forged answer"
    with pytest.raises(PackValidationError):
        rescore_report(pack, report, "k" * 32)
    report["results"][0]["evaluation_input"]["answer"] = "The metric increased to 42 [S1]."
    report["results"][0]["evaluation_input"]["elapsed_seconds"] = 0.5
    report["results"][0]["elapsed_seconds"] = 0.5
    with pytest.raises(PackValidationError):
        rescore_report(pack, report, "k" * 32)
    report["results"][0]["evaluation_input"]["elapsed_seconds"] = 0.25
    report["results"][0]["elapsed_seconds"] = 0.25
    report["results"][0]["evaluation_input"]["evaluation_target"] = {
        **EVALUATION_TARGET,
        "model_sha256": "9" * 64,
    }
    with pytest.raises(PackValidationError):
        rescore_report(pack, report, "k" * 32)
    report["results"][0]["evaluation_input"]["evaluation_target"] = EVALUATION_TARGET
    report["summary"]["claim_accuracy"] = 0.0
    with pytest.raises(PackValidationError):
        rescore_report(pack, report)
    report["summary"] = summarize_results([result])
    report["results"].append(result)
    with pytest.raises(PackValidationError):
        rescore_report(pack, report)


def test_retrieval_recall_uses_ledger_doc_ids() -> None:
    case = model_case()
    case["claims"].append(
        {
            "id": "claim-2",
            "required_terms": [],
            "any_terms": [],
            "numeric_values": [],
            "supporting_doc_ids": ["doc-2"],
            "prohibited_terms": [],
        }
    )
    assert retrieval_recall(case, [{"doc_id": "doc-2"}, {"doc_id": "other"}]) == 0.5


def test_nearest_rank_percentiles() -> None:
    assert nearest_rank([1.0, 2.0, 3.0, 4.0], 50) == 2.0
    assert nearest_rank([1.0, 2.0, 3.0, 4.0], 95) == 4.0
    assert nearest_rank([], 95) is None


def test_summary_slices_and_modes() -> None:
    first = score_answer(
        model_case(),
        {"answer": "The metric increased to 42 [S1].", "answer_mode": "model", "sources": [{"doc_id": "doc-1"}]},
        1.0,
    )
    first["lane"] = "live"
    second = {
        "id": "case-2",
        "lane": "live",
        "expected_mode": "refusal",
        "language": "vi",
        "task_type": "unsupported",
        "critical": False,
        "answer_mode": "insufficient_evidence",
        "elapsed_seconds": 2.0,
        "claim_accuracy": 1.0,
        "numeric_accuracy": None,
        "numeric_expected": 0,
        "valid_citation_syntax": False,
        "citation_precision": 0.0,
        "citation_completeness": 1.0,
        "retrieval_recall_at_k": None,
        "false_refusal": False,
        "refusal_correct": True,
        "passed": True,
        "quality_score": 1.0,
        "claims": [],
    }
    summary = summarize_results([first, second])
    assert summary["mode_distribution"] == {"model": 1, "insufficient_evidence": 1}
    assert summary["by_language"]["en"]["total"] == 1
    assert summary["by_task_type"]["unsupported"]["refusal_correctness"] == 1.0
    assert summary["p50_latency_seconds"] == 1.0


def test_baseline_detects_critical_regression_and_identity_mismatch() -> None:
    baseline = {
        "pack_id": "pack",
        "question_identities": {"case-1": "a"},
        "results": [{"lane": "live", "id": "case-1", "quality_score": 1.0, "critical": True}],
    }
    candidate = {
        "pack_id": "pack",
        "question_identities": {"case-1": "b"},
        "results": [{"lane": "live", "id": "case-1", "quality_score": 0.5, "critical": True}],
    }
    comparison = compare_baseline(baseline, candidate)
    assert comparison["identity_matches"] is False
    assert comparison["loss"] == 1
    assert comparison["regressed_ids"] == ["case-1"]
    assert comparison["critical_regressions"] == ["case-1"]
