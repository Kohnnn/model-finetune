from __future__ import annotations

from deployment.evaluate_live_query import (
    compare_summaries,
    score_result,
    summarize_results,
)


def test_score_result_requires_cited_model_answer() -> None:
    item = {"expected_mode": "model", "minimum_keyword_hits": 2}

    passed, _ = score_result(
        item,
        "Margins and credit risk improved [S1].",
        "model",
        ["margin", "credit"],
        source_count=1,
    )
    fallback_passed, _ = score_result(
        item,
        "Top evidence [S1]",
        "evidence_fallback",
        ["margin", "credit"],
        source_count=1,
    )

    assert passed is True
    assert fallback_passed is False


def test_score_result_accepts_safe_refusal() -> None:
    passed, _ = score_result(
        {"expected_mode": "refusal"},
        "Insufficient evidence.",
        "insufficient_evidence",
        [],
        source_count=0,
    )

    assert passed is True


def test_score_result_rejects_unknown_citation_and_fallback_refusal() -> None:
    cited, _ = score_result(
        {"expected_mode": "model"},
        "Margins improved [S999].",
        "model",
        ["margin"],
        source_count=2,
    )
    refusal, _ = score_result(
        {"expected_mode": "refusal"},
        "Top evidence [S1].",
        "evidence_fallback",
        [],
        source_count=1,
    )

    assert cited is False
    assert refusal is False


def test_summary_tracks_fallback_safety() -> None:
    summary = summarize_results(
        [
            {
                "passed": True,
                "expected_mode": "model",
                "answer_mode": "model",
            },
            {
                "passed": True,
                "expected_mode": "refusal",
                "answer_mode": "insufficient_evidence",
            },
        ]
    )

    assert summary == {
        "passed": 2,
        "total": 2,
        "fallback_safe": True,
        "model_answers": 1,
        "evidence_fallbacks": 0,
    }


def test_comparison_detects_regression() -> None:
    comparison = compare_summaries(
        {
            "summary": {"passed": 1},
            "results": [
                {"id": "a", "passed": True},
                {"id": "b", "passed": False},
            ],
        },
        {"passed": 1},
        [
            {"id": "a", "passed": False},
            {"id": "b", "passed": True},
        ],
    )

    assert comparison["pass_delta"] == 0
    assert comparison["question_set_matches"] is True
    assert comparison["regressed_ids"] == ["a"]
    assert comparison["regressed"] is True
