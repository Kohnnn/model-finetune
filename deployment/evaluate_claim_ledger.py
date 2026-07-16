from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])"
)
EVALUATION_TARGET_FIELDS = {
    "model_sha256",
    "mmproj_sha256",
    "corpus_sha256",
    "index_sha256",
    "index_generation",
    "app_sha256",
    "runtime_sha256",
    "generation_config_sha256",
    "collection_name",
    "embedding_model",
}


class PackValidationError(ValueError):
    pass


def fail_pack() -> None:
    raise PackValidationError("Invalid claim-ledger pack.")


def require_string(value: Any, minimum: int, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        fail_pack()
    return value


def require_string_list(value: Any, maximum: int = 32) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        fail_pack()
    return [require_string(item, 1, 300) for item in value]


def require_keys(value: Any, required: set[str], optional: set[str] = set()) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - required - optional or required - set(value):
        fail_pack()
    return value


def validate_source(value: Any) -> dict[str, Any]:
    source = require_keys(
        value,
        {"relative_source", "doc_id", "excerpt"},
        {"title", "chunk_index"},
    )
    require_string(source["relative_source"], 1, 1000)
    require_string(source["doc_id"], 1, 256)
    require_string(source["excerpt"], 1, 12000)
    if "title" in source and source["title"] is not None:
        require_string(source["title"], 1, 1000)
    if "chunk_index" in source and (
        isinstance(source["chunk_index"], bool)
        or not isinstance(source["chunk_index"], int)
        or source["chunk_index"] < 0
    ):
        fail_pack()
    return source


def validate_claim(value: Any) -> dict[str, Any]:
    claim = require_keys(
        value,
        {
            "id",
            "required_terms",
            "any_terms",
            "numeric_values",
            "supporting_doc_ids",
            "prohibited_terms",
        },
    )
    require_string(claim["id"], 1, 128)
    for key in ("required_terms", "any_terms", "supporting_doc_ids", "prohibited_terms"):
        require_string_list(claim[key])
    numeric_values = claim["numeric_values"]
    if not isinstance(numeric_values, list) or len(numeric_values) > 32:
        fail_pack()
    for number in numeric_values:
        if isinstance(number, str):
            require_string(number, 1, 64)
        elif (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            fail_pack()
    return claim


def validate_pack(payload: Any) -> dict[str, Any]:
    pack = require_keys(payload, {"pack_id", "cases"})
    require_string(pack["pack_id"], 1, 128)
    cases = pack["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 1000:
        fail_pack()
    seen_case_ids: set[str] = set()
    for case in cases:
        item = require_keys(
            case,
            {"id", "question", "expected_mode", "language", "task_type", "critical", "claims"},
            {"frozen_sources"},
        )
        case_id = require_string(item["id"], 1, 128)
        if case_id in seen_case_ids:
            fail_pack()
        seen_case_ids.add(case_id)
        require_string(item["question"], 3, 2000)
        if item["expected_mode"] not in {"model", "refusal"}:
            fail_pack()
        require_string(item["language"], 2, 32)
        require_string(item["task_type"], 1, 64)
        if not isinstance(item["critical"], bool):
            fail_pack()
        if not isinstance(item["claims"], list) or len(item["claims"]) > 32:
            fail_pack()
        claim_ids: set[str] = set()
        for claim in item["claims"]:
            validated_claim = validate_claim(claim)
            if validated_claim["id"] in claim_ids:
                fail_pack()
            claim_ids.add(validated_claim["id"])
        if item["expected_mode"] == "model" and not item["claims"]:
            fail_pack()
        if "frozen_sources" in item:
            sources = item["frozen_sources"]
            if not isinstance(sources, list) or not 1 <= len(sources) <= 8:
                fail_pack()
            for source in sources:
                validate_source(source)
    return pack


def load_pack(path: Path) -> dict[str, Any]:
    try:
        return validate_pack(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, PackValidationError) as exc:
        if isinstance(exc, PackValidationError):
            raise
        raise PackValidationError("Invalid claim-ledger pack.") from exc


def question_identity(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def pack_identity(pack: dict[str, Any]) -> str:
    canonical = json.dumps(pack, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_evaluation_target(value: Any) -> dict[str, str]:
    target = require_keys(value, EVALUATION_TARGET_FIELDS)
    for key in (
        "model_sha256",
        "mmproj_sha256",
        "corpus_sha256",
        "index_sha256",
        "app_sha256",
        "runtime_sha256",
        "generation_config_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(target[key])):
            fail_pack()
    if not re.fullmatch(r"[0-9a-f]{32}", str(target["index_generation"])):
        fail_pack()
    require_string(target["collection_name"], 1, 128)
    require_string(target["embedding_model"], 1, 256)
    return target


def target_identity(target: dict[str, str]) -> str:
    canonical = json.dumps(
        target,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_identity(sources: list[dict[str, Any]]) -> str:
    normalized = [
        {key: value for key, value in source.items() if value is not None}
        for source in sources
    ]
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def numeric_hits(answer: str, values: list[int | float | str]) -> list[int | float | str]:
    observed = [float(match.replace(",", "")) for match in NUMBER_PATTERN.findall(answer)]
    hits: list[int | float | str] = []
    for value in values:
        if isinstance(value, str):
            if re.search(
                rf"(?<!\d){re.escape(value)}(?!\d)",
                answer,
                flags=re.IGNORECASE,
            ):
                hits.append(value)
            continue
        target = float(value)
        if any(math.isclose(candidate, target, rel_tol=1e-9, abs_tol=1e-9) for candidate in observed):
            hits.append(value)
    return hits


def cited_source_indexes(answer: str) -> list[int]:
    return [int(index) for index in CITATION_PATTERN.findall(answer)]


def score_answer(case: dict[str, Any], response: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    answer = str(response.get("answer", ""))
    answer_mode = str(response.get("answer_mode", "unknown"))
    sources = response.get("sources") if isinstance(response.get("sources"), list) else []
    citations = cited_source_indexes(answer)
    valid_citation_syntax = bool(citations) and all(1 <= index <= len(sources) for index in citations)
    cited_doc_ids = {
        str(sources[index - 1].get("doc_id"))
        for index in citations
        if 1 <= index <= len(sources) and sources[index - 1].get("doc_id")
    }
    claim_results: list[dict[str, Any]] = []
    numeric_total = 0
    numeric_hit_total = 0
    for claim in case["claims"]:
        required_hits = [term for term in claim["required_terms"] if term.casefold() in answer.casefold()]
        any_hits = [term for term in claim["any_terms"] if term.casefold() in answer.casefold()]
        prohibited_hits = [term for term in claim["prohibited_terms"] if term.casefold() in answer.casefold()]
        number_hits = numeric_hits(answer, claim["numeric_values"])
        numeric_total += len(claim["numeric_values"])
        numeric_hit_total += len(number_hits)
        passed = (
            len(required_hits) == len(claim["required_terms"])
            and (not claim["any_terms"] or bool(any_hits))
            and not prohibited_hits
            and len(number_hits) == len(claim["numeric_values"])
        )
        supported = bool(set(claim["supporting_doc_ids"]) & cited_doc_ids)
        claim_results.append(
            {
                "id": claim["id"],
                "passed": passed,
                "supported_by_citation": supported,
                "required_terms_hit": len(required_hits),
                "required_terms_total": len(claim["required_terms"]),
                "numeric_hits": len(number_hits),
                "numeric_total": len(claim["numeric_values"]),
            }
        )
    claim_accuracy = mean([float(item["passed"]) for item in claim_results]) if claim_results else 1.0
    numeric_accuracy = numeric_hit_total / numeric_total if numeric_total else 1.0
    supporting_docs = {
        doc_id for claim in case["claims"] for doc_id in claim["supporting_doc_ids"]
    }
    citation_precision = len(cited_doc_ids & supporting_docs) / len(cited_doc_ids) if cited_doc_ids else 0.0
    citation_completeness = (
        mean([float(item["supported_by_citation"]) for item in claim_results])
        if claim_results
        else 1.0
    )
    false_refusal = case["expected_mode"] == "model" and answer_mode == "insufficient_evidence"
    refusal_correct = case["expected_mode"] == "refusal" and answer_mode == "insufficient_evidence"
    if case["expected_mode"] == "refusal":
        passed = refusal_correct
        quality_score = float(refusal_correct)
    else:
        passed = (
            answer_mode == "model"
            and not false_refusal
            and bool(valid_citation_syntax)
            and claim_accuracy == 1.0
            and numeric_accuracy == 1.0
            and citation_completeness == 1.0
        )
        quality_score = mean(
            [
                float(answer_mode == "model"),
                float(valid_citation_syntax),
                claim_accuracy,
                numeric_accuracy,
                citation_precision,
                citation_completeness,
            ]
        )
    return {
        "id": case["id"],
        "lane": "",
        "expected_mode": case["expected_mode"],
        "language": case["language"],
        "task_type": case["task_type"],
        "critical": case["critical"],
        "answer_mode": answer_mode,
        "elapsed_seconds": elapsed_seconds,
        "claim_accuracy": claim_accuracy if case["expected_mode"] == "model" else None,
        "numeric_accuracy": (
            numeric_accuracy
            if case["expected_mode"] == "model" and numeric_total
            else None
        ),
        "numeric_expected": numeric_total,
        "valid_citation_syntax": valid_citation_syntax if case["expected_mode"] == "model" else None,
        "citation_precision": citation_precision if case["expected_mode"] == "model" else None,
        "citation_completeness": citation_completeness if case["expected_mode"] == "model" else None,
        "retrieval_recall_at_k": None,
        "false_refusal": false_refusal,
        "refusal_correct": refusal_correct,
        "passed": passed,
        "quality_score": quality_score,
        "claims": claim_results,
    }


def retrieval_recall(
    case: dict[str, Any], sources: list[dict[str, Any]]
) -> float | None:
    expected = {doc_id for claim in case["claims"] for doc_id in claim["supporting_doc_ids"]}
    if not expected:
        return None
    returned = {str(source.get("doc_id")) for source in sources if source.get("doc_id")}
    return len(expected & returned) / len(expected)


def score_retrieval(case: dict[str, Any], response: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    sources = response.get("sources") if isinstance(response.get("sources"), list) else []
    recall = retrieval_recall(case, sources)
    return {
        "id": case["id"],
        "lane": "retrieval",
        "expected_mode": case["expected_mode"],
        "language": case["language"],
        "task_type": case["task_type"],
        "critical": case["critical"],
        "answer_mode": "retrieval_only",
        "elapsed_seconds": elapsed_seconds,
        "claim_accuracy": None,
        "numeric_accuracy": None,
        "numeric_expected": 0,
        "valid_citation_syntax": None,
        "citation_precision": None,
        "citation_completeness": None,
        "retrieval_recall_at_k": recall,
        "false_refusal": None,
        "refusal_correct": None,
        "passed": recall is None or recall == 1.0,
        "quality_score": 1.0 if recall is None else recall,
        "claims": [],
    }


def nearest_rank(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(percentile / 100 * len(values)) - 1)]


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float | None:
        values = [float(result[key]) for result in results if result.get(key) is not None]
        return mean(values) if values else None

    latencies = [float(result["elapsed_seconds"]) for result in results]
    modes: dict[str, int] = {}
    for result in results:
        mode = str(result["answer_mode"])
        modes[mode] = modes.get(mode, 0) + 1
    refusal_results = [
        result
        for result in results
        if result["expected_mode"] == "refusal"
        and result.get("refusal_correct") is not None
    ]
    return {
        "total": len(results),
        "passed": sum(bool(result["passed"]) for result in results),
        "pass_rate": rate("passed"),
        "claim_accuracy": rate("claim_accuracy"),
        "numeric_accuracy": rate("numeric_accuracy"),
        "numeric_expected": sum(int(result.get("numeric_expected", 0)) for result in results),
        "model_results": sum(
            result["expected_mode"] == "model" and result["lane"] != "retrieval"
            for result in results
        ),
        "refusal_results": sum(
            result["expected_mode"] == "refusal" and result["lane"] != "retrieval"
            for result in results
        ),
        "retrieval_results": sum(
            result["lane"] == "retrieval" and result["retrieval_recall_at_k"] is not None
            for result in results
        ),
        "valid_citation_syntax": rate("valid_citation_syntax"),
        "citation_precision": rate("citation_precision"),
        "citation_completeness": rate("citation_completeness"),
        "retrieval_recall_at_k": rate("retrieval_recall_at_k"),
        "false_refusals": sum(bool(result.get("false_refusal")) for result in results),
        "refusal_correctness": mean([float(result["refusal_correct"]) for result in refusal_results]) if refusal_results else None,
        "mode_distribution": modes,
        "p50_latency_seconds": nearest_rank(latencies, 50),
        "p95_latency_seconds": nearest_rank(latencies, 95),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = aggregate(results)
    for field, key in (("by_language", "language"), ("by_task_type", "task_type")):
        values = sorted({str(result[key]) for result in results})
        summary[field] = {
            value: aggregate([result for result in results if result[key] == value])
            for value in values
        }
    return summary


def attestation_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": value["endpoint"],
        "query_sha256": value["query_sha256"],
        "top_k": value["top_k"],
        "evidence_sha256": value["evidence_sha256"],
        "answer": value["answer"],
        "answer_mode": value["answer_mode"],
        "source_doc_ids": value["source_doc_ids"],
        "elapsed_seconds": value["elapsed_seconds"],
        "evaluation_target": value["evaluation_target"],
    }


def evaluation_attestation(value: dict[str, Any], key: str) -> str:
    canonical = json.dumps(
        attestation_payload(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()


def rescore_report(
    pack: dict[str, Any],
    report: Any,
    attestation_key: str | None = None,
) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        fail_pack()
    lanes = report.get("lanes")
    if (
        not isinstance(lanes, list)
        or len(lanes) != len(set(lanes))
        or any(lane not in {"frozen", "retrieval", "live"} for lane in lanes)
    ):
        fail_pack()
    expected_pack_hash = pack_identity(pack)
    report_target = validate_evaluation_target(report.get("evaluation_target"))
    if report.get("evaluation_target_sha256") != target_identity(report_target):
        fail_pack()
    expected_questions = {
        case["id"]: question_identity(case["question"]) for case in pack["cases"]
    }
    report_top_k = report.get("top_k")
    if (
        isinstance(report_top_k, bool)
        or not isinstance(report_top_k, int)
        or not 1 <= report_top_k <= 8
        or report.get("pack_id") != pack["pack_id"]
        or report.get("pack_sha256") != expected_pack_hash
        or report.get("case_count") != len(pack["cases"])
        or report.get("question_identities") != expected_questions
    ):
        fail_pack()
    if "frozen" in lanes and any("frozen_sources" not in case for case in pack["cases"]):
        fail_pack()
    cases = {case["id"]: case for case in pack["cases"]}
    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        fail_pack()
    rescored: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            fail_pack()
        case_id = raw_result.get("id")
        lane = raw_result.get("lane")
        elapsed = raw_result.get("elapsed_seconds")
        key = (str(lane), str(case_id))
        if (
            case_id not in cases
            or lane not in lanes
            or key in seen
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            fail_pack()
        seen.add(key)
        raw_input = raw_result.get("evaluation_input")
        response = scoring_response(raw_input)
        endpoint = {
            "frozen": "/generate-with-evidence",
            "retrieval": "/retrieve",
            "live": "/query",
        }[lane]
        expected_evidence_hash = (
            evidence_identity(cases[case_id]["frozen_sources"])
            if lane == "frozen"
            else None
        )
        expected_top_k = report.get("top_k") if lane in {"retrieval", "live"} else None
        if (
            raw_input["endpoint"] != endpoint
            or not math.isclose(
                float(raw_input["elapsed_seconds"]),
                float(elapsed),
                abs_tol=1e-12,
            )
            or raw_input["query_sha256"] != expected_questions[case_id]
            or raw_input["evaluation_target"] != report_target
            or raw_input["top_k"] != expected_top_k
            or raw_input["evidence_sha256"] != expected_evidence_hash
            or (
                attestation_key is not None
                and not hmac.compare_digest(
                    raw_input["attestation"],
                    evaluation_attestation(raw_input, attestation_key),
                )
            )
        ):
            fail_pack()
        signed_elapsed = float(raw_input["elapsed_seconds"])
        if lane == "retrieval":
            result = score_retrieval(cases[case_id], response, signed_elapsed)
        else:
            result = score_answer(cases[case_id], response, signed_elapsed)
        result["lane"] = lane
        result["evaluation_input"] = raw_result["evaluation_input"]
        rescored.append(result)
    expected = {(lane, case_id) for lane in lanes for case_id in cases}
    if seen != expected:
        fail_pack()
    canonical = {
        "schema_version": 1,
        "pack_id": pack["pack_id"],
        "pack_sha256": expected_pack_hash,
        "case_count": len(pack["cases"]),
        "lanes": lanes,
        "top_k": report_top_k,
        "question_identities": expected_questions,
        "evaluation_target": report_target,
        "evaluation_target_sha256": target_identity(report_target),
        "results": rescored,
        "summary": summarize_results(rescored),
    }
    if report.get("summary") != canonical["summary"]:
        fail_pack()
    return canonical


def compare_baseline(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_pack_hash = baseline.get("pack_sha256")
    candidate_pack_hash = candidate.get("pack_sha256")
    baseline_target = baseline.get("evaluation_target")
    candidate_target = candidate.get("evaluation_target")
    shared_target_keys = (
        "corpus_sha256",
        "index_sha256",
        "app_sha256",
        "runtime_sha256",
        "generation_config_sha256",
        "collection_name",
        "embedding_model",
    )
    identity_matches = (
        baseline.get("pack_id") == candidate.get("pack_id")
        and isinstance(baseline_pack_hash, str)
        and len(baseline_pack_hash) == 64
        and baseline_pack_hash == candidate_pack_hash
        and baseline.get("question_identities") == candidate.get("question_identities")
        and baseline.get("top_k") == candidate.get("top_k")
        and isinstance(baseline_target, dict)
        and isinstance(candidate_target, dict)
        and all(
            baseline_target.get(key) == candidate_target.get(key)
            for key in shared_target_keys
        )
    )
    baseline_results = {
        (str(result.get("lane")), str(result.get("id"))): result
        for result in baseline.get("results", [])
    }
    candidate_results = {
        (str(result.get("lane")), str(result.get("id"))): result
        for result in candidate.get("results", [])
    }
    result_set_matches = set(baseline_results) == set(candidate_results)
    pairs: list[dict[str, Any]] = []
    regressed_ids: set[str] = set()
    critical_regressions: set[str] = set()
    for key in sorted(set(baseline_results) & set(candidate_results)):
        previous = float(baseline_results[key].get("quality_score", 0.0))
        current = float(candidate_results[key].get("quality_score", 0.0))
        outcome = "tie" if math.isclose(previous, current, abs_tol=1e-12) else ("win" if current > previous else "loss")
        item = {"lane": key[0], "id": key[1], "baseline_score": previous, "candidate_score": current, "delta": current - previous, "outcome": outcome}
        pairs.append(item)
        if outcome == "loss":
            regressed_ids.add(key[1])
            if bool(candidate_results[key].get("critical")):
                critical_regressions.add(key[1])
    return {
        "identity_matches": identity_matches,
        "result_set_matches": result_set_matches,
        "win": sum(pair["outcome"] == "win" for pair in pairs),
        "tie": sum(pair["outcome"] == "tie" for pair in pairs),
        "loss": sum(pair["outcome"] == "loss" for pair in pairs),
        "pairs": pairs,
        "regressed_ids": sorted(regressed_ids),
        "critical_regressions": sorted(critical_regressions),
    }


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    evaluation_token: str = "",
) -> tuple[dict[str, Any], str, float, dict[str, str]]:
    headers = {"Content-Type": "application/json"}
    if evaluation_token:
        headers["X-Evaluation-Token"] = evaluation_token
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
        attestation = response.headers.get("X-Evaluation-Attestation", "")
        elapsed_text = response.headers.get("X-Evaluation-Elapsed-Seconds", "")
        target_text = response.headers.get("X-Evaluation-Target", "")
    try:
        elapsed_seconds = float(elapsed_text)
        target = validate_evaluation_target(json.loads(target_text))
    except (ValueError, json.JSONDecodeError, PackValidationError) as exc:
        raise RuntimeError("Endpoint returned an invalid response.") from exc
    if (
        not isinstance(data, dict)
        or not re.fullmatch(r"[0-9a-f]{64}", attestation)
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise RuntimeError("Endpoint returned an invalid response.")
    return data, attestation, elapsed_seconds, target


def evaluation_input(
    response: dict[str, Any],
    endpoint: str,
    query: str,
    top_k: int | None,
    evidence_hash: str | None,
    attestation: str,
    elapsed_seconds: float,
    target: dict[str, str],
) -> dict[str, Any]:
    sources = response.get("sources") if isinstance(response.get("sources"), list) else []
    return {
        "endpoint": endpoint,
        "query_sha256": question_identity(query),
        "top_k": top_k,
        "evidence_sha256": evidence_hash,
        "answer": str(response.get("answer", "")),
        "answer_mode": str(response.get("answer_mode", "unknown")),
        "source_doc_ids": [
            str(source["doc_id"])
            for source in sources
            if isinstance(source, dict) and source.get("doc_id")
        ],
        "elapsed_seconds": elapsed_seconds,
        "evaluation_target": target,
        "attestation": attestation,
    }


def scoring_response(value: Any) -> dict[str, Any]:
    item = require_keys(
        value,
        {
            "endpoint",
            "query_sha256",
            "top_k",
            "evidence_sha256",
            "answer",
            "answer_mode",
            "source_doc_ids",
            "elapsed_seconds",
            "evaluation_target",
            "attestation",
        },
    )
    require_string(item["endpoint"], 1, 64)
    if not re.fullmatch(r"[0-9a-f]{64}", str(item["query_sha256"])):
        fail_pack()
    if item["top_k"] is not None and (
        isinstance(item["top_k"], bool)
        or not isinstance(item["top_k"], int)
        or not 1 <= item["top_k"] <= 8
    ):
        fail_pack()
    if item["evidence_sha256"] is not None and not re.fullmatch(
        r"[0-9a-f]{64}", str(item["evidence_sha256"])
    ):
        fail_pack()
    validate_evaluation_target(item["evaluation_target"])
    elapsed_seconds = item["elapsed_seconds"]
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
        or not re.fullmatch(r"[0-9a-f]{64}", str(item["attestation"]))
    ):
        fail_pack()
    answer = require_string(item["answer"], 0, 20000)
    answer_mode = require_string(item["answer_mode"], 1, 64)
    source_doc_ids = require_string_list(item["source_doc_ids"], maximum=8)
    return {
        "answer": answer,
        "answer_mode": answer_mode,
        "sources": [{"doc_id": doc_id} for doc_id in source_doc_ids],
    }


def evaluate_lane(case: dict[str, Any], lane: str, base_url: str, top_k: int, timeout_seconds: int, evaluation_token: str) -> dict[str, Any]:
    if lane == "frozen":
        if "frozen_sources" not in case:
            raise PackValidationError("Invalid claim-ledger pack.")
        endpoint = "/generate-with-evidence"
        payload = {"query": case["question"], "sources": case["frozen_sources"]}
        effective_top_k = None
        evidence_hash = evidence_identity(case["frozen_sources"])
    elif lane == "retrieval":
        endpoint = "/retrieve"
        payload = {"query": case["question"], "top_k": top_k}
        effective_top_k = top_k
        evidence_hash = None
    else:
        endpoint = "/query"
        payload = {"query": case["question"], "top_k": top_k}
        effective_top_k = top_k
        evidence_hash = None
    start = time.perf_counter()
    response, attestation, signed_elapsed, target = post_json(
        f"{base_url.rstrip('/')}{endpoint}",
        payload,
        timeout_seconds,
        evaluation_token,
    )
    client_elapsed = time.perf_counter() - start
    if client_elapsed < signed_elapsed:
        raise RuntimeError("Endpoint returned an invalid response.")
    result = score_retrieval(case, response, signed_elapsed) if lane == "retrieval" else score_answer(case, response, signed_elapsed)
    result["lane"] = lane
    result["evaluation_input"] = evaluation_input(
        response,
        endpoint,
        case["question"],
        effective_top_k,
        evidence_hash,
        attestation,
        signed_elapsed,
        target,
    )
    return result


def evaluate_pack(
    pack: dict[str, Any],
    lanes: list[str],
    base_url: str,
    top_k: int,
    timeout_seconds: int,
    evaluation_token: str = "",
) -> list[dict[str, Any]]:
    if "frozen" in lanes and any("frozen_sources" not in case for case in pack["cases"]):
        raise PackValidationError("Invalid claim-ledger pack.")
    return [
        evaluate_lane(
            case,
            lane,
            base_url,
            top_k,
            timeout_seconds,
            evaluation_token,
        )
        for lane in lanes
        for case in pack["cases"]
    ]


def thresholds_failed(summary: dict[str, Any], args: argparse.Namespace) -> bool:
    checks = {
        "claim_accuracy": args.min_claim_accuracy,
        "numeric_accuracy": args.min_numeric_accuracy,
        "citation_precision": args.min_citation_precision,
        "citation_completeness": args.min_citation_completeness,
        "retrieval_recall_at_k": args.min_retrieval_recall,
    }
    return (
        any(
            value is not None
            and (summary.get(key) is None or summary[key] < value)
            for key, value in checks.items()
        )
        or summary["numeric_expected"] < 1
        or summary["model_results"] < 1
        or summary["refusal_results"] < 1
        or summary["retrieval_results"] < 1
        or summary["false_refusals"] > args.max_false_refusals
        or summary["refusal_correctness"] is None
        or summary["refusal_correctness"] < 1.0
        or summary["p95_latency_seconds"] is None
        or summary["p95_latency_seconds"] > args.max_p95_seconds
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Claim Ledger Evaluation",
        "",
        f"- Pack: `{report['pack_id']}`",
        f"- Pack cases: `{report['case_count']}`",
        f"- Evaluation target: `{report['evaluation_target_sha256']}`",
        f"- Scored lane results: `{summary['total']}`",
        f"- Lanes: `{', '.join(report['lanes'])}`",
        f"- Passed: `{summary['passed']}`",
        f"- Claim accuracy: `{summary['claim_accuracy']}`",
        f"- Citation completeness: `{summary['citation_completeness']}`",
        f"- Retrieval recall@k: `{summary['retrieval_recall_at_k']}`",
        f"- p95 latency: `{summary['p95_latency_seconds']}`",
        "",
        "| Lane | ID | Pass | Mode | Claim accuracy | Recall@k | Seconds |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        lines.append("| {lane} | {id} | {passed} | {mode} | {claims} | {recall} | {elapsed:.3f} |".format(lane=result["lane"], id=result["id"], passed="yes" if result["passed"] else "no", mode=result["answer_mode"], claims=result["claim_accuracy"], recall=result["retrieval_recall_at_k"], elapsed=result["elapsed_seconds"]))
    if report.get("baseline"):
        comparison = report["baseline"]
        lines.extend(["", "## Baseline", "", f"- Identity matches: `{comparison['identity_matches']}`", f"- Win/tie/loss: `{comparison['win']}/{comparison['tie']}/{comparison['loss']}`", f"- Regressed IDs: `{', '.join(comparison['regressed_ids']) or '-'}`", f"- Critical regressions: `{', '.join(comparison['critical_regressions']) or '-'}`"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a private claim-ledger pack.")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument(
        "--print-pack-sha256",
        action="store_true",
        help="Print the canonical reviewed-pack SHA-256 and exit.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--lane", choices=("frozen", "retrieval", "live", "all"), default="all")
    parser.add_argument("--top-k", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--json-output", type=Path, default=Path(__file__).resolve().parent / "benchmarks" / "claim_ledger_latest.json")
    parser.add_argument("--markdown-output", type=Path, default=Path(__file__).resolve().parent / "benchmarks" / "claim_ledger_latest.md")
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--min-claim-accuracy", type=float, default=0.75)
    parser.add_argument("--min-numeric-accuracy", type=float, default=0.75)
    parser.add_argument("--min-citation-precision", type=float, default=0.75)
    parser.add_argument("--min-citation-completeness", type=float, default=0.75)
    parser.add_argument("--min-retrieval-recall", type=float, default=0.75)
    parser.add_argument("--max-false-refusals", type=int, default=0)
    parser.add_argument("--max-p95-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        pack = load_pack(args.pack)
        if args.print_pack_sha256:
            print(pack_identity(pack))
            return 0
        lanes = ["frozen", "retrieval", "live"] if args.lane == "all" else [args.lane]
        results = evaluate_pack(
            pack,
            lanes,
            args.base_url,
            args.top_k,
            args.timeout_seconds,
            os.getenv("EVALUATION_API_TOKEN", ""),
        )
    except (PackValidationError, HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
        print("Claim-ledger evaluation failed.", file=sys.stderr)
        return 2
    targets = {
        target_identity(result["evaluation_input"]["evaluation_target"])
        for result in results
    }
    if len(targets) != 1:
        print("Claim-ledger evaluation failed.", file=sys.stderr)
        return 2
    report_target = results[0]["evaluation_input"]["evaluation_target"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "pack_id": pack["pack_id"],
        "pack_sha256": pack_identity(pack),
        "case_count": len(pack["cases"]),
        "lanes": lanes,
        "top_k": args.top_k,
        "question_identities": {case["id"]: question_identity(case["question"]) for case in pack["cases"]},
        "evaluation_target": report_target,
        "evaluation_target_sha256": target_identity(report_target),
        "results": results,
        "summary": summarize_results(results),
    }
    try:
        report = rescore_report(pack, report)
        if args.baseline_json:
            baseline = rescore_report(
                pack,
                json.loads(args.baseline_json.read_text(encoding="utf-8")),
            )
            report["baseline"] = compare_baseline(baseline, report)
    except (OSError, json.JSONDecodeError, PackValidationError, ValueError):
        print("Claim-ledger evaluation failed.", file=sys.stderr)
        return 2
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    if thresholds_failed(report["summary"], args):
        return 1
    comparison = report.get("baseline")
    if comparison and (
        not comparison["identity_matches"]
        or not comparison["result_set_matches"]
        or comparison["critical_regressions"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
