from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    default_questions = (
        Path(__file__).resolve().parent / "benchmarks" / "default_questions.json"
    )
    parser = argparse.ArgumentParser(
        description="Run a small benchmark set against the live /query endpoint."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the local analyst service.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=default_questions,
        help="Path to a JSON file with benchmark questions.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write a Markdown report.",
    )
    parser.add_argument(
        "--json-output-path",
        type=Path,
        default=None,
        help="Optional path to write machine-readable benchmark results.",
    )
    parser.add_argument(
        "--label",
        default="candidate",
        help="Run label written to reports, such as baseline or candidate.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional prior benchmark JSON used for baseline-versus-candidate comparison.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="HTTP timeout per query.",
    )
    return parser.parse_args()


def get_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(
    url: str, payload: dict[str, Any], timeout_seconds: int
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("Questions file must contain a JSON array.")
    return payload


def keyword_hits(answer: str, expected_keywords: list[str]) -> list[str]:
    lowered = answer.casefold()
    return [keyword for keyword in expected_keywords if keyword.casefold() in lowered]


def score_result(
    item: dict[str, Any],
    answer: str,
    answer_mode: str,
    hits: list[str],
    source_count: int,
) -> tuple[bool, str]:
    expected_mode = str(item.get("expected_mode", "model"))
    if expected_mode == "refusal":
        passed = answer_mode == "insufficient_evidence"
        return passed, "safe refusal" if passed else "expected safe refusal"

    minimum_hits = int(item.get("minimum_keyword_hits", 1))
    citations = [int(index) for index in re.findall(r"\[S(\d+)\]", answer)]
    has_valid_citation = bool(citations) and all(
        1 <= index <= source_count for index in citations
    )
    passed = answer_mode == "model" and has_valid_citation and len(hits) >= minimum_hits
    reason = (
        "grounded model answer"
        if passed
        else "requires cited model answer and expected keyword coverage"
    )
    return passed, reason


def evaluate_questions(
    *,
    base_url: str,
    questions: list[dict[str, Any]],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in questions:
        question = str(item["question"])
        expected_keywords = [str(x) for x in item.get("expected_keywords", [])]
        start = time.perf_counter()
        response = post_json(
            f"{base_url.rstrip('/')}/query",
            {"query": question},
            timeout_seconds=timeout_seconds,
        )
        elapsed = time.perf_counter() - start

        answer = str(response.get("answer", ""))
        answer_mode = str(response.get("answer_mode", "unknown"))
        sources = response.get("sources", []) or []
        hits = keyword_hits(answer, expected_keywords)
        passed, reason = score_result(
            item, answer, answer_mode, hits, source_count=len(sources)
        )

        results.append(
            {
                "id": item.get("id") or question[:40],
                "question": question,
                "expected_mode": item.get("expected_mode", "model"),
                "elapsed_seconds": round(elapsed, 2),
                "context_used": response.get("context_used", 0),
                "source_count": len(sources),
                "answer_mode": answer_mode,
                "expected_keywords": expected_keywords,
                "keyword_hits": hits,
                "passed": passed,
                "reason": reason,
                "answer": answer,
                "source_labels": [source.get("source_label") for source in sources[:3]],
            }
        )
    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    refusal_results = [
        result for result in results if result["expected_mode"] == "refusal"
    ]
    return {
        "passed": sum(bool(result["passed"]) for result in results),
        "total": len(results),
        "fallback_safe": bool(refusal_results)
        and all(bool(result["passed"]) for result in refusal_results),
        "model_answers": sum(result["answer_mode"] == "model" for result in results),
        "evidence_fallbacks": sum(
            result["answer_mode"] == "evidence_fallback" for result in results
        ),
    }


def compare_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    candidate_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    baseline_summary = baseline.get("summary") or {}
    baseline_results = baseline.get("results") or []
    candidate_results = candidate_results or []
    baseline_ids = {str(result.get("id")) for result in baseline_results}
    candidate_ids = {str(result.get("id")) for result in candidate_results}
    baseline_passed_ids = {
        str(result.get("id")) for result in baseline_results if result.get("passed") is True
    }
    candidate_passed_ids = {
        str(result.get("id")) for result in candidate_results if result.get("passed") is True
    }
    question_set_matches = bool(baseline_ids) and baseline_ids == candidate_ids
    regressed_ids = sorted(baseline_passed_ids - candidate_passed_ids)
    baseline_passed = int(baseline_summary.get("passed", 0))
    candidate_passed = int(candidate.get("passed", 0))
    return {
        "baseline_passed": baseline_passed,
        "candidate_passed": candidate_passed,
        "pass_delta": candidate_passed - baseline_passed,
        "question_set_matches": question_set_matches,
        "regressed_ids": regressed_ids,
        "regressed": candidate_passed < baseline_passed
        or bool(regressed_ids)
        or not question_set_matches,
    }


def render_markdown(
    base_url: str,
    label: str,
    health: dict[str, Any],
    results: list[dict[str, Any]],
    comparison: dict[str, Any] | None = None,
) -> str:
    summary = summarize_results(results)
    lines = [
        "# Live Query Benchmark",
        "",
        f"- Label: `{label}`",
        f"- Base URL: `{base_url}`",
        f"- Status: `{health.get('status')}`",
        f"- Collection: `{health.get('collection_name')}`",
        f"- Embedding model: `{health.get('embedding_model_name')}`",
        f"- LLM model: `{health.get('llm_model_name')}`",
        f"- Passed: `{summary['passed']}/{summary['total']}`",
        f"- Fallback safety: `{summary['fallback_safe']}`",
    ]
    if comparison is not None:
        lines.extend(
            [
                f"- Baseline passed: `{comparison['baseline_passed']}`",
                f"- Pass delta: `{comparison['pass_delta']}`",
                f"- Regressed: `{comparison['regressed']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| ID | Pass | Mode | Seconds | Context | Sources | Keyword hits |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )

    for result in results:
        lines.append(
            "| {id} | {passed} | {mode} | {elapsed_seconds} | {context_used} | {source_count} | {hits} |".format(
                id=result["id"],
                passed="yes" if result["passed"] else "no",
                mode=result["answer_mode"],
                elapsed_seconds=result["elapsed_seconds"],
                context_used=result["context_used"],
                source_count=result["source_count"],
                hits=", ".join(result["keyword_hits"]) or "-",
            )
        )

    lines.append("")
    for result in results:
        lines.extend(
            [
                f"### {result['id']}",
                "",
                f"- Question: `{result['question']}`",
                f"- Runtime: `{result['elapsed_seconds']}s`",
                f"- Passed: `{result['passed']}`",
                f"- Answer mode: `{result['answer_mode']}`",
                f"- Reason: `{result['reason']}`",
                f"- Context used: `{result['context_used']}`",
                f"- Sources: `{', '.join([x for x in result['source_labels'] if x]) or '-'}`",
                f"- Keyword hits: `{', '.join(result['keyword_hits']) or '-'}`",
                "",
                "```text",
                result["answer"],
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        health = get_json(f"{args.base_url.rstrip('/')}/healthz", args.timeout_seconds)
        questions = load_questions(args.questions_file)
        results = evaluate_questions(
            base_url=args.base_url,
            questions=questions,
            timeout_seconds=args.timeout_seconds,
        )
        summary = summarize_results(results)
        comparison = None
        if args.baseline_json is not None:
            baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
            comparison = compare_summaries(baseline, summary, results)
        report = render_markdown(
            args.base_url, args.label, health, results, comparison=comparison
        )
        payload = {
            "schema_version": 1,
            "label": args.label,
            "base_url": args.base_url,
            "health": health,
            "summary": summary,
            "comparison": comparison,
            "results": results,
        }

        if args.output_path is not None:
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_path.write_text(report, encoding="utf-8")
            print(f"Benchmark report written to {args.output_path}")
        else:
            sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(b"\n")

        if args.json_output_path is not None:
            args.json_output_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_output_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            print(f"Benchmark JSON written to {args.json_output_path}")

        passed_threshold = summary["passed"] >= 4 and summary["fallback_safe"]
        regressed = comparison is not None and comparison["regressed"]
        return 0 if passed_threshold and not regressed else 2
    except (FileNotFoundError, RuntimeError, HTTPError, URLError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
