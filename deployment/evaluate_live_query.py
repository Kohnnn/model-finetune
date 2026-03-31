from __future__ import annotations

import argparse
import json
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
        sources = response.get("sources", []) or []
        hits = keyword_hits(answer, expected_keywords)

        results.append(
            {
                "id": item.get("id") or question[:40],
                "question": question,
                "elapsed_seconds": round(elapsed, 2),
                "context_used": response.get("context_used", 0),
                "source_count": len(sources),
                "expected_keywords": expected_keywords,
                "keyword_hits": hits,
                "answer": answer,
                "source_labels": [source.get("source_label") for source in sources[:3]],
            }
        )
    return results


def render_markdown(
    base_url: str, health: dict[str, Any], results: list[dict[str, Any]]
) -> str:
    lines = [
        "# Live Query Benchmark",
        "",
        f"- Base URL: `{base_url}`",
        f"- Status: `{health.get('status')}`",
        f"- Collection: `{health.get('collection_name')}`",
        f"- Embedding model: `{health.get('embedding_model_name')}`",
        f"- LLM model: `{health.get('llm_model_name')}`",
        "",
        "## Results",
        "",
        "| ID | Seconds | Context | Sources | Keyword hits |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for result in results:
        lines.append(
            "| {id} | {elapsed_seconds} | {context_used} | {source_count} | {hits} |".format(
                id=result["id"],
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
        report = render_markdown(args.base_url, health, results)

        if args.output_path is not None:
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_path.write_text(report, encoding="utf-8")
            print(f"Benchmark report written to {args.output_path}")
        else:
            sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(b"\n")

        return 0
    except (FileNotFoundError, RuntimeError, HTTPError, URLError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
