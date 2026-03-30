from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

DEPLOYMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOYMENT_DIR.parent
ENV_PATH = DEPLOYMENT_DIR / ".env"
COMPOSE_FILE = DEPLOYMENT_DIR / "docker-compose.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate local deployment settings and boot the RAG MVP stack."
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip the one-shot Chroma ingestion step.",
    )
    parser.add_argument(
        "--with-proxy",
        action="store_true",
        help="Start the optional nginx proxy profile after the app is healthy.",
    )
    parser.add_argument(
        "--health-timeout-seconds",
        type=int,
        default=120,
        help="How long to wait for the app health check to succeed.",
    )
    parser.add_argument(
        "--ingest-limit",
        type=int,
        default=None,
        help="Optional cap on ingested rows for smoke-test runs.",
    )
    parser.add_argument(
        "--ingest-batch-size",
        type=int,
        default=256,
        help="Batch size passed to the ingest container.",
    )
    return parser.parse_args()


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy deployment/.env.example to deployment/.env first."
        )

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_inputs(env_values: dict[str, str], with_proxy: bool) -> None:
    chroma_token = env_values.get("CHROMA_AUTH_TOKEN", "")
    require(
        chroma_token not in {"", "change-me", "replace-with-a-long-random-token"},
        "Set CHROMA_AUTH_TOKEN in deployment/.env to a real secret value.",
    )

    model_filename = env_values.get("LLAMA_MODEL_FILENAME", "Qwen3.5-4B.Q4_K_M.gguf")
    model_path = DEPLOYMENT_DIR / "models" / model_filename
    require(
        model_path.exists(),
        f"Model file not found: {model_path}. Place your GGUF model in deployment/models.",
    )

    mmproj_filename = env_values.get("LLAMA_MMPROJ_FILENAME", "").strip()
    require(
        bool(mmproj_filename),
        "Set LLAMA_MMPROJ_FILENAME in deployment/.env for the exported Qwen 3.5 companion file.",
    )
    mmproj_path = DEPLOYMENT_DIR / "models" / mmproj_filename
    require(
        mmproj_path.exists(),
        f"mmproj file not found: {mmproj_path}. Place the exported mmproj file in deployment/models.",
    )

    dataset_path = REPO_ROOT / "ocr_pipeline" / "chroma_chunks.jsonl"
    require(
        dataset_path.exists(),
        f"Missing dataset file: {dataset_path}. Run ocr_pipeline/process_pdfs.py first.",
    )

    if with_proxy:
        cert_path = DEPLOYMENT_DIR / "certs" / "cert.pem"
        key_path = DEPLOYMENT_DIR / "certs" / "key.pem"
        require(
            cert_path.exists() and key_path.exists(),
            "The proxy profile requires deployment/certs/cert.pem and deployment/certs/key.pem.",
        )


def run_compose(*args: str) -> None:
    command = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "--env-file",
        str(ENV_PATH),
        *args,
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_compose_with_retries(
    *args: str, retries: int = 5, delay_seconds: int = 5
) -> None:
    last_error: subprocess.CalledProcessError | None = None

    for attempt in range(1, retries + 1):
        try:
            run_compose(*args)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay_seconds)

    if last_error is not None:
        raise last_error


def wait_for_health(timeout_seconds: int) -> None:
    url = "http://localhost:8000/healthz"
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if payload.get("status") == "ok":
                    return
        except (OSError, URLError, ValueError):
            time.sleep(2)
            continue
        time.sleep(2)

    raise RuntimeError(
        f"App health check did not succeed within {timeout_seconds} seconds: {url}"
    )


def main() -> int:
    args = parse_args()

    try:
        env_values = read_env_file(ENV_PATH)
        validate_inputs(env_values, with_proxy=args.with_proxy)

        run_compose("up", "-d", "chromadb", "llama")

        if not args.skip_ingest:
            ingest_command = [
                "--profile",
                "ingest",
                "run",
                "--rm",
                "ingest",
                "python",
                "ingest.py",
                "--input-path",
                "/data/ocr_pipeline/chroma_chunks.jsonl",
                "--batch-size",
                str(args.ingest_batch_size),
            ]
            if args.ingest_limit is not None:
                ingest_command.extend(["--limit", str(args.ingest_limit)])

            run_compose_with_retries(
                *ingest_command,
                retries=6,
                delay_seconds=5,
            )

        run_compose("up", "-d", "app")
        wait_for_health(args.health_timeout_seconds)

        if args.with_proxy:
            run_compose("--profile", "proxy", "up", "-d", "nginx")

        print("RAG MVP stack is ready.")
        print("Health: http://localhost:8000/healthz")
        return 0
    except subprocess.CalledProcessError as exc:
        print(
            f"Command failed with exit code {exc.returncode}: {exc.cmd}",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except Exception as exc:  # noqa: PERF203
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
