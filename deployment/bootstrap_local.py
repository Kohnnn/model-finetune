from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        "--inference",
        choices=["localgguf", "ollama"],
        default="localgguf",
        help=(
            "Inference backend to start. 'localgguf' runs the llama.cpp server from a "
            "local GGUF file (default); 'ollama' uses a preinstalled Ollama model."
        ),
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sha256sums(path: Path) -> dict[str, str]:
    require(path.exists(), f"Checksum file not found: {path}")
    checksums: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split(maxsplit=1)
        if len(parts) == 2:
            checksums[Path(parts[1].lstrip("* ")).name] = parts[0]
    return checksums


def verify_model_checksum(model_path: Path, checksums: dict[str, str]) -> None:
    expected = checksums.get(model_path.name)
    require(expected is not None, f"No SHA-256 entry for {model_path.name}")
    require(
        sha256_file(model_path) == expected,
        f"SHA-256 mismatch for {model_path}",
    )


def validate_inputs(env_values: dict[str, str], with_proxy: bool, inference: str) -> None:
    chroma_token = env_values.get("CHROMA_AUTH_TOKEN", "")
    require(
        chroma_token not in {"", "change-me", "replace-with-a-long-random-token"},
        "Set CHROMA_AUTH_TOKEN in deployment/.env to a real secret value.",
    )

    if inference == "localgguf":
        # docker-compose 'llama-server' reads MODEL=/models/${LLM_MODEL:-...};
        # keep this default in sync with docker-compose.yml.
        model_filename = env_values.get(
            "LLM_MODEL", "Qwen3.5-4B.Clean-Recovery.Q4_K_M.gguf"
        ).strip()
        require(bool(model_filename), "Set LLM_MODEL to a GGUF file name.")
        model_path = DEPLOYMENT_DIR / "models" / model_filename
        require(
            model_path.is_file(),
            f"Model file not found: {model_path}. Place your GGUF model in deployment/models "
            "or set LLM_MODEL in deployment/.env.",
        )

        mmproj_filename = env_values.get(
            "LLAMA_MMPROJ_FILENAME", "Qwen3.5-4B.BF16-mmproj.gguf"
        ).strip()
        require(
            bool(mmproj_filename),
            "Set LLAMA_MMPROJ_FILENAME for the Qwen3.5 deployment bundle.",
        )
        mmproj_path = DEPLOYMENT_DIR / "models" / mmproj_filename
        require(
            mmproj_path.is_file(),
            f"mmproj file not found: {mmproj_path}. Place the matching exported mmproj file in deployment/models.",
        )
        checksums = read_sha256sums(DEPLOYMENT_DIR / "models" / "SHA256SUMS")
        verify_model_checksum(model_path, checksums)
        verify_model_checksum(mmproj_path, checksums)
    else:
        require(
            bool(env_values.get("OLLAMA_MODEL", "").strip()),
            "Set OLLAMA_MODEL in deployment/.env to a model already installed in Ollama.",
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


def run_compose(*args: str, extra_env: dict[str, str] | None = None) -> None:
    command = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "--env-file",
        str(ENV_PATH),
        *args,
    ]
    env = None
    if extra_env:
        env = {**os.environ, **extra_env}
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


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


def verify_ollama_model(
    model_name: str, retries: int = 12, delay_seconds: int = 5
) -> None:
    command = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "--env-file",
        str(ENV_PATH),
        "--profile",
        "ollama",
        "exec",
        "-T",
        "ollama",
        "ollama",
        "show",
        model_name,
    ]
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(retries):
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt + 1 < retries:
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
                if (
                    payload.get("status") == "ok"
                    and payload.get("collection_available") is True
                    and payload.get("inference_available") is True
                ):
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
        validate_inputs(env_values, with_proxy=args.with_proxy, inference=args.inference)

        # The inference services are profile-gated in docker-compose.yml:
        #   localgguf -> llama-server   |   ollama -> ollama
        inference_service = "llama-server" if args.inference == "localgguf" else "ollama"
        run_compose("up", "-d", "chromadb")
        run_compose("--profile", args.inference, "up", "-d", inference_service)
        if args.inference == "ollama":
            verify_ollama_model(env_values["OLLAMA_MODEL"])

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

        app_env = {
            "LLAMA_API_URL": (
                "http://llama-server:8080/v1"
                if args.inference == "localgguf"
                else "http://ollama:11434/v1"
            ),
            "LLM_MODEL_NAME": (
                env_values.get("LLM_MODEL_NAME", "qwen3.5-private-analyst")
                if args.inference == "localgguf"
                else env_values["OLLAMA_MODEL"]
            ),
        }
        run_compose("up", "-d", "app", extra_env=app_env)
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
