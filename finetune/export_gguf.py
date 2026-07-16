from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an Unsloth model or adapter path to GGUF."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to this run's local merged_model folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Validated training run directory where the GGUF folder is written.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        required=True,
        help="run_manifest.json from the training run that produced --model-path.",
    )
    parser.add_argument(
        "--gguf-name",
        required=True,
        help="Base GGUF file name without extension.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Max sequence length for model loading.",
    )
    parser.add_argument(
        "--hub-token-env",
        default="HF_TOKEN",
        help="Environment variable that stores the Hugging Face token.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_hub_token(env_name: str) -> str | None:
    token = os.getenv(env_name)
    if token and token.strip():
        return token.strip()
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def gguf_hashes(gguf_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(gguf_dir).as_posix(): sha256_file(path)
        for path in sorted(gguf_dir.rglob("*.gguf"))
        if path.is_file()
    }


def gguf_state(gguf_dir: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(gguf_dir).as_posix(): (
            sha256_file(path),
            path.stat().st_mtime_ns,
        )
        for path in sorted(gguf_dir.rglob("*.gguf"))
        if path.is_file()
    }


def write_gguf_checksums(gguf_dir: Path, checksums: dict[str, str]) -> Path:
    if not checksums:
        raise RuntimeError(f"GGUF export produced no .gguf files under {gguf_dir}")
    checksum_path = gguf_dir / "SHA256SUMS"
    checksum_path.write_text(
        "\n".join(
            f"{digest}  {relative_path}"
            for relative_path, digest in sorted(checksums.items())
        )
        + "\n",
        encoding="utf-8",
    )
    return checksum_path


def validate_export_inputs(
    model_path: Path, output_dir: Path, run_manifest_path: Path
) -> tuple[Path, dict, str]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    expected_manifest = (output_dir / "run_manifest.json").resolve()
    if run_manifest_path.resolve() != expected_manifest or not expected_manifest.is_file():
        raise RuntimeError("--run-manifest must be output-dir/run_manifest.json.")
    try:
        source_model_path = model_path.resolve().relative_to(output_dir.resolve())
    except ValueError as exc:
        raise RuntimeError("--model-path must be inside --output-dir.") from exc
    if source_model_path.as_posix() != "merged_model":
        raise RuntimeError("--model-path must be output-dir/merged_model.")
    run_manifest = json.loads(expected_manifest.read_text(encoding="utf-8"))
    if not run_manifest.get("base_model_revision"):
        raise RuntimeError("run_manifest.json has no immutable base model revision.")
    return expected_manifest, run_manifest, hash_directory(model_path)


def ensure_windows_build_tools_on_path() -> None:
    if os.name != "nt":
        return

    candidate_paths = [
        r"C:\Program Files\CMake\bin",
        r"C:\Program Files\OpenSSL-Win64\bin",
        r"C:\Program Files\Git\mingw64\bin",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin",
    ]

    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    for candidate in candidate_paths:
        if os.path.isdir(candidate) and candidate not in path_entries:
            path_entries.append(candidate)

    os.environ["PATH"] = os.pathsep.join(path_entries)

    openssl_root = r"C:\Program Files\OpenSSL-Win64"
    if os.path.isdir(openssl_root):
        os.environ.setdefault("OPENSSL_ROOT_DIR", openssl_root)


def patch_unsloth_openssl_detection() -> None:
    if os.name != "nt":
        return

    candidate_roots = [
        Path(r"C:\ProgramData\openbb\Library"),
        Path(r"C:\ProgramData\openbb\envs\obb\Library"),
        Path(r"C:\Program Files\OpenSSL-Win64"),
    ]

    usable_root = None
    for root in candidate_roots:
        if (root / "include" / "openssl" / "ssl.h").exists():
            usable_root = root
            break

    if usable_root is None:
        return

    os.environ["OPENSSL_ROOT_DIR"] = str(usable_root)

    try:
        import unsloth_zoo.llama_cpp as llama_cpp_module

        llama_cpp_module._find_openssl_root = lambda: str(usable_root)
        llama_cpp_module.check_libcurl_dev = lambda: (True, "OpenSSL")

        original_check_build_requirements = llama_cpp_module.check_build_requirements

        def _patched_check_build_requirements():
            missing, system_type = original_check_build_requirements()
            if system_type == "windows":
                missing = [package for package in missing if package != "openssl"]
            return missing, system_type

        llama_cpp_module.check_build_requirements = _patched_check_build_requirements
    except Exception:
        return


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    expected_manifest, run_manifest, source_model_sha256 = validate_export_inputs(
        args.model_path, args.output_dir, args.run_manifest
    )
    source_model_path = args.model_path.resolve().relative_to(args.output_dir.resolve())

    ensure_windows_build_tools_on_path()
    patch_unsloth_openssl_detection()

    import torch
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for GGUF export with this workflow.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GGUF export requires the bf16-capable training environment.")

    hub_token = resolve_hub_token(args.hub_token_env)
    LOGGER.info("Loading model from %s", args.model_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.model_path),
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
        token=hub_token,
    )

    gguf_dir = args.output_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    before_state = gguf_state(gguf_dir)
    gguf_target = gguf_dir / args.gguf_name

    LOGGER.info("Exporting GGUF (q4_k_m) to %s", gguf_target)
    model.save_pretrained_gguf(
        str(gguf_target),
        tokenizer,
        quantization_method="q4_k_m",
    )
    after_state = gguf_state(gguf_dir)
    generated = {
        path: digest
        for path, (digest, modified_ns) in after_state.items()
        if before_state.get(path) != (digest, modified_ns)
    }
    model_files = [path for path in generated if "mmproj" not in Path(path).name.casefold()]
    mmproj_files = [path for path in generated if "mmproj" in Path(path).name.casefold()]
    if len(model_files) != 1 or len(mmproj_files) != 1:
        raise RuntimeError(
            "Export must produce exactly one model GGUF and one matching mmproj GGUF."
        )
    checksum_path = write_gguf_checksums(gguf_dir, generated)
    export_manifest = {
        "schema_version": 1,
        "run_manifest_sha256": sha256_file(expected_manifest),
        "base_model_revision": run_manifest.get("base_model_revision"),
        "source_model": source_model_path.as_posix(),
        "source_model_sha256": source_model_sha256,
        "gguf_files": generated,
    }
    (gguf_dir / "export_manifest.json").write_text(
        json.dumps(export_manifest, indent=2), encoding="utf-8"
    )
    LOGGER.info("GGUF export complete. Checksums: %s", checksum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
