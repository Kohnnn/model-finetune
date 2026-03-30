from __future__ import annotations

import argparse
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
        help="Path to a local adapter/model folder loadable by Unsloth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the GGUF file should be written.",
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

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    ensure_windows_build_tools_on_path()
    patch_unsloth_openssl_detection()

    import torch
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for GGUF export with this workflow.")

    hub_token = resolve_hub_token(args.hub_token_env)
    LOGGER.info("Loading model from %s", args.model_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.model_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=hub_token,
    )

    gguf_dir = args.output_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_target = gguf_dir / args.gguf_name

    LOGGER.info("Exporting GGUF (q4_k_m) to %s", gguf_target)
    model.save_pretrained_gguf(
        str(gguf_target),
        tokenizer,
        quantization_method="q4_k_m",
    )
    LOGGER.info("GGUF export complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
