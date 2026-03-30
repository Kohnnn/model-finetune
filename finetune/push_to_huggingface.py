from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local model folder to the Hugging Face Hub."
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="Folder to upload."
    )
    parser.add_argument(
        "--repo-id", required=True, help="Target Hub repo, e.g. user/model-name."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private if it does not exist.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload fine-tuned model",
        help="Commit message for the upload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN before uploading to Hugging Face.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(args.model_dir),
        commit_message=args.commit_message,
    )
    print(f"Uploaded {args.model_dir} -> {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
