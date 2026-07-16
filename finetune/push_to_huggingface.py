from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_artifacts(release_manifest: dict) -> dict[str, str]:
    artifacts = release_manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("Release manifest contains no artifact checksums.")
    return {str(path): str(digest) for path, digest in artifacts.items()}


def resolve_release_path(run_dir: Path, relative_path: str) -> Path:
    path = (run_dir / relative_path).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Release artifact escapes run directory: {relative_path}") from exc
    return path


def read_sha256sums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            checksums[parts[1].lstrip("* ").replace("\\", "/")] = parts[0]
    return checksums


def verify_local_release(run_dir: Path, release_manifest: dict) -> None:
    run_manifest = run_dir / "run_manifest.json"
    expected_run_manifest = release_manifest.get("run_manifest_sha256")
    if not run_manifest.is_file() or sha256_file(run_manifest) != expected_run_manifest:
        raise RuntimeError("run_manifest.json does not match the validated release.")
    artifacts = release_artifacts(release_manifest)
    for relative_path, expected in artifacts.items():
        path = resolve_release_path(run_dir, relative_path)
        if not path.is_file():
            raise RuntimeError(f"Release artifact is missing: {relative_path}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"Release artifact checksum mismatch: {path}")
    checksum_path = run_dir / "SHA256SUMS"
    if not checksum_path.is_file() or read_sha256sums(checksum_path) != artifacts:
        raise RuntimeError("SHA256SUMS does not match the release manifest inventory.")


def publish_paths(release_manifest: dict) -> set[str]:
    return {*release_artifacts(release_manifest), "release_manifest.json", "SHA256SUMS"}


def verify_remote_inventory(
    remote_names: set[str], expected_names: set[str], *, require_all: bool = True
) -> None:
    extras = sorted(remote_names - expected_names - {".gitattributes"})
    if extras:
        raise RuntimeError(f"Hub repository contains unhashed files: {extras}")
    if require_all:
        missing = sorted(expected_names - remote_names)
        if missing:
            raise RuntimeError(f"Hub verification failed; missing files: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local model folder to the Hugging Face Hub."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Validated training run directory containing merged_model and release metadata.",
    )
    parser.add_argument(
        "--repo-id", required=True, help="Target Hub repo, e.g. user/model-name."
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        required=True,
        help="Passed release_manifest.json produced by validate_release.py.",
    )
    parser.add_argument(
        "--dataset-card",
        type=Path,
        default=None,
        help="Optional metadata-only dataset card to upload to a private dataset repo.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Private dataset repository for the metadata-only dataset card.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload validated private model release",
        help="Commit message for the upload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
    model_dir = args.run_dir / "merged_model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Merged model directory not found: {model_dir}")
    expected_manifest_path = (args.run_dir / "release_manifest.json").resolve()
    if args.release_manifest.resolve() != expected_manifest_path:
        raise RuntimeError("--release-manifest must be run-dir/release_manifest.json.")
    if not expected_manifest_path.is_file():
        raise FileNotFoundError(f"Release manifest not found: {expected_manifest_path}")
    release_manifest = json.loads(expected_manifest_path.read_text(encoding="utf-8"))
    if release_manifest.get("passed") is not True:
        raise RuntimeError("Release manifest has not passed all gates.")
    if release_manifest.get("visibility") != "private":
        raise RuntimeError("Release manifest must require private visibility.")
    verify_local_release(args.run_dir, release_manifest)
    if bool(args.dataset_card) != bool(args.dataset_repo_id):
        raise RuntimeError("Pass both --dataset-card and --dataset-repo-id, or neither.")
    if args.dataset_card and args.dataset_card.resolve() != (
        args.run_dir / "dataset_card.md"
    ).resolve():
        raise RuntimeError("--dataset-card must be run-dir/dataset_card.md.")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN before uploading to Hugging Face.")

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=True, exist_ok=True)
    if api.model_info(args.repo_id).private is not True:
        raise RuntimeError("Hugging Face repository is not private; upload aborted.")
    expected_names = publish_paths(release_manifest)
    verify_remote_inventory(
        set(api.list_repo_files(args.repo_id)), expected_names, require_all=False
    )
    commit = api.create_commit(
        repo_id=args.repo_id,
        operations=[
            CommitOperationAdd(
                path_in_repo=relative_path,
                path_or_fileobj=str(resolve_release_path(args.run_dir, relative_path)),
            )
            for relative_path in sorted(expected_names)
        ],
        commit_message=args.commit_message,
    )
    info = api.model_info(args.repo_id, revision=commit.oid, files_metadata=True)
    if info.private is not True:
        raise RuntimeError("Uploaded Hugging Face repository is not private.")
    siblings = {
        getattr(item, "rfilename", str(item)): item for item in (info.siblings or [])
    }
    verify_remote_inventory(set(siblings), expected_names)
    expected_hashes = release_artifacts(release_manifest)
    expected_hashes.update(
        {
            name: sha256_file(resolve_release_path(args.run_dir, name))
            for name in ["release_manifest.json", "SHA256SUMS"]
        }
    )
    for relative_path, expected in expected_hashes.items():
        remote_file = siblings[relative_path]
        lfs = getattr(remote_file, "lfs", None)
        remote_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        if remote_sha256 is None:
            downloaded = hf_hub_download(
                repo_id=args.repo_id,
                filename=relative_path,
                revision=commit.oid,
                token=token,
            )
            remote_sha256 = sha256_file(Path(downloaded))
        if remote_sha256 != expected:
            raise RuntimeError(f"Hub checksum mismatch: {relative_path}")

    if args.dataset_card and args.dataset_repo_id:
        if not args.dataset_card.exists():
            raise FileNotFoundError(f"Dataset card not found: {args.dataset_card}")
        api.create_repo(
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        if api.dataset_info(args.dataset_repo_id).private is not True:
            raise RuntimeError("Hugging Face dataset repository is not private; upload aborted.")
        verify_remote_inventory(
            set(api.list_repo_files(args.dataset_repo_id, repo_type="dataset")),
            {"README.md"},
            require_all=False,
        )
        dataset_commit = api.upload_file(
            path_or_fileobj=str(args.dataset_card),
            path_in_repo="README.md",
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
            commit_message="Upload private metadata-only dataset card",
        )
        dataset_info = api.dataset_info(
            args.dataset_repo_id,
            revision=dataset_commit.oid,
            files_metadata=True,
        )
        if dataset_info.private is not True:
            raise RuntimeError("Uploaded Hugging Face dataset repository is not private.")
        dataset_names = {
            getattr(item, "rfilename", str(item))
            for item in (dataset_info.siblings or [])
        }
        verify_remote_inventory(dataset_names, {"README.md"})
        downloaded_card = hf_hub_download(
            repo_id=args.dataset_repo_id,
            filename="README.md",
            repo_type="dataset",
            revision=dataset_commit.oid,
            token=token,
        )
        if sha256_file(Path(downloaded_card)) != sha256_file(args.dataset_card):
            raise RuntimeError("Hub checksum mismatch: dataset README.md")

    print(f"Uploaded and verified private release {getattr(commit, 'oid', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
