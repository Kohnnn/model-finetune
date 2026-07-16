# Fine-Tuning Pipeline

This folder contains the Windows GPU Qwen3.5-4B training, export, release validation, and private Hugging Face publication path.

## Canonical workflow

Follow [`../docs/fine-tuning-visual-journal.md`](../docs/fine-tuning-visual-journal.md). It is the maintained source for commands and beginner explanations.

## Environment

```powershell
./finetune/setup_gpu_env.ps1
```

Target runtime:

- Python 3.11
- RTX 4060 Ti 16 GB or better
- CUDA PyTorch from the `cu128` wheel index
- bf16 support

## Data contract

Production rows must contain:

- ordered chat-style `messages` with a non-empty reviewed assistant response;
- source file and context hashes plus valid source spans;
- `metadata.doc_id` and `metadata.document_family_id`;
- task type, `en` or `vi` language, reviewer identity, timezone-aware review timestamp, and checklist version;
- `metadata.review_status` equal to `approved` after human review.

`prepare_seed_dataset.py` creates heuristic review candidates marked `draft`. The trainer does not accept them. Audit the final private file first:

```powershell
.venv\Scripts\python.exe finetune/audit_dataset.py `
  --dataset-path finetune/outputs/datasets/qwen35_approved_sft.jsonl `
  --report-path finetune/outputs/datasets/qwen35_approved_sft.audit.json
```

Safe parser-template validation:

```powershell
.venv\Scripts\python.exe finetune/train.py --dry-run --allow-empty-assistant --max-samples 10
```

Approved-data validation:

```powershell
.venv\Scripts\python.exe finetune/train.py `
  --dry-run `
  --dataset-path finetune/outputs/datasets/qwen35_approved_sft.jsonl
```

## Training contract

`train.py` enforces:

- a clean committed worktree for production runs;
- deterministic seed and document-family train/eval split;
- a zero-error approved-dataset audit;
- complete-family sampling and full-target fit without truncation;
- Qwen3.5-4B bf16 LoRA loading;
- assistant-only loss masking;
- baseline and final evaluation loss;
- checkpoint saving and `--resume-from-checkpoint True`;
- immutable base-model revision and dataset SHA-256 in `run_manifest.json`.

Smoke first:

```powershell
.venv\Scripts\python.exe finetune/train.py `
  --dataset-path finetune/outputs/datasets/qwen35_approved_sft.jsonl `
  --output-dir finetune/outputs/qwen35_4b_approved_smoke `
  --max-samples 64 `
  --num-epochs 0.1 `
  --skip-gguf-export
```

## GGUF export

```powershell
.venv\Scripts\python.exe finetune/export_gguf.py `
  --model-path finetune/outputs/qwen35_4b_approved_v1/merged_model `
  --output-dir finetune/outputs/qwen35_4b_approved_v1 `
  --run-manifest finetune/outputs/qwen35_4b_approved_v1/run_manifest.json `
  --gguf-name qwen3_5_4b_private_analyst_v1
```

The exporter writes SHA-256 checksums. Keep the Q4_K_M model and matching bf16 mmproj together.

## Release validation

Freeze and human-review the private pack, then compute its canonical JSON hash with `deployment/evaluate_claim_ledger.py --pack <path> --print-pack-sha256`. Independently approve the baseline report's `evaluation_target_sha256`. Set both pins before validation:

```powershell
$env:EVALUATION_ATTESTATION_KEY = "<independently managed attestation key>"
$env:EXPECTED_CLAIM_LEDGER_PACK_SHA256 = "<reviewed canonical pack SHA-256>"
$env:EXPECTED_BASELINE_EVALUATION_TARGET_SHA256 = "<approved baseline evaluation_target_sha256>"
.venv\Scripts\python.exe finetune/validate_release.py `
  --run-dir finetune/outputs/qwen35_4b_approved_v1 `
  --benchmark-json deployment/benchmarks/candidate.json `
  --claim-ledger-pack deployment/benchmarks/private_claim_ledger_v1.json `
  --claim-ledger-baseline-json deployment/benchmarks/claim_ledger_baseline.json `
  --claim-ledger-json deployment/benchmarks/claim_ledger_candidate.json
```

A passed release includes `release_manifest.json`, `SHA256SUMS`, a model card, a metadata-only dataset card, and sanitized run, fixed-smoke, and claim-ledger summaries. Validation requires at least 100 cases across frozen, retrieval, and live lanes against an identical baseline pack. Raw questions, answers, evidence, case IDs, local paths, and document IDs are not uploaded.

## Private Hugging Face upload

```powershell
.venv\Scripts\python.exe finetune/push_to_huggingface.py `
  --run-dir finetune/outputs/qwen35_4b_approved_v1 `
  --release-manifest finetune/outputs/qwen35_4b_approved_v1/release_manifest.json `
  --repo-id YOUR_ACCOUNT/private-analyst-qwen35-v1
```

The helper requires `HF_TOKEN`, verifies private visibility before upload, publishes only the exact hashed inventory, and verifies remote bytes. Never commit the token, corpus, generated dataset, model weights, or release benchmark answers containing private excerpts.

## Historical evidence

`QWEN35_TRAINING_NOTES.md` records the earlier draft-data run and troubleshooting. Its full-sequence training command is historical, not the current release path.
