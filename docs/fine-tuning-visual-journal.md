# Private Analyst Fine-Tuning: A Visual Beginner Journal

This is the canonical learning path for this repository. It explains what each stage does, why it exists, what can go wrong, and which evidence is required before a private model can be released.

The historical run remains in [`DEVELOPMENT_JOURNAL.md`](../DEVELOPMENT_JOURNAL.md). Its draft-data result is useful engineering evidence, but it is not the release standard described here.

## What you will learn

By the end, you can:

1. explain the difference between retrieval-augmented generation (RAG) and fine-tuning;
2. turn private documents into traceable JSONL examples;
3. review examples instead of training on synthetic drafts blindly;
4. keep documents out of both training and evaluation at the same time;
5. train Qwen3.5-4B with bf16 LoRA and assistant-only loss;
6. compare the base model with the trained candidate;
7. export and hash a GGUF bundle;
8. publish only a gate-approved private Hugging Face release;
9. verify the exact artifact used by the local grounded RAG app.

## Safety rules

- Never commit `raw_dataset/`, generated JSONL, model weights, `.env`, tokens, Chroma data, or benchmark answers containing private excerpts.
- A row marked `draft` is not training data. A human must inspect and mark it `approved`.
- Fine-tuning teaches behavior and style. RAG supplies current facts and evidence.
- Training loss alone never proves model quality.
- A safe evidence fallback is good application behavior, but it does not count as a successful model answer.
- Upload only after `validate_release.py` passes. The upload helper always creates private repositories.

![The complete private-model lifecycle](assets/fine-tuning-journal/01_lifecycle_map.png)

*The lifecycle has evidence gates between stages. Files do not become a release merely because training completed.*

## 1. The mental model: RAG and fine-tuning solve different problems

RAG retrieves passages when a question arrives. Fine-tuning changes how the model responds. This stack uses both:

- **RAG:** Chroma retrieves source chunks with company, report, year, and chunk metadata.
- **Fine-tuning:** reviewed examples teach concise bilingual equity-research structure.
- **Grounding guard:** the app rejects answers without source citations.

![RAG compared with fine-tuning](assets/fine-tuning-journal/02_rag_vs_fine_tuning.png)

Use RAG when facts change, must be cited, or must remain outside model weights. Use fine-tuning when the desired improvement is tone, structure, instruction following, or a repeatable analytical pattern.

## 2. Terminal, repository, environment, and command

A **terminal** is a text interface for running programs. The **repository root** is `D:\finetune`; commands in this guide start there. A Python **virtual environment** isolates this project's packages from other Python projects.

![Terminal and environment basics](assets/fine-tuning-journal/03_terminal_and_environment.png)

Create the validated Windows GPU environment:

```powershell
./finetune/setup_gpu_env.ps1
```

Confirm Python and CUDA:

```powershell
.venv\Scripts\python.exe --version
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

Stop if CUDA or bf16 support is false. The production path intentionally does not fall back to a less reliable precision.

## 3. Parse documents into two outputs

The parser reads PDF, DOCX, and PPTX files, removes known boilerplate, chunks useful text, and writes:

- `ocr_pipeline/chroma_chunks.jsonl` for retrieval;
- `ocr_pipeline/finetune_template.jsonl` for supervised examples with empty assistant responses.

```powershell
.venv\Scripts\python.exe ocr_pipeline/process_pdfs.py `
  --input-dir raw_dataset `
  --output-dir ocr_pipeline `
  --extensions .pdf .docx .pptx
```

Each retained document carries content and file hashes, a stable `doc_id`, and a `document_family_id`. Exact normalized-content duplicates are removed; strict SimHash grouping keeps likely revisions in one split family. `parse_manifest.jsonl` records parsed, skipped, and failed files without changing the private output contract. Review `parse_failures.log` and extraction warnings, especially image-only PDF pages, before dataset preparation.

## 4. Understand one JSONL example

**JSONL** means one JSON object per line. It is streamable and easy to validate. A training row contains a conversation plus source metadata:

```json
{
  "messages": [
    {"role": "system", "content": "You are a grounded financial analyst."},
    {"role": "user", "content": "Context:\n...\n\nTask: Analyze the margin risk."},
    {"role": "assistant", "content": "The reviewed ideal response."}
  ],
  "metadata": {
    "doc_id": "stable_source_document_id",
    "document_family_id": "stable_revision_family_id",
    "source_file_sha256": "64_hex_characters",
    "context_sha256": "64_hex_characters",
    "source_spans": [{"start_page": 3, "end_page": 3}],
    "task_type": "risk_analysis",
    "language": "en",
    "review_status": "approved",
    "reviewed_by": "reviewer_id",
    "reviewed_at": "2026-07-16T10:00:00+00:00",
    "approval_checklist_version": "v1"
  }
}
```

![Anatomy of a JSONL training row](assets/fine-tuning-journal/04_jsonl_training_example.png)

The user context is evidence, the assistant content is the behavior target, and metadata controls traceability and release safety.

## 5. Generate drafts, then perform human review

The seed generator extracts candidate analytical text. It deliberately marks every row as `draft`:

```powershell
.venv\Scripts\python.exe finetune/prepare_seed_dataset.py `
  --input-path ocr_pipeline/finetune_template.jsonl `
  --output-path finetune/outputs/datasets/qwen35_review_queue.jsonl `
  --max-rows 512 `
  --max-context-words 450
```

For every row selected for training:

1. open the source context, recorded source span, and assistant completion;
2. replace heuristic draft prose with an original reviewed target;
3. verify every claim and number against the source; list deliberately verified external numbers in `verified_external_numbers`;
4. correct copied boilerplate, broken numbers, unsupported claims, and poor prose;
5. reject weak, ambiguous, or extraction-damaged examples;
6. complete `reviewed_by`, timezone-aware `reviewed_at`, and `approval_checklist_version`;
7. set `metadata.review_status` to `approved` only after inspection;
8. save approved rows to a separate private JSONL file.

![Draft, review, and approval gate](assets/fine-tuning-journal/05_draft_review_approval_gate.png)

Audit the private approved file before loading a model:

```powershell
.venv\Scripts\python.exe finetune/audit_dataset.py `
  --dataset-path finetune/outputs/datasets/qwen35_approved_sft.jsonl `
  --report-path finetune/outputs/datasets/qwen35_approved_sft.audit.json
```

Require zero errors. The audit checks message order, review fields, hashes, spans, source-supported numbers, duplicate rows, conflicting targets, shared shingles, and assistant-to-context copy ratio without printing row content. `train.py` reruns this audit and accepts only approved rows.

Validate formatting and splitting without loading a model:

```powershell
.venv\Scripts\python.exe finetune/train.py `
  --dry-run `
  --dataset-path finetune/outputs/datasets/qwen35_approved_sft.jsonl
```

For parser-only validation of the empty template, use the explicit non-training exception:

```powershell
.venv\Scripts\python.exe finetune/train.py --dry-run --allow-empty-assistant --max-samples 10
```

## 6. Split by document family, not row

Overlapping chunks, exact copies, and minor report revisions are similar. A row-level or document-only split can put one revision in training and another in evaluation, making the score look better than reality.

The pipeline chooses evaluation **document family IDs** deterministically, then puts every document and chunk from each chosen family into evaluation. `--max-samples` also selects complete families where possible.

![Document-level train and evaluation split](assets/fine-tuning-journal/06_document_level_split.png)

The run manifest records document and family lists with `split_strategy=document_family_id`. The release gate fails if either document IDs or family IDs overlap.

## 7. Chat templates and assistant-only loss

A chat template turns the message objects into model tokens. During supervised fine-tuning, only assistant tokens should contribute to the loss. Otherwise the model spends capacity learning to reproduce system instructions, user questions, and private context.

![Chat template and assistant-only loss](assets/fine-tuning-journal/07_chat_template_assistant_loss.png)

`train.py` requires Unsloth response-only masking. It stops instead of silently switching to full-sequence training.

## 8. Why bf16 LoRA fits this machine

The four-billion-parameter base model stays mostly frozen. LoRA adds small trainable matrices to attention and feed-forward projections. Current Unsloth guidance estimates about 10 GB VRAM for Qwen3.5-4B bf16 LoRA, which fits the target RTX 4060 Ti 16 GB.

![bf16 LoRA memory model](assets/fine-tuning-journal/08_bf16_lora_memory.png)

The production defaults are deliberately boring:

- bf16 base loading with `load_in_16bit=True`;
- batch size `1`;
- gradient accumulation `4`;
- LoRA rank `16`, alpha `16`;
- Unsloth gradient checkpointing;
- maximum sequence length `1024`;
- seed `3407`.

## 9. Run a small smoke training job

First use a small approved sample and skip expensive exports:

```powershell
.venv\Scripts\python.exe finetune/train.py `
  --dataset-path finetune/outputs/datasets/qwen35_approved_sft.jsonl `
  --output-dir finetune/outputs/qwen35_4b_approved_smoke `
  --max-samples 64 `
  --num-epochs 0.1 `
  --skip-gguf-export
```

A successful smoke run proves that loading, formatting, masking, document splitting, evaluation, checkpoints, and adapter saving work. It does not prove quality.

## 10. Train and resume safely

Commit the intended training code and confirm `git status --short` is empty. Production training refuses a dirty worktree. Then run the approved dataset:

```powershell
.venv\Scripts\python.exe finetune/train.py `
  --dataset-path finetune/outputs/datasets/qwen35_approved_sft.jsonl `
  --output-dir finetune/outputs/qwen35_4b_approved_v1 `
  --max-seq-length 1024 `
  --batch-size 1 `
  --gradient-accumulation 4 `
  --num-epochs 1 `
  --eval-split 0.1 `
  --save-steps 200 `
  --save-merged-model `
  --skip-gguf-export
```

If interrupted, rerun the same command with:

```powershell
--resume-from-checkpoint True
```

![Checkpoint and resume loop](assets/fine-tuning-journal/09_checkpoint_resume_loop.png)

The immutable record is `run_manifest.json`. It contains the base model commit, dataset hash, code commit, package versions, seed, split, precision, assistant-only setting, baseline loss, final loss, and training metrics.

## 11. Compare the base model and trained candidate

Use the same retrieval corpus and five-question benchmark for both models. First save a baseline:

```powershell
.venv\Scripts\python.exe deployment/evaluate_live_query.py `
  --label baseline `
  --json-output-path deployment/benchmarks/baseline.json `
  --output-path deployment/benchmarks/baseline.md
```

After serving the candidate:

```powershell
.venv\Scripts\python.exe deployment/evaluate_live_query.py `
  --label candidate `
  --baseline-json deployment/benchmarks/baseline.json `
  --json-output-path deployment/benchmarks/candidate.json `
  --output-path deployment/benchmarks/candidate.md
```

A grounded model answer must use `answer_mode=model`, cite only labels actually returned in `sources`, and meet keyword coverage. Evidence fallback is safe but fails model-quality and refusal questions. The unsupported Antarctica question must return `answer_mode=insufficient_evidence`.

The five cases are a fixed release smoke, not a statistically useful quality benchmark. Build a private 100–150-case claim-ledger pack from document families excluded from training. Start from `deployment/benchmarks/claim_ledger_template.json`; every case records expected claims, numbers, supporting `doc_id` values, prohibited terms, language, task type, criticality, and frozen evidence. Use JSON numbers for locale-neutral values and exact strings such as `"1,5%"` or `"1.500"` when locale-specific display matters. Keep the completed pack local and gitignored.

Use the checksum-verified `localgguf` backend and pin `LLAMA_CPP_IMAGE` as `repository@sha256:<64 hex>`. Bootstrap launches only Compose values validated from `deployment/.env`, and the running app independently verifies mounted GGUF/mmproj bytes against `SHA256SUMS`. Ingestion uses an immutable source snapshot and a shared exclusive lock; serving fails closed while that lock exists. Set distinct random values of at least 32 characters for `EVALUATION_API_TOKEN` and `EVALUATION_ATTESTATION_KEY` in `deployment/.env`. Export only `EVALUATION_API_TOKEN` while running the evaluator. Export only `EVALUATION_ATTESTATION_KEY` while running release validation. Empty values disable `/retrieve` and `/generate-with-evidence`; Compose exposes both the app and raw llama-server only on `127.0.0.1`, and nginx blocks evaluation routes and credentials. After human review freezes the pack, run `deployment/evaluate_claim_ledger.py --pack deployment/benchmarks/private_claim_ledger_v1.json --print-pack-sha256` and independently record the result as `EXPECTED_CLAIM_LEDGER_PACK_SHA256`. After approving the baseline, independently record its `evaluation_target_sha256` as `EXPECTED_BASELINE_EVALUATION_TARGET_SHA256`.

Serve the base model and run all three lanes:

```powershell
.venv\Scripts\python.exe deployment/evaluate_claim_ledger.py `
  --pack deployment/benchmarks/private_claim_ledger_v1.json `
  --lane all `
  --json-output deployment/benchmarks/claim_ledger_baseline.json `
  --markdown-output deployment/benchmarks/claim_ledger_baseline.md
```

Replace the model bundle, recreate the app container, then serve the trained candidate against the same Chroma collection and unchanged pack:

```powershell
.venv\Scripts\python.exe deployment/evaluate_claim_ledger.py `
  --pack deployment/benchmarks/private_claim_ledger_v1.json `
  --lane all `
  --baseline-json deployment/benchmarks/claim_ledger_baseline.json `
  --json-output deployment/benchmarks/claim_ledger_candidate.json `
  --markdown-output deployment/benchmarks/claim_ledger_candidate.md
```

The lanes isolate failure causes:

- `frozen`: generation against identical supplied evidence through `/generate-with-evidence`;
- `retrieval`: recall@k only through `/retrieve`;
- `live`: end-to-end RAG through `/query`.

The evaluator requires matching pack hashes, question identities, top-k, case/lane inventories, and shared corpus/index/app/runtime/generation-config/collection/embedding identities for paired comparison. The app signs endpoint, request identity, retrieval settings, evidence identity, answer, mode, cited document IDs, end-to-end server latency, and an evaluation target derived from actual mounted GGUF/mmproj/corpus bytes, actual Chroma vectors/content plus ingestion generation, running app code, the digest-pinned llama.cpp runtime/configuration, and generation settings. Local reports retain these attestations and scoring inputs but omit source excerpts and paths. Release validation takes the reviewed pack plus baseline and candidate reports, verifies every HMAC attestation and independent pin, requires the candidate model/mmproj hashes to match the export manifest, recomputes every result, aggregate, and paired comparison, and hashes all three inputs. It exits nonzero below claim, numeric, citation, retrieval, refusal, or latency thresholds, and on any critical regression. Citation support means the answer cited a source whose `doc_id` is listed for the claim; human review is still required for semantic entailment.

![Baseline, candidate, and release gates](assets/fine-tuning-journal/10_baseline_candidate_release_gates.png)

## 12. Export GGUF and hash every artifact

Export from the saved merged model. The exporter binds the GGUF and mmproj hashes to that merged-model directory and the exact training manifest:

```powershell
.venv\Scripts\python.exe finetune/export_gguf.py `
  --model-path finetune/outputs/qwen35_4b_approved_v1/merged_model `
  --output-dir finetune/outputs/qwen35_4b_approved_v1 `
  --run-manifest finetune/outputs/qwen35_4b_approved_v1/run_manifest.json `
  --gguf-name qwen3_5_4b_private_analyst_v1
```

The exporter writes `SHA256SUMS` beside the GGUF outputs. Qwen3.5 deployment requires both the Q4_K_M model and its matching bf16 mmproj companion.

![Artifact lineage and SHA-256 verification](assets/fine-tuning-journal/11_artifact_lineage_hashes.png)

Copy the two GGUF files and checksum file to `deployment/models/`. Bootstrap recalculates both hashes and refuses stale or corrupted files.

## 13. Enforce release gates

Run the gate after candidate evaluation and GGUF export:

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

The command requires:

- every eligible approved row and a zero-error dataset audit, with no sample truncation;
- disjoint train and evaluation documents and document families;
- assistant-only loss and bf16 LoRA;
- an immutable base model revision loaded by commit;
- committed training code with a clean worktree;
- train loss below `1.2`;
- final evaluation loss below baseline evaluation loss;
- at least four of five fixed smoke cases passing with a safe unsupported-question fallback;
- at least 100 claim-ledger cases, including model, refusal, retrieval, and numeric coverage, scored in frozen, retrieval, and live lanes;
- claim, numeric, citation precision/completeness, and retrieval recall@k each at least `0.75`;
- perfect refusal correctness, zero false refusals, p95 latency at most 180 seconds, matching baseline identity/inventory, and no critical regression;
- merged model, model GGUF, and mmproj files;
- SHA-256 verification.

On success it writes `release_manifest.json`, `SHA256SUMS`, a model card, a metadata-only dataset card, `run_summary.json`, `benchmark_summary.json`, and `claim_ledger_summary.json`. Raw benchmark answers, questions, per-case IDs, private evidence, paths, and document IDs remain local.

## 14. Publish privately and verify the Hub

Never put a token in a command, notebook, file, or log. Set `HF_TOKEN` in the user environment. Upload the validated run directory:

```powershell
.venv\Scripts\python.exe finetune/push_to_huggingface.py `
  --run-dir finetune/outputs/qwen35_4b_approved_v1 `
  --release-manifest finetune/outputs/qwen35_4b_approved_v1/release_manifest.json `
  --repo-id YOUR_ACCOUNT/private-analyst-qwen35-v1 `
  --dataset-card finetune/outputs/qwen35_4b_approved_v1/dataset_card.md `
  --dataset-repo-id YOUR_ACCOUNT/private-analyst-reviewed-sft-metadata
```

The helper verifies private visibility before upload, rejects unlisted remote files, uploads only the hashed inventory, and verifies remote SHA-256 bytes. The dataset repository contains metadata only, never corpus text, examples, raw benchmark answers, paths, or document IDs.

![Private Hub publication and verified GGUF deployment](assets/fine-tuning-journal/12_private_hub_gguf_deployment.png)

## 15. Start the verified local stack

Create `deployment/.env` from `.env.example`, set a real Chroma token, and ensure the three deployment files exist:

- the Q4_K_M GGUF;
- the matching bf16 mmproj GGUF;
- `SHA256SUMS` with entries for both filenames.

Then run:

```powershell
.venv\Scripts\python.exe deployment/bootstrap_local.py --ingest-limit 1024
```

`/healthz` reports `ok` only when inference responds and a real embedding query succeeds against a non-empty Chroma collection. Query:

```powershell
curl.exe -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"query":"What are the key margin risks for ACB?"}'
```

The response includes `answer_mode`:

- `model`: cited model answer;
- `evidence_fallback`: safe excerpts because model output failed grounding;
- `insufficient_evidence`: no usable evidence or an explicit model refusal because the retrieved context is insufficient.

## 16. Reading a failed gate

A failed release is useful evidence:

- **approved_data_only:** finish human review;
- **full_approved_dataset:** remove a truncating `--max-samples` cap;
- **committed_training_code:** commit the intended code, then train from a clean worktree;
- **approved_dataset_audit:** repair review, provenance, hash, span, duplicate, or numeric-support errors;
- **document_family_split:** repair missing or reused `document_family_id` metadata;
- **document_split:** repair missing or reused `doc_id` metadata;
- **assistant_only_loss:** fix the model chat template or Unsloth version;
- **eval_improvement:** revise data or hyperparameters; do not release;
- **benchmark:** inspect fixed-smoke modes, citations, and expected concepts;
- **fallback_safety:** restore the refusal path before any deployment;
- **claim_ledger_quality:** inspect claim, number, citation, retrieval, refusal, lane inventory, and latency metrics;
- **claim_ledger_comparison:** use the identical pack and result inventory, then fix critical regressions;
- **gguf_bundle:** export or copy the matching mmproj;
- **checksum mismatch:** recopy artifacts; never update a hash to hide unexplained drift.

## 17. Interactive practice

Open [`notebooks/private_analyst_fine_tuning_tutorial.ipynb`](../notebooks/private_analyst_fine_tuning_tutorial.ipynb). Its default cells use synthetic text and standard-library checks. Private corpus inspection, GPU training, Hub upload, and deployment are disabled until you explicitly change their opt-in flags.

## Completion checklist

- [ ] I can explain why RAG owns facts and fine-tuning owns behavior.
- [ ] Every training row is human-reviewed and marked `approved`.
- [ ] Every row has source hashes/spans, `metadata.doc_id`, and `metadata.document_family_id`.
- [ ] `audit_dataset.py` reports zero errors.
- [ ] Train and evaluation documents and families are disjoint.
- [ ] The run used bf16 LoRA and assistant-only loss.
- [ ] `run_manifest.json` names an immutable base revision and dataset hash.
- [ ] The 100–150-case frozen, retrieval, and live comparison meets thresholds with no critical regression.
- [ ] At least four of five fixed smoke cases pass and refusal is safe.
- [ ] GGUF and mmproj hashes verify.
- [ ] `validate_release.py` passes.
- [ ] Hugging Face model and metadata-only dataset repositories are private.
- [ ] Local bootstrap reports both collection and inference available.
