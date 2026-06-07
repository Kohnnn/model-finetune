# Qwen 3.5 Recovery Master Plan

## Problems to Fix

| # | Problem | Impact | Fix |
|---|---------|--------|-----|
| 1 | Dirty SFT data | Hallucinated completions in model | Use only reviewed clean subset or regenerate |
| 2 | Missing assistant-only loss | Prompt tokens learned as responses | Enable response-only masking |
| 3 | Stale GGUF provenance | Unknown model lineage | Export fresh from known-good HF artifact |
| 4 | App fallback masking | Ungrounded analyst-style answers | Detect no-evidence and respond accordingly |

## Current Progress — 2026-05-16

- Completed a v0.1 full-corpus draft Qwen 3.5 run at `finetune/outputs/qwen35_4b_full_corpus_draft23974/` using `23,974` rows and final `train_loss=1.0765`.
- Exported deployment artifacts are present at `deployment/models/Qwen3.5-4B.Q4_K_M.gguf` and `deployment/models/Qwen3.5-4B.BF16-mmproj.gguf`.
- Live benchmark exists at `deployment/benchmarks/latest_report.md`; `/healthz` is ok, but sample `/query` results returned fallback evidence snippets because model answers were not grounded/cited.
- App fallback behavior has been corrected from generic analyst answers to insufficient-evidence/fallback excerpts.
- Remaining recovery focus is data quality: create or review a clean SFT subset before another training run.

## Phased Recovery Plan

### Phase 1: Data Audit and Cleaning

1. **Stop using dirty dataset** — do not train on `outputs/datasets/vietcap_sft_generated.jsonl`
2. **Generate clean completions** — use current good Qwen model to re-generate completions on clean prompts
3. **Human review** — spot-check at least 10% of generated completions
4. **Export reviewed subset** — save as `outputs/datasets/qwen35_clean_sft.jsonl`

### Phase 2: Clean Training Run

```bash
python finetune/train.py \
  --dataset-path finetune/outputs/datasets/qwen35_clean_sft.jsonl \
  --output-dir finetune/outputs/qwen35_4b_clean_recovery \
  --max-seq-length 1024 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --num-epochs 1 \
  --eval-split 0 \
  --log-steps 100 \
  --save-steps 500 \
  --warmup-steps 100 \
  --save-merged-model \
  --skip-gguf-export
```

**Critical flag**: Do NOT use `--disable-response-only-masking` — this is what caused the prompt-loss issue.

### Phase 3: Fresh GGUF Export

```bash
python finetune/export_gguf.py \
  --model-path finetune/outputs/qwen35_4b_clean_recovery/merged_model \
  --output-dir finetune/outputs/qwen35_4b_clean_recovery \
  --gguf-name qwen3_5_4b_clean_recovery
```

### Phase 4: Provenance Verification

1. Compare new GGUF against Ollama model with same benchmark queries
2. Verify no degradation on known-good test cases
3. Document exact Hugging Face commit used as base

### Phase 5: App Integration and Fallback Fix

1. Copy new GGUF to `deployment/models/`
2. Update `deployment/docker-compose.yml` if needed
3. Fix fallback behavior in `deployment/app/main.py`:
   - If retrieved chunks are empty or low-confidence, return "I don\'t have evidence for that"
   - Do NOT fall back to general analyst persona

## Acceptance Gates

| Gate | Criteria |
|------|----------|
| Data clean | Zero hallucinated completions in spot-check of 10% |
| Training loss | Final loss < 1.2 (indicates learning) |
| Benchmark | Passes 3/5 held-out analyst questions with grounded answers |
| Provenance | GGUF exported from specific Hugging Face commit, documented |
| Fallback | App returns "no evidence" instead of generic answer when appropriate |

## What NOT To Do

- Do not merge the dirty and clean datasets
- Do not use `--disable-response-only-masking` again
- Do not treat the Ollama model as a trusted base for new exports
- Do not skip the human review step
