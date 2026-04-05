# VietCap Qwen3.5-4B Fine-tuning Development Journal

**Project**: Private AI Analyst Stack (OCR -> Fine-tune -> RAG App -> OCI Deployment)
**Status**: Phase 2 — VietCap SFT Generation (IN PROGRESS)
**Last Updated**: 2026-04-04

---

## Reference Pipeline (Jackrong Opus Distillation)
Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF:
- Base: Qwen/Qwen3.5-27B, Framework: Unsloth LoRA SFT
- Training format: `<thinking> {reasoning} </thinking>\n{answer}`, `<|im_start|>` chat template
- Starting loss: 0.74356 -> Final: 0.23984, Context: 16,384 tokens

Datasets used: `nohurry/Opus-4.6-Reasoning-3000x-filtered` (2330 rows),
`Roman1111111/claude-opus-4.6-10000x` (9633 rows),
`TeichAI/claude-4.5-opus-high-reasoning-250x` (250 rows),
`Jackrong/Qwen3.5-reasoning-700x` (633 rows)

---

## Two-Phase Pipeline Overview

```
Phase 1: Opus reasoning fine-tune
    Dataset: outputs/datasets/opus_reasoning.jsonl (12,688 rows)
    Base: Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled
    Output: finetune/outputs/qwen35_4b_opus_phase1/merged_model/
    Status: ✅ DONE

Phase 2: VietCap domain fine-tune
    Step 1: OCR pipeline -> finetune_template.jsonl (6,972 rows)
    Step 2: SFT generation -> vietcap_sft_generated.jsonl
    Step 3: Training -> finetune/outputs/qwen35_4b_opus_phase2/merged_model/
    Step 4: GGUF export -> Ollama serving
    Status: 🚧 IN PROGRESS
```

---

## Phase 1 — Opus Reasoning Fine-tune ✅

**Command:**
```bash
.venv311\Scripts\python finetune/train.py ^
    --dataset-path outputs/datasets/opus_reasoning.jsonl ^
    --model-name Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled ^
    --skip-gguf-export
```

**Config:** lr=5e-5, weight_decay=0.05, num_epochs=1.0, lora_r=16, lora_alpha=32

**Result:**
- 12,053 train rows, 635 eval rows
- Best checkpoint: step 400 (eval_loss: 0.7343)
- Final train_loss: 0.048 (low — possible overfitting on small dataset)
- Runtime: ~75 min
- Best model: `finetune/outputs/checkpoints/checkpoint-400/`
- Merged model: `finetune/outputs/qwen35_4b_opus_phase1/merged_model/`

**Datasets downloaded by `download_opus_datasets.py`:**
- `nohurry/Opus-4.6-Reasoning-3000x-filtered` (2,330 rows)
- `Roman1111111/claude-opus-4.6-10000x` (9,633 rows) — streaming mode
- `TeichAI/claude-4.5-opus-high-reasoning-250x` (250 rows)
- `Jackrong/Qwen3.5-reasoning-700x` (633 rows)

---

## Phase 2 — VietCap Domain Fine-tune 🚧

### Step 2a: OCR Pipeline — Parse documents into chunks

**Command:**
```bash
.venv311\Scripts\python ocr_pipeline/process_pdfs.py
```

**Input:** `raw_dataset/` (~8,180 files: PDFs, DOCX, PPTX)
**Output:**
- `ocr_pipeline/finetune_template.jsonl` (6,972 rows — SFT template with empty assistant)
- `ocr_pipeline/chroma_chunks.jsonl` (6,972 rows — RAG chunks)

**Pipeline:**
1. Extract text per file type (PyMuPDF for PDF, python-docx for DOCX, python-pptx for PPTX)
2. Trim trailing pages (boilerplate disclaimers, VCSC rating system, contact pages)
3. Strip head boilerplate (disclaimers, VietCap headers)
4. Chunk into 800-word segments, 100-word overlap, min 200 words
5. Quality filter: require ≥ 2 analytical markers per chunk

**Language detection:** Vietnamese chars pattern (`ăâđêôơưáàảãạ...`) + filename `[vn]`/`vietnamese`

**Known Vietnamese boilerplate stripped:**
- `xác nhận của chuyên viên phân tích`
- `phương pháp định giá và hệ thống khuyến nghị của vcsc`
- `phòng giao dịch chứng khoán`, `báo cáo này được viết và phát hành bởi`
- `công ty cổ phần chứng khoán bản việt`

**⚠️ Critical:** Running with `--limit N` overwrites output files. Full run required.

**Bug fixed:** `process_pdfs.py` had literal `\n` strings instead of actual newlines in content strings.

---

### Step 2b: SFT Generation — Generate assistant responses

**Why not OpenAI API:**
- `qwen3.5:9b` via `/v1/chat/completions` → empty `content` field
- Thinking output goes in `reasoning` field, consumes all tokens
- **Fix:** Use Ollama CLI (`ollama run`) instead — answer appears after `...done thinking.` marker

**Generation speed:**
- Short prompts (~100 words context): ~64s/row
- Full prompts (~600 words context): ~120s/row
- Average: ~2 min/row
- Full 6,972 rows: ~230 hours (too long → batched into 6 phases)

**Script features added:**
- `--start` / `--end`: process specific slice of shuffled dataset
- `--resume`: skip rows already in output (idempotent restart)
- `--timeout`: configurable per-row timeout (default 600s)
- `--max-rows 6972`: always load full shuffled dataset for consistent index ordering
- Fixed seed=3407 for reproducible shuffling

**⚠️ Warning:** First row of each batch is slowest (Ollama model loading). Factor this into timeout.

---

### Phase 2b — Batch Commands (6 phases × 250 rows = 1,500 rows)

**Total coverage:** 1,500 / 6,972 rows (~22% of dataset — representative sample)
**Time per batch:** ~8 hours at 250 rows × 2 min/row
**Total generation time:** ~50 hours across 6 overnight runs

```bash
# ============================================================
# PHASE 1 — rows 0-250 (~8 hrs, run first)
# ============================================================
.venv311\Scripts\python finetune/generate_sft_dataset.py ^
    --resume ^
    --max-rows 6972 ^
    --start 0 --end 250 ^
    --model qwen3.5:9b ^
    --output outputs/datasets/vietcap_sft_generated.jsonl ^
    --timeout 600 ^
    --max-context-words 600 ^
    --max-response-tokens 400

# ============================================================
# PHASE 2 — rows 250-500 (~8 hrs)
# ============================================================
.venv311\Scripts\python finetune/generate_sft_dataset.py ^
    --resume ^
    --max-rows 6972 ^
    --start 250 --end 500 ^
    --model qwen3.5:9b ^
    --output outputs/datasets/vietcap_sft_generated.jsonl ^
    --timeout 600 ^
    --max-context-words 600 ^
    --max-response-tokens 400

# ============================================================
# PHASE 3 — rows 500-750 (~8 hrs)
# ============================================================
.venv311\Scripts\python finetune/generate_sft_dataset.py ^
    --resume ^
    --max-rows 6972 ^
    --start 500 --end 750 ^
    --model qwen3.5:9b ^
    --output outputs/datasets/vietcap_sft_generated.jsonl ^
    --timeout 600 ^
    --max-context-words 600 ^
    --max-response-tokens 400

# ============================================================
# PHASE 4 — rows 750-1000 (~8 hrs)
# ============================================================
.venv311\Scripts\python finetune/generate_sft_dataset.py ^
    --resume ^
    --max-rows 6972 ^
    --start 750 --end 1000 ^
    --model qwen3.5:9b ^
    --output outputs/datasets/vietcap_sft_generated.jsonl ^
    --timeout 600 ^
    --max-context-words 600 ^
    --max-response-tokens 400

# ============================================================
# PHASE 5 — rows 1000-1250 (~8 hrs)
# ============================================================
.venv311\Scripts\python finetune/generate_sft_dataset.py ^
    --resume ^
    --max-rows 6972 ^
    --start 1000 --end 1250 ^
    --model qwen3.5:9b ^
    --output outputs/datasets/vietcap_sft_generated.jsonl ^
    --timeout 600 ^
    --max-context-words 600 ^
    --max-response-tokens 400

# ============================================================
# PHASE 6 — rows 1250-1500 (~8 hrs)
# ============================================================
.venv311\Scripts\python finetune/generate_sft_dataset.py ^
    --resume ^
    --max-rows 6972 ^
    --start 1250 --end 1500 ^
    --model qwen3.5:9b ^
    --output outputs/datasets/vietcap_sft_generated.jsonl ^
    --timeout 600 ^
    --max-context-words 600 ^
    --max-response-tokens 400
```

**Resume:** If any batch crashes or is interrupted, re-run the **same command**. Already-completed rows are matched by `doc_id_chunk_index` and skipped.

---

### Step 2c: Phase 2 Training

**Run after all 6 SFT phases complete:**

```bash
.venv311\Scripts\python finetune/train.py ^
    --dataset-path outputs/datasets/vietcap_sft_generated.jsonl ^
    --model-name finetune/outputs/qwen35_4b_opus_phase1/merged_model ^
    --skip-gguf-export
```

**Training config (RTX 4060 Ti 16GB — verified working):**
- batch_size=4, gradient_accumulation=2 (total=8)
- max_seq_length=1024
- learning_rate=5e-5, weight_decay=0.05, num_epochs=1.0
- lora_r=16, lora_alpha=32
- `--max-memory-ratio=0.85` (warns if <15% VRAM free)
- **VRAM usage:** ~73% free (11.7 GiB / 16 GiB) — stable, no OOM

**Pilot run (121 rows):**
- train=114, eval=7, steps=15
- Runtime: ~7 min (batch_size=4, grad_accum=2)
- Train loss: 2.77

---

### Step 2d: GGUF Export

```bash
.venv311\Scripts\python finetune/train.py ^
    --dataset-path outputs/datasets/vietcap_sft_generated.jsonl ^
    --model-name finetune/outputs/qwen35_4b_opus_phase1/merged_model ^
    --save-merged-model
```

---

## Ollama Serving

**Phase 1 model (reasoning only):**
- Pushed to `hf.co/Mikkkkoooo/qwen35-4b-private-analyst-full-corpus:Q4_K_M`
- **Issue:** Only echoes system prompt (trained on empty assistants — no actual responses)
- Root cause: Phase 1 used empty assistant responses; model learned to ignore them

**Phase 2 model (final):**
- Will be exported as GGUF after Phase 2 training completes
- Expected: Actually useful VietCap analyst responses

---

## Key Issues & Fixes Log

| Date | Issue | Fix |
|------|-------|-----|
| 2026-03-31 | Literal `\n` in process_pdfs.py string literals | Fixed to actual newlines |
| 2026-03-31 | generate_sft_dataset.py never wrote to output file | Added `out_fh.write()` call |
| 2026-03-31 | Malformed JSON crashed dataset loading | Added try/except around json.loads |
| 2026-04-04 | Ollama API returns empty for qwen3.5:9b | Switch to CLI (`ollama run`), extract after `...done thinking.` |
| 2026-04-04 | SFT generation too slow (230 hrs) | Split into 6 resumable batch phases |
| 2026-04-04 | 300s timeout too short | Increased to 600s, added `--timeout` CLI arg |
| 2026-04-04 | finetune_template.jsonl overwritten by --limit 1 | Re-ran full process_pdfs.py (6,972 rows) |
| 2026-04-04 | Train OOM risk on RTX 4060 Ti | batch_size=4, grad_accum=2 with memory guard logging |
| 2026-04-04 | Vietnamese OCR not handled | Added VIETNAMESE_CHAR_PATTERN, Vietnamese boilerplate/analytical markers |
