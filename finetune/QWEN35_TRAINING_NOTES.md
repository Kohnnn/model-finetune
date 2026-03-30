# Qwen 3.5 Fine-Tuning Notes

This document records the troubleshooting and final execution path used to fine-tune `unsloth/Qwen3.5-4B` on the local RTX 4060 Ti 16GB machine and publish the final merged model to a private Hugging Face repo.

## Goal

Train a finance-style LoRA adapter from the local `raw_dataset/` corpus, keep the workflow reproducible on Windows, and produce a private merged Hugging Face artifact from the full cleaned corpus.

## Problems Encountered

1. The initial Python environment was CPU-only, so `torch.cuda.is_available()` was false even though the machine had an RTX 4060 Ti.
2. The original `transformers` version did not recognize the `qwen3_5` architecture.
3. The installed Unsloth version and dependency set were not aligned with the latest Qwen 3.5 stack.
4. The parser still emitted many disclaimer and contact chunks, which polluted both RAG data and SFT templates.
5. `unsloth/Qwen3.5-4B` loads through a processor-backed stack, so the training script could not assume a plain tokenizer.
6. The training backend had version drift across `TrainingArguments` and `Trainer` signatures on the installed stack.
7. Response-only masking was not reliable in the processor-backed Qwen 3.5 path, so the final local run used full-sequence training.
8. The Hugging Face upload had to be done with the merged model folder directly because the earlier helper invocation did not inherit the user-scoped token into the current process.

## Fixes Applied

### Environment

- Stored `HF_TOKEN` as a user-scoped environment variable.
- Upgraded the local `.venv` to CUDA PyTorch `2.11.0+cu128`.
- Upgraded `transformers` to `5.3.0` and refreshed `unsloth` / `unsloth-zoo` from source.
- Added `finetune/setup_gpu_env.ps1` so the GPU environment can be recreated quickly.

### Parser cleanup

- Expanded boilerplate detection in `ocr_pipeline/process_pdfs.py`.
- Removed disclaimer and contact sections before chunking.
- Filtered residual disclaimer chunks after chunking.
- Added metadata to `finetune_template.jsonl` rows so generated training examples are traceable to their source chunk.

### Training pipeline

- Changed the default base model in `finetune/train.py` to `unsloth/Qwen3.5-4B`.
- Added CUDA and model-config preflight checks.
- Added Hugging Face push options and merged-model saving support.
- Added a tokenizer/processor split so Qwen 3.5 can train with a real tokenizer backend while still saving the full processor.
- Added compatibility handling for `eval_strategy`, `processing_class`, `remove_unused_columns`, and tokenizer/processor training on Windows.
- Added `training_summary.json` output for completed runs.

### Dataset preparation

- Rebuilt the OCR outputs from `raw_dataset/`.
- Generated `finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl` from the entire cleaned corpus.
- Truncated draft contexts to `450` words to keep the local 4B run stable at `1024` tokens.

## Cleaned Parse Result

- Supported files discovered: `8180`
- Successfully processed: `8179`
- Remaining failure: `ocr_pipeline/parse_failures.log` -> `SIP-20231101-KQKD.docx`
- Cleaned chunk rows written: `23978`
- Full-corpus draft training rows written: `23974`

## Final Full-Corpus Training Command

```powershell
.venv\Scripts\python.exe finetune/train.py \
  --dataset-path finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl \
  --output-dir finetune/outputs/qwen35_4b_full_corpus_draft23974 \
  --max-seq-length 1024 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --num-epochs 1 \
  --eval-split 0 \
  --log-steps 100 \
  --save-steps 500 \
  --warmup-steps 100 \
  --save-merged-model \
  --skip-gguf-export \
  --disable-response-only-masking
```

## Final Full-Corpus Run Result

- Output directory: `finetune/outputs/qwen35_4b_full_corpus_draft23974`
- Adapter: `finetune/outputs/qwen35_4b_full_corpus_draft23974/adapter`
- Merged model: `finetune/outputs/qwen35_4b_full_corpus_draft23974/merged_model`
- Summary file: `finetune/outputs/qwen35_4b_full_corpus_draft23974/training_summary.json`
- Train rows: `23974`
- Epochs: `1.0`
- Runtime: `68249.53s` (about `18.96h`)
- Train loss: `1.0765`
- Adapter size: about `0.10 GB`
- Merged model size: about `8.70 GB`

## Hugging Face Publication

- Repo: `Mikkkkoooo/qwen35-4b-private-analyst-full-corpus`
- Visibility: `private`
- Uploaded artifact: merged 16-bit model folder plus model card and training summary

Upload path used:

```python
from huggingface_hub import HfApi

api = HfApi(token="<HF_TOKEN>")
api.create_repo(
    repo_id="Mikkkkoooo/qwen35-4b-private-analyst-full-corpus",
    private=True,
    exist_ok=True,
)
api.upload_folder(
    repo_id="Mikkkkoooo/qwen35-4b-private-analyst-full-corpus",
    folder_path=r"D:\finetune\finetune\outputs\qwen35_4b_full_corpus_draft23974\merged_model",
    commit_message="Upload full-corpus Qwen 3.5 model",
)
```

## GGUF Export

- Export helper: `finetune/export_gguf.py`
- Built local `llama.cpp` under `C:\Users\Admin\.unsloth\llama.cpp`
- Export output directory:
  `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf`
- Main quantized model:
  `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.Q4_K_M.gguf`
- Companion mmproj file:
  `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.BF16-mmproj.gguf`

Export command used:

```powershell
.venv\Scripts\python.exe finetune/export_gguf.py \
  --model-path finetune/outputs/qwen35_4b_full_corpus_draft23974/adapter \
  --output-dir finetune/outputs/qwen35_4b_full_corpus_draft23974 \
  --gguf-name qwen3_5_4b_private_analyst_full_corpus_q4_k_m \
  --max-seq-length 1024
```

## Practical Notes

- The current full-corpus run is a strong bootstrap artifact trained on the entire cleaned draft corpus.
- Because the dataset is still draft-generated rather than fully human-labeled, treat this as a usable internal model, not necessarily the final house-style checkpoint.
- GGUF export now succeeds locally after manually building `llama.cpp` on Windows.
- For a stronger follow-up version, manually review the highest-value rows in `qwen35_full_corpus_draft.jsonl` and rerun the same command.
