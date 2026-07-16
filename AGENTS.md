# AGENTS.md

## Repo Shape
- Private AI analyst stack: `raw_dataset/` -> `ocr_pipeline/process_pdfs.py` -> `ocr_pipeline/chroma_chunks.jsonl` and `ocr_pipeline/finetune_template.jsonl` -> Chroma ingestion/RAG app -> optional Unsloth fine-tune -> GGUF deployment.
- Main entrypoints: OCR `ocr_pipeline/process_pdfs.py`, dataset prep `finetune/prepare_seed_dataset.py`, training `finetune/train.py`, GGUF export `finetune/export_gguf.py`, app `deployment/app/main.py`, ingest `deployment/app/ingest.py`.
- Generated/private-heavy paths include `raw_dataset/`, `ocr_pipeline/*.jsonl`, `finetune/outputs/`, `deployment/models/`, `deployment/chroma_data/`, `deployment/model_cache/`, and `deployment/.env`; do not commit or inspect more private data than needed.

## Setup
- Dependencies are split by stage, not centralized: `python -m pip install -r ocr_pipeline/requirements.txt`, `python -m pip install -r deployment/app/requirements.txt`, and for training first install CUDA PyTorch from `cu128`, then `python -m pip install -r finetune/requirements.txt`.
- Preferred training setup on this Windows machine is `./finetune/setup_gpu_env.ps1`; docs target Python 3.11, CUDA PyTorch, and a 16GB GPU.
- Root `package.json` only provides Playwright as a dev dependency; there are no Node app scripts.

## Verification
- Run all unit tests with `pytest -q`.
- Run focused tests with `pytest tests/test_process_pdfs.py -q`, `pytest tests/test_train.py -q`, or `pytest tests/test_rag.py -q`.
- `tests/conftest.py` injects repo root and `deployment/app` into `sys.path`, so app tests import modules as `from rag import ...` rather than package-qualified imports.
- No committed lint/type config exists; do not invent repo-specific formatter/typecheck requirements unless adding config in the same change.

## OCR Pipeline
- Install OCR deps before running: `python -m pip install -r ocr_pipeline/requirements.txt`.
- Standard parse command: `python ocr_pipeline/process_pdfs.py --input-dir raw_dataset --output-dir ocr_pipeline --extensions .pdf .docx .pptx`.
- PageIndex/vectorless RAG export command: `python ocr_pipeline/export_markdown_reports.py --input-dir raw_dataset --output-dir ocr_pipeline/markdown_reports --extensions .pdf .docx .pptx`; output is generated/private and gitignored.
- Despite the folder name, parsing is mostly text extraction; scanned/image-only PDFs are a known caveat.
- Office lock files matching `~$*` are intentionally skipped; failed files go to `ocr_pipeline/parse_failures.log`.
- `finetune_template.jsonl` contains assistant placeholders and is not meaningful SFT data until completions are filled or a seed dataset is generated.

## Fine-Tuning
- Fast validation only: `python finetune/train.py --dry-run --allow-empty-assistant --max-samples 10`.
- Build the current draft dataset with `python finetune/prepare_seed_dataset.py --input-path ocr_pipeline/finetune_template.jsonl --output-path finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl --max-rows 1000000 --max-context-words 450`.
- `train.py` blocks empty assistant-only data unless `--allow-empty-assistant` is set; that flag is for validation, not real training.
- Current documented full run writes under `finetune/outputs/qwen35_4b_full_corpus_draft23974/`; treat those model artifacts as large/private.
- Hugging Face upload uses `HF_TOKEN` from the environment; never write it into repo files.

## Deployment
- Required before local stack startup: copy `deployment/.env.example` to `deployment/.env`, set a real `CHROMA_AUTH_TOKEN`, provide `ocr_pipeline/chroma_chunks.jsonl`, and place both the `LLM_MODEL` GGUF (default `Qwen3.5-4B.Clean-Recovery.Q4_K_M.gguf`) and matching `Qwen3.5-4B.BF16-mmproj.gguf` in `deployment/models/`.
- Recommended smoke command from docs: `python deployment/bootstrap_local.py --ingest-limit 1024` (defaults to the `localgguf` llama.cpp backend; pass `--inference ollama` to use the Ollama profile instead).
- `bootstrap_local.py` starts profile-gated services correctly: `chromadb`, then the chosen inference service (`llama-server` for `localgguf`, `ollama` for `ollama`), then `app`. Local-GGUF validation checks the `LLM_MODEL` file used by `docker-compose.yml` (default `Qwen3.5-4B.Clean-Recovery.Q4_K_M.gguf`).
- Manual compose flow uses `--env-file deployment/.env`: start Chroma/inference, run the `ingest` profile, then start `app`.
- App endpoints are `GET /healthz` and `POST /query`; live benchmark command is `python deployment/evaluate_live_query.py --output-path deployment/benchmarks/latest_report.md`.
- The RAG app rejects ungrounded model answers and returns insufficient evidence/fallback excerpts instead of hallucinated text; preserve this behavior when changing prompt or answer parsing code.

## Existing Instructions
- No `.cursor/rules/`, `.cursorrules`, `.github/copilot-instructions.md`, CI workflows, or repo-local `opencode.json` were present when this file was updated.
