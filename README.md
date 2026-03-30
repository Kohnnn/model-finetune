# Private AI Analyst Stack

Private AI analyst stack for local financial research workflows.

- parse private research documents into clean chunks
- index the corpus for retrieval with ChromaDB
- serve grounded analyst answers through a local app
- fine-tune `unsloth/Qwen3.5-4B` on the cleaned corpus
- export a deployment-ready GGUF and publish a private Hugging Face model

The repository now contains a full end-to-end working path from `raw_dataset/` to:

- a private Hugging Face model: `Mikkkkoooo/qwen35-4b-private-analyst-full-corpus`
- a deployment-ready GGUF: `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.Q4_K_M.gguf`

## What This Repo Does

1. parses `.pdf`, `.docx`, and `.pptx` research files from `raw_dataset/`
2. removes most disclaimer and contact boilerplate before chunking
3. writes clean retrieval and SFT template datasets in `ocr_pipeline/`
4. ingests the retrieval corpus into ChromaDB
5. serves a private analyst-style RAG API through FastAPI and llama.cpp
6. fine-tunes `unsloth/Qwen3.5-4B` on the cleaned corpus and exports both merged HF and GGUF artifacts

## Current Status

- `ocr_pipeline/` is implemented and now strips most disclaimer/contact boilerplate before chunking
- latest full parse processed `8179/8180` supported files and produced `23978` cleaned chunks
- `deployment/` contains a working RAG-first app, ingestion flow, Docker packaging, and bootstrap script
- `finetune/` contains a validated Windows GPU training path for `unsloth/Qwen3.5-4B`
- the local RTX `4060 Ti 16GB` CUDA environment is validated and completed a full LoRA run
- final full-run artifact: `finetune/outputs/qwen35_4b_full_corpus_draft23974`
- final private Hugging Face repo: `Mikkkkoooo/qwen35-4b-private-analyst-full-corpus`
- final deployment GGUF: `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.Q4_K_M.gguf`

## System Architecture

```text
                              +---------------------------+
                              |       raw_dataset/        |
                              |  PDFs / DOCX / PPTX       |
                              +-------------+-------------+
                                            |
                                            v
                              +---------------------------+
                              | ocr_pipeline/process_pdfs |
                              | - extract text            |
                              | - remove boilerplate      |
                              | - chunk documents         |
                              +------+------+-------------+
                                     |      |
                  +------------------+      +-------------------+
                  |                                         |
                  v                                         v
    +-------------------------------+        +--------------------------------+
    | ocr_pipeline/chroma_chunks    |        | ocr_pipeline/finetune_template |
    | JSONL retrieval corpus        |        | JSONL chat templates           |
    +---------------+---------------+        +----------------+---------------+
                    |                                         |
                    v                                         v
    +-------------------------------+        +--------------------------------+
    | deployment/app/ingest.py      |        | finetune/prepare_seed_dataset  |
    | embed + upsert into ChromaDB  |        | build draft SFT rows           |
    +---------------+---------------+        +----------------+---------------+
                    |                                         |
                    v                                         v
    +-------------------------------+        +--------------------------------+
    |        ChromaDB collection    |        | qwen35_full_corpus_draft.jsonl |
    |     research_chunks_v1        |        | cleaned full-corpus SFT set    |
    +---------------+---------------+        +----------------+---------------+
                    |                                         |
                    |                                         v
                    |                         +--------------------------------+
                    |                         |      finetune/train.py          |
                    |                         |  Unsloth + LoRA on Qwen 3.5    |
                    |                         +----------------+---------------+
                    |                                         |
                    |                     +-------------------+-------------------+
                    |                     |                                       |
                    |                     v                                       v
                    |    +--------------------------------+     +--------------------------------------+
                    |    | merged_model/                  |     | gguf/Qwen3.5-4B.Q4_K_M.gguf         |
                    |    | private HF upload              |     | deployment-ready llama.cpp artifact  |
                    |    +--------------------------------+     +--------------------------------------+
                    |
                    v
    +-------------------------------+        +--------------------------------+
    | deployment/app/main.py        | -----> | llama.cpp server               |
    | FastAPI RAG orchestration     |        | local OpenAI-compatible API    |
    +---------------+---------------+        +----------------+---------------+
                    ^                                         |
                    |                                         v
    +-------------------------------+        +--------------------------------+
    | end users / analyst prompts   | <----- | grounded answer + citations    |
    +-------------------------------+        +--------------------------------+
```

## Data Flow

```text
[1] Private documents
    raw_dataset/
        |
        v
[2] Parse + cleanup
    ocr_pipeline/process_pdfs.py
        |
        +--> chroma_chunks.jsonl
        |       |
        |       v
        |   deployment/app/ingest.py
        |       |
        |       v
        |   ChromaDB
        |
        +--> finetune_template.jsonl
                |
                v
            finetune/prepare_seed_dataset.py
                |
                v
            qwen35_full_corpus_draft.jsonl
                |
                v
            finetune/train.py
                |
                +--> adapter/
                +--> merged_model/
                +--> training_summary.json
                +--> gguf/
                        |
                        v
                deployment/models/
                        |
                        v
                deployment/docker-compose.yml
                        |
                        v
                local private analyst service
```

## Repository Layout

```text
deployment/         Docker, app API, ingestion, bootstrap, nginx, env template
finetune/           Qwen 3.5 training pipeline, export helpers, training notes
ocr_pipeline/       local parsing and chunk generation pipeline
raw_dataset/        private source documents (gitignored)
tests/              focused regression tests for parser and training helpers
DEVELOPMENT_JOURNAL.md
FINE_TUNING_GUIDE.md
IMPLEMENTATION_NOTES.md
README.md
```

## Key Outputs

### Clean parse outputs

- retrieval corpus: `ocr_pipeline/chroma_chunks.jsonl`
- chat template corpus: `ocr_pipeline/finetune_template.jsonl`
- parse failures: `ocr_pipeline/parse_failures.log`

### Fine-tune outputs

- full-corpus draft dataset: `finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl`
- adapter: `finetune/outputs/qwen35_4b_full_corpus_draft23974/adapter`
- merged HF model: `finetune/outputs/qwen35_4b_full_corpus_draft23974/merged_model`
- training summary: `finetune/outputs/qwen35_4b_full_corpus_draft23974/training_summary.json`

### GGUF outputs

- quantized model: `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.Q4_K_M.gguf`
- companion mmproj: `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.BF16-mmproj.gguf`

### Published artifact

- private HF repo: `Mikkkkoooo/qwen35-4b-private-analyst-full-corpus`

## Quick Start

### 1. Install parser dependencies

```bash
python -m pip install -r ocr_pipeline/requirements.txt
```

### 2. Generate retrieval + SFT template data

```bash
python ocr_pipeline/process_pdfs.py \
  --input-dir raw_dataset \
  --output-dir ocr_pipeline \
  --extensions .pdf .docx .pptx
```

### 3. Prepare deployment settings

Windows PowerShell:

```powershell
Copy-Item deployment/.env.example deployment/.env
```

macOS/Linux/Git Bash:

```bash
cp deployment/.env.example deployment/.env
```

### 4. Wire the current full-corpus deployment model

Deployment is now wired to:

- `deployment/models/Qwen3.5-4B.Q4_K_M.gguf`
- `deployment/models/Qwen3.5-4B.BF16-mmproj.gguf`

These are copied from:

- `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf`

### 5. Bootstrap the local RAG stack

```bash
python deployment/bootstrap_local.py
```

For a quick ingest-only smoke start first:

```bash
python deployment/bootstrap_local.py --ingest-limit 1024
```

## Fine-Tuning Workflow

### GPU environment setup

```powershell
./finetune/setup_gpu_env.ps1
```

### Build the full-corpus draft dataset

```bash
python finetune/prepare_seed_dataset.py \
  --input-path ocr_pipeline/finetune_template.jsonl \
  --output-path finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl \
  --max-rows 1000000 \
  --max-context-words 450
```

### Run the full-corpus Qwen 3.5 fine-tune

```bash
python finetune/train.py \
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

### Export GGUF

```bash
python finetune/export_gguf.py \
  --model-path finetune/outputs/qwen35_4b_full_corpus_draft23974/adapter \
  --output-dir finetune/outputs/qwen35_4b_full_corpus_draft23974 \
  --gguf-name qwen3_5_4b_private_analyst_full_corpus_q4_k_m
```

### Push merged model to Hugging Face

```bash
python finetune/push_to_huggingface.py \
  --model-dir finetune/outputs/qwen35_4b_full_corpus_draft23974/merged_model \
  --repo-id Mikkkkoooo/qwen35-4b-private-analyst-full-corpus \
  --private
```

## Deployment Model Wiring

The deployment path expects these local environment values:

- `LLAMA_MODEL_FILENAME=Qwen3.5-4B.Q4_K_M.gguf`
- `LLAMA_MMPROJ_FILENAME=Qwen3.5-4B.BF16-mmproj.gguf`
- `LLM_MODEL_NAME=qwen3.5-private-analyst`

The compose stack passes both files into llama.cpp so the current export can be served directly.

## Training Metrics

Final full-corpus training summary:

- base model: `unsloth/Qwen3.5-4B`
- train rows: `23974`
- epochs: `1.0`
- runtime: `68249.53s` (~`18.96h`)
- train loss: `1.0765`
- adapter size: about `0.10 GB`
- merged model size: about `8.70 GB`
- GGUF directory size: about `3.15 GB`

## Documentation Map

- `DEVELOPMENT_JOURNAL.md` - chronological engineering journal for the whole stack
- `deployment/README.md` - deployment, env wiring, and bootstrap workflow
- `deployment/app/README.md` - app service internals and file map
- `ocr_pipeline/README.md` - parser details and output schema
- `finetune/README.md` - training workflow and artifact layout
- `finetune/QWEN35_TRAINING_NOTES.md` - detailed Qwen 3.5 troubleshooting and run notes
- `FINE_TUNING_GUIDE.md` - higher-level fine-tuning and deployment guide

## Validation Already Run

- `python -m compileall deployment/app deployment/bootstrap_local.py ocr_pipeline/process_pdfs.py tests`
- `pytest tests/test_process_pdfs.py tests/test_train.py tests/test_rag.py -q`
- `docker compose -f deployment/docker-compose.yml --env-file deployment/.env config`
- full-corpus fine-tune completed successfully
- merged model uploaded to private Hugging Face repo
- GGUF export completed successfully

## Publishing Notes

- `raw_dataset/` is private and gitignored
- generated JSONL datasets are gitignored
- model artifacts are gitignored locally
- `deployment/.env` is local-only and not committed
- the Hugging Face repo is private because the underlying corpus is private

## Commit Readiness

The repository is now initialized with git and cleaned for an initial source commit.

Ignored local-only state includes:

- `.venv/`
- `.agent/`
- `raw_dataset/`
- `finetune/outputs/`
- `ocr_pipeline/chroma_chunks.jsonl`
- `ocr_pipeline/finetune_template.jsonl`
- `deployment/.env`
- deployment runtime caches and local model files
