# Private AI Analyst Stack

Private AI analyst stack for local financial research workflows.

- parse private research documents into clean chunks
- index the corpus for retrieval with ChromaDB
- serve grounded analyst answers through a local app
- fine-tune `unsloth/Qwen3.5-4B` on the cleaned corpus
- export a deployment-ready GGUF and publish a private Hugging Face model

Key links:

- GitHub code: `https://github.com/Kohnnn/finetune`
- GitHub model mirror: `https://github.com/Kohnnn/model-finetune`
- Hugging Face model: `https://huggingface.co/Mikkkkoooo/qwen35-4b-private-analyst-full-corpus`

## Snapshot

- parse result: `8179/8180` supported files processed
- cleaned chunks: `23978`
- full-corpus training rows: `23974`
- final train loss: `1.0765`
- final model folder: `finetune/outputs/qwen35_4b_full_corpus_draft23974`
- deployment model: `deployment/models/Qwen3.5-4B.Q4_K_M.gguf`

## Architecture

```text
+------------------+      +---------------------------+      +----------------------+
|   raw_dataset/   | ---> | ocr_pipeline/process_pdfs | ---> | chroma_chunks.jsonl  |
| PDF / DOCX / PPTX|      | extract + clean + chunk   |      | finetune_template    |
+------------------+      +---------------------------+      +----------+-----------+
                                                                       / \
                                                                      /   \
                                                                     v     v
                                                     +------------------+   +---------------------------+
                                                     | deployment ingest|   | prepare_seed_dataset.py   |
                                                     | embed -> Chroma  |   | build draft SFT dataset   |
                                                     +--------+---------+   +-------------+-------------+
                                                              |                           |
                                                              v                           v
                                                     +------------------+      +-------------------------+
                                                     | ChromaDB         |      | qwen35_full_corpus     |
                                                     | research_chunks  |      | _draft.jsonl           |
                                                     +--------+---------+      +------------+------------+
                                                              |                             |
                                                              |                             v
                                                              |                +--------------------------+
                                                              |                | finetune/train.py        |
                                                              |                | Unsloth LoRA on Qwen 3.5 |
                                                              |                +-------------+------------+
                                                              |                              |
                                                              |                   +----------+----------+
                                                              |                   |                     |
                                                              |                   v                     v
                                                              |        +-------------------+   +----------------------+
                                                              |        | merged_model/     |   | gguf/ Q4_K_M export  |
                                                              |        | private HF upload |   | + mmproj companion   |
                                                              |        +-------------------+   +----------+-----------+
                                                              |                                         |
                                                              +------------------------------+          |
                                                                                             |          v
+------------------+      +----------------------+      +------------------+      +--------------> +------------------+
| analyst prompts  | ---> | deployment/app/main | ---> | llama.cpp server | ---> | grounded API   | | deployment/      |
+------------------+      | FastAPI orchestration|      | OpenAI-style     |      | /query         | | models/         |
                          +----------------------+      +------------------+      +----------------+ +------------------+
```

## Data Flow

```text
raw_dataset/
  -> ocr_pipeline/process_pdfs.py
     -> ocr_pipeline/chroma_chunks.jsonl
        -> deployment/app/ingest.py
           -> ChromaDB collection: research_chunks_v1
     -> ocr_pipeline/finetune_template.jsonl
        -> finetune/prepare_seed_dataset.py
           -> finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl
              -> finetune/train.py
                 -> adapter/
                 -> merged_model/
                 -> training_summary.json
                 -> gguf/
                    -> deployment/models/
                       -> deployment/docker-compose.yml
                          -> live analyst service
```

## How To Run

### 1. Parse documents

```bash
python ocr_pipeline/process_pdfs.py \
  --input-dir raw_dataset \
  --output-dir ocr_pipeline \
  --extensions .pdf .docx .pptx
```

### 2. Run the merged Hugging Face model with `transformers`

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "Mikkkkoooo/qwen35-4b-private-analyst-full-corpus"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {"role": "user", "content": "Summarize the key margin risks for a consumer lender."}
]

prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

### 3. Run the GGUF with `llama.cpp`

```bash
llama-cli \
  -m finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.Q4_K_M.gguf \
  --mmproj finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.BF16-mmproj.gguf \
  -cnv \
  -p "Summarize the key margin risks for a consumer lender."
```

### 4. Run in Ollama

This path is for local experimentation. Keep the model private.

1. Put these two files in the same folder:
   - `Qwen3.5-4B.Q4_K_M.gguf`
   - `Qwen3.5-4B.BF16-mmproj.gguf`
2. Create a `Modelfile`:

```text
FROM ./Qwen3.5-4B.Q4_K_M.gguf
TEMPLATE "{{ .Prompt }}"
PARAMETER num_ctx 4096
SYSTEM You are a private financial research analyst. Answer concisely and stay grounded in provided evidence.
```

3. Build and run:

```bash
ollama create private-analyst-qwen35 -f Modelfile
ollama run private-analyst-qwen35 "Summarize the key margin risks for a consumer lender."
```

If your Ollama build does not handle the Qwen 3.5 companion projection file cleanly, use the `llama.cpp` path above instead. The `llama.cpp` path is the validated one in this repo.

### 5. Run the full private analyst service

Prepare `deployment/.env` from `deployment/.env.example`, then run:

```bash
python deployment/bootstrap_local.py
```

Query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key margin risks for ACB?"}'
```

### 6. Run a small live benchmark against `/query`

```bash
python deployment/evaluate_live_query.py --output-path deployment/benchmarks/latest_report.md
```

## Fine-Tuning Workflow

### GPU environment

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

### Train the full-corpus Qwen 3.5 model

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

## Release Checklist

- rerun parse and review `ocr_pipeline/parse_failures.log`
- regenerate the SFT dataset
- train and save `training_summary.json`
- export merged HF and GGUF artifacts
- copy new `.gguf` and `mmproj` into `deployment/models/`
- run `python deployment/bootstrap_local.py --ingest-limit 1024`
- verify `/healthz`, `/query`, and `deployment/evaluate_live_query.py`
- update README, journal, release notes, and model card
- push GitHub commits and upload the HF model

## Roadmap

### Model quality

- replace draft-generated completions with a curated human-reviewed SFT set
- add a held-out evaluation pack of analyst questions and golden answers
- compare full-corpus training against a smaller higher-quality reviewed subset
- retry response-only masking once the Windows Qwen 3.5 path is more stable

### Retrieval quality

- tune chunking and overlap per document type
- add metadata-aware retrieval filters by company, sector, and year
- benchmark embedding choices against your actual analyst questions

### Deployment

- benchmark the new Qwen 3.5 GGUF in the live RAG app
- reduce fallback-only answers through better serving prompts and evaluation loops
- add a production profile for HTTPS and remote access

### Ops and publishing

- tag future model versions consistently across GitHub, GGUF, and Hugging Face
- automate smoke tests for parse, ingest, query, train, and export flows
- keep all private corpora and generated datasets out of public distribution

## Docs

- `DEVELOPMENT_JOURNAL.md` - chronological engineering journal
- `RELEASE_NOTES.md` - milestone summary
- `deployment/README.md` - deployment and bootstrap workflow
- `deployment/app/README.md` - app internals
- `ocr_pipeline/README.md` - parser details and output schema
- `finetune/README.md` - training workflow and artifact layout
- `finetune/QWEN35_TRAINING_NOTES.md` - detailed Qwen 3.5 troubleshooting log
- `FINE_TUNING_GUIDE.md` - higher-level fine-tuning guide

## Privacy Notes

- `raw_dataset/` is private and gitignored
- generated JSONL datasets are gitignored
- local model binaries are gitignored
- `deployment/.env` is local-only
- the Hugging Face model repo is private because the source corpus is private
