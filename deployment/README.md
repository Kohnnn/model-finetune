# Deployment MVP

This folder contains the RAG-first local deployment path for the private analyst stack.

## Services

- `chromadb` stores embedded research chunks
- `llama` runs a GGUF model through the llama.cpp server image
- `app` exposes the retrieval API at `/healthz` and `/query`
- `ingest` is a one-shot container that reads `ocr_pipeline/chroma_chunks.jsonl` and upserts it into Chroma
- `nginx` is optional and only runs through the `proxy` profile
- `model_cache` persists FastEmbed and Hugging Face downloads between runs

## Current Deployment Model

The deployment path is now wired to the full-corpus Qwen 3.5 export:

- model: `deployment/models/Qwen3.5-4B.Q4_K_M.gguf`
- companion projection file: `deployment/models/Qwen3.5-4B.BF16-mmproj.gguf`
- source export folder: `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf`

## Required Files

Before startup, provide:

1. `deployment/.env` copied from `deployment/.env.example`
2. a real `CHROMA_AUTH_TOKEN`
3. the GGUF file inside `deployment/models/`
4. the matching mmproj file inside `deployment/models/`
5. `ocr_pipeline/chroma_chunks.jsonl`

Optional for the `proxy` profile:

- `deployment/certs/cert.pem`
- `deployment/certs/key.pem`

## Recommended Defaults

- collection: `research_chunks_v1`
- embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- higher-quality but much slower CPU option: `intfloat/multilingual-e5-large`
- retrieval: `top_k=4`
- llama model: `Qwen3.5-4B.Q4_K_M.gguf`
- llama mmproj: `Qwen3.5-4B.BF16-mmproj.gguf`

## Recommended Startup

From the repository root:

```bash
python deployment/bootstrap_local.py
```

The bootstrap script will:

1. validate `.env`, dataset, and GGUF model path
2. start `chromadb` and `llama`
3. run ingestion unless `--skip-ingest` is used
4. start the app
5. verify `http://localhost:8000/healthz`
6. optionally start nginx when `--with-proxy` is passed

Smoke-test the full flow on a subset first:

```bash
python deployment/bootstrap_local.py --ingest-limit 1024
```

Inspect options:

```bash
python deployment/bootstrap_local.py --help
```

The bootstrap script also accepts `--ingest-batch-size` for tuning local CPU runs.

## Manual Workflow

Start core services:

```bash
docker compose -f deployment/docker-compose.yml --env-file deployment/.env up -d chromadb llama
```

Ingest the retrieval corpus:

```bash
docker compose -f deployment/docker-compose.yml --env-file deployment/.env --profile ingest run --rm ingest
```

Start the app:

```bash
docker compose -f deployment/docker-compose.yml --env-file deployment/.env up -d app
```

Start nginx later if needed:

```bash
docker compose -f deployment/docker-compose.yml --env-file deployment/.env --profile proxy up -d nginx
```

## API Usage

Health check:

```bash
curl http://localhost:8000/healthz
```

Query endpoint:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key margin risks for AAA?"}'
```

The response includes:

- `answer`
- `sources`
- `context_used`
- `collection_name`

When a very small local model fails to produce a grounded cited answer, the app falls back to extractive evidence snippets instead of returning hallucinated output.

## Troubleshooting

- if bootstrap fails immediately, confirm `deployment/.env` exists and the token is not left as a placeholder
- if `llama` fails, confirm both files in `deployment/models/` match `LLAMA_MODEL_FILENAME` and `LLAMA_MMPROJ_FILENAME`
- if `/query` fails with collection errors, rerun the `ingest` profile
- if ingestion is too slow on CPU, keep the default MiniLM embedding model for first-pass indexing
- embedding downloads are cached under `deployment/model_cache/`
- if nginx fails, confirm both TLS files exist under `deployment/certs/`

## Publish Notes

- do not publish `.env`, model weights, local certificates, or Chroma runtime state
- do not publish private source documents from `raw_dataset/`
- generated retrieval data can also contain proprietary document text and is gitignored by default
