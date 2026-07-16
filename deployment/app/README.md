# App Service

This folder contains the retrieval and orchestration service for the RAG MVP.

## Responsibilities

- expose `/healthz` and `/query`
- expose strict local evaluation endpoints `/retrieve` and `/generate-with-evidence`
- embed incoming queries
- retrieve relevant chunks from ChromaDB
- assemble a grounded analyst prompt
- call the llama.cpp OpenAI-compatible API
- return answer text plus cited sources

If the local model fails to return a grounded cited answer, the service falls back to extractive evidence snippets built from the retrieved chunks. `/retrieve` isolates retrieval without generation; `/generate-with-evidence` isolates generation using one to eight explicitly supplied evidence chunks. Ingestion snapshots the source once and holds a shared exclusive lock; active or interrupted ingestion fails closed for health and queries, while a successful limited pilot remains queryable but cannot be release-evaluated. Evaluation access requires mounted GGUF/mmproj files matching `SHA256SUMS`, a digest-pinned `LLAMA_CPP_IMAGE`, the exact internal llama-server URL, distinct secrets of at least 32 characters, a complete Chroma snapshot, and `X-Evaluation-Token` matching `EVALUATION_API_TOKEN`. Every signed request rehashes changed target files and revalidates Chroma. Responses sign request/result metadata, end-to-end server latency, and the actual model/mmproj/corpus/index generation/app/runtime/generation-config target with `EVALUATION_ATTESTATION_KEY`; empty or invalid configuration returns `404`. nginx blocks the evaluation-only routes and strips the evaluation token from proxied `/query`, so attestations remain loopback-only.

## File Map

- `main.py` - FastAPI entrypoint and request flow
- `ingest.py` - one-shot ingestion script for `ocr_pipeline/chroma_chunks.jsonl`
- `settings.py` - environment-backed configuration
- `schemas.py` - request and response models
- `rag.py` - retrieval parsing, context assembly, citation formatting
- `prompts.py` - system and user prompt construction
- `embeddings.py` - FastEmbed wrapper for passage and query embeddings
- `Dockerfile` - container build
- `requirements.txt` - runtime dependencies

## Local Smoke Checks

```bash
python deployment/bootstrap_local.py --help
python deployment/bootstrap_local.py --ingest-limit 1024
docker compose -f deployment/docker-compose.yml --env-file deployment/.env build app
docker run --rm deployment-app python -c "from main import app; print(app.title)"
```
