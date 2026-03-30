# App Service

This folder contains the retrieval and orchestration service for the RAG MVP.

## Responsibilities

- expose `/healthz` and `/query`
- embed incoming queries
- retrieve relevant chunks from ChromaDB
- assemble a grounded analyst prompt
- call the llama.cpp OpenAI-compatible API
- return answer text plus cited sources

If the local model fails to return a grounded cited answer, the service falls back to extractive evidence snippets built from the retrieved chunks.

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
