# Deployment MVP

This folder contains the RAG-first local deployment path for the private analyst stack.

## Services

- `chromadb` stores embedded research chunks
- `llama-server` runs a verified GGUF model through the llama.cpp server image
- `app` exposes `/healthz` and `/query`, plus local evaluation endpoints `/retrieve` and `/generate-with-evidence`
- `ingest` is a one-shot container that snapshots `ocr_pipeline/chroma_chunks.jsonl`, holds a shared exclusive lock, and upserts the immutable snapshot into Chroma
- `nginx` is optional and only runs through the `proxy` profile
- `model_cache` persists FastEmbed and Hugging Face downloads between runs

## Historical v0.1 Deployment Model

The prior local deployment used the full-corpus draft Qwen 3.5 export. New releases must pass the visual journal's release gates:

- model: `deployment/models/Qwen3.5-4B.Q4_K_M.gguf`
- companion projection file: `deployment/models/Qwen3.5-4B.BF16-mmproj.gguf`
- source export folder: `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf`

## Required Files

Before startup, provide:

1. `deployment/.env` copied from `deployment/.env.example`
2. a real `CHROMA_AUTH_TOKEN`
3. the GGUF file inside `deployment/models/`
4. the matching mmproj file inside `deployment/models/`
5. `deployment/models/SHA256SUMS` with entries for both files
6. `ocr_pipeline/chroma_chunks.jsonl`

Optional for the `proxy` profile:

- `deployment/certs/cert.pem`
- `deployment/certs/key.pem`

## Recommended Defaults

- collection: `research_chunks_v1`
- embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- higher-quality but much slower CPU option: `intfloat/multilingual-e5-large`
- retrieval: `top_k=4`
- llama model: `Qwen3.5-4B.Clean-Recovery.Q4_K_M.gguf`
- llama mmproj: `Qwen3.5-4B.BF16-mmproj.gguf`

## Recommended Startup

From the repository root:

```bash
python deployment/bootstrap_local.py
```

The bootstrap script will:

1. validate `.env`, dataset, GGUF, mmproj, and both SHA-256 checksums for `localgguf`
2. start `chromadb` and the selected inference service (`llama-server` for the
   default `localgguf` backend, or `ollama` for `--inference ollama`)
3. run ingestion unless `--skip-ingest` is used
4. start the app (pointing it at the chosen backend)
5. verify Chroma and inference readiness through `http://localhost:8000/healthz`
6. optionally start nginx when `--with-proxy` is passed

By default the script uses the local GGUF backend. To use Ollama instead, first create or pull the private model into the persistent Ollama volume, set its exact local name as `OLLAMA_MODEL`, then run:

```bash
python deployment/bootstrap_local.py --inference ollama
```

Bootstrap retries `ollama show` and stops if that local model is missing. This proves availability, not artifact identity; use the default local-GGUF path when SHA-256 reproducibility is required.

Smoke-test the full flow on a subset first:

```bash
python deployment/bootstrap_local.py --ingest-limit 1024
```

Inspect options:

```bash
python deployment/bootstrap_local.py --help
```

The bootstrap script also accepts `--ingest-batch-size` for tuning local CPU runs. It removes inherited shell overrides for Compose variables and launches only the values validated from `deployment/.env`.

## Manual Workflow

Start core services:

```bash
docker compose -f deployment/docker-compose.yml --env-file deployment/.env --profile localgguf up -d chromadb llama-server
```

Ingest the retrieval corpus:

```bash
docker compose -f deployment/docker-compose.yml --env-file deployment/.env --profile ingest run --rm ingest
```

Start the app for local GGUF:

```bash
docker compose -f deployment/docker-compose.yml --env-file deployment/.env up -d app
```

For manual Ollama startup, set `LLAMA_API_URL=http://ollama:11434/v1` and `LLM_MODEL_NAME` to the exact local Ollama name in `deployment/.env`, then run:

```bash
docker compose -f deployment/docker-compose.yml --env-file deployment/.env --profile ollama up -d chromadb ollama
docker compose -f deployment/docker-compose.yml --env-file deployment/.env up -d app
```

Run the fixed five-case release smoke against `/query`:

```bash
python deployment/evaluate_live_query.py --output-path deployment/benchmarks/latest_report.md
```

For release-quality comparison, create a private 100–150-case pack from held-out document families using `deployment/benchmarks/claim_ledger_template.json`. Release evaluation requires the checksum-verified `localgguf` backend and `LLAMA_CPP_IMAGE` pinned as `repository@sha256:<64 hex>`. Both bootstrap and the running app verify the mounted GGUF and mmproj against `deployment/models/SHA256SUMS`. Set distinct random values of at least 32 characters for `EVALUATION_API_TOKEN` and `EVALUATION_ATTESTATION_KEY` in `deployment/.env`. Export only the API token while running the evaluator; export only the attestation key while running release validation. Empty values disable `/retrieve` and `/generate-with-evidence`. Compose binds both the app and raw llama-server ports to `127.0.0.1`; nginx returns `404` for evaluation routes and strips evaluation credentials from `/query`.

After human review freezes the pack, record its canonical JSON hash in protected configuration; do not derive the release pin from an unreviewed caller-supplied file:

```bash
python deployment/evaluate_claim_ledger.py --pack deployment/benchmarks/private_claim_ledger_v1.json --print-pack-sha256
```

After reviewing the approved baseline report, separately record its `evaluation_target_sha256`. Export both independently approved values for release validation as `EXPECTED_CLAIM_LEDGER_PACK_SHA256` and `EXPECTED_BASELINE_EVALUATION_TARGET_SHA256`. Run the base model:

```bash
python deployment/evaluate_claim_ledger.py --pack deployment/benchmarks/private_claim_ledger_v1.json --lane all --json-output deployment/benchmarks/claim_ledger_baseline.json --markdown-output deployment/benchmarks/claim_ledger_baseline.md
```

Then replace the served model bundle, recreate the app container, and serve the candidate against the same Chroma collection and unchanged pack:

```bash
python deployment/evaluate_claim_ledger.py --pack deployment/benchmarks/private_claim_ledger_v1.json --lane all --baseline-json deployment/benchmarks/claim_ledger_baseline.json --json-output deployment/benchmarks/claim_ledger_candidate.json --markdown-output deployment/benchmarks/claim_ledger_candidate.md
```

`frozen` scores generation with identical supplied evidence, `retrieval` scores recall@k, and `live` scores end-to-end RAG. Local reports retain HMAC-attested request identities, answers, modes, cited document IDs, end-to-end server latency, and an evaluation-target identity derived from the mounted GGUF/mmproj bytes, corpus bytes, actual Chroma vectors/content, an ingestion generation, running app code, the digest-pinned llama.cpp runtime/configuration, and generation settings. They omit source excerpts and paths. Baseline and candidate must share corpus, index, app, runtime, generation configuration, collection, and embedding identities; candidate model/mmproj hashes must match its export manifest. Completed packs and reports are private and recursively gitignored. The evaluator exits nonzero on threshold failure, pack/inventory/target mismatch, or critical regression.

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
- `answer_mode` (`model`, `evidence_fallback`, or `insufficient_evidence`)

Local evaluation also uses:

- `POST /retrieve` with the same body as `/query` to return retrieval sources without generation;
- `POST /generate-with-evidence` with `query` and one to eight explicit source objects to test generation on frozen evidence.

These endpoints accept strict bounded schemas and exist to isolate retrieval from generation. `/query` remains the application endpoint.

When the local model fails to produce a grounded cited answer, the app falls back to extractive evidence snippets instead of returning hallucinated output. Benchmarks count this as safe behavior, not a successful model-quality answer.

## Troubleshooting

- if bootstrap fails immediately, confirm `deployment/.env` exists and the token is not left as a placeholder
- if `llama-server` fails, confirm `LLM_MODEL`, `LLAMA_MMPROJ_FILENAME`, and both `SHA256SUMS` entries match files in `deployment/models/`
- if `/healthz` is degraded or `/query` reports an incomplete index, rerun the `ingest` profile; interrupted ingestion fails closed, while a successful limited pilot remains queryable but cannot be release-evaluated
- if a crashed container leaves the shared ingestion lock behind, stop the stack before removing the stale lock from the `ingestion_locks` Compose volume, then restart and rerun ingestion; never clear a lock while any ingester can run
- if ingestion is too slow on CPU, keep the default MiniLM embedding model for first-pass indexing
- embedding downloads are cached under `deployment/model_cache/`
- if nginx fails, confirm both TLS files exist under `deployment/certs/`

## Publish Notes

- do not publish `.env`, model weights, local certificates, or Chroma runtime state
- do not publish private source documents from `raw_dataset/`
- generated retrieval data can also contain proprietary document text and is gitignored by default
