# Model Evaluation and Release Plan

## Purpose

Use two evaluation tiers:

1. a fixed five-case live smoke for stable release continuity;
2. a private 100–150-case claim-ledger pack for baseline-versus-candidate quality decisions.

Training loss is a training-health signal, not a release-quality result. The candidate must also improve evaluation loss, preserve grounded refusal behavior, pass the fixed smoke, and meet the claim-ledger gates.

## Held-Out Evaluation Pack

Build the private pack from document families excluded from training. Start from `deployment/benchmarks/claim_ledger_template.json`; completed packs and reports remain gitignored.

Cover both English and Vietnamese across:

- factual extraction;
- numeric extraction and trend analysis;
- comparison across documents;
- earnings and margin analysis;
- risk analysis;
- valuation analysis;
- analytical synthesis;
- unsupported questions requiring refusal.

Each case records:

- a stable case ID and question;
- expected `model` or `refusal` mode;
- language and task type;
- whether regression is critical;
- required terms, acceptable alternatives, expected numeric values, prohibited terms, and supporting `doc_id` values per claim; use JSON numbers for locale-neutral values and exact strings such as `"1,5%"` or `"1.500"` when locale-specific display matters;
- one to eight frozen evidence sources, including deliberately irrelevant evidence for refusal cases.

Do not reuse train or evaluation-split families in this pack. Human reviewers must verify the ledger and frozen evidence before using it as a gate.

## Three Evaluation Lanes

| Lane | Endpoint | Isolates | Main metric |
| --- | --- | --- | --- |
| Frozen context | `POST /generate-with-evidence` | generation with identical evidence | claim, numeric, and citation quality |
| Retrieval only | `POST /retrieve` | retriever/index quality | recall@k against supporting document IDs |
| Live RAG | `POST /query` | complete application behavior | grounded answer/refusal quality and latency |

Citation support is a deterministic proxy: a claim counts as citation-supported when the answer cites a returned source whose `doc_id` appears in that claim's ledger. It does not prove semantic entailment; reviewers still inspect sampled answers.

## Baseline and Candidate Commands

Use the checksum-verified `localgguf` backend and pin `LLAMA_CPP_IMAGE` as `repository@sha256:<64 hex>`. Bootstrap launches only Compose values validated from `deployment/.env`, and the running app independently verifies mounted GGUF/mmproj bytes against `SHA256SUMS`. Ingestion uses an immutable source snapshot and a shared exclusive lock; serving fails closed while that lock exists. Set distinct random values of at least 32 characters for `EVALUATION_API_TOKEN` and `EVALUATION_ATTESTATION_KEY` in `deployment/.env`. Export only the API token while running the evaluator; export only the attestation key while running release validation. Empty values disable the evaluation endpoints, Compose binds both app and raw llama-server ports to `127.0.0.1`, and nginx blocks evaluation routes/headers.

After human review freezes the pack, print its canonical JSON identity and store it independently as `EXPECTED_CLAIM_LEDGER_PACK_SHA256`:

```powershell
.venv\Scripts\python.exe deployment/evaluate_claim_ledger.py `
  --pack deployment/benchmarks/private_claim_ledger_v1.json `
  --print-pack-sha256
```

After reviewing the approved baseline report, store its `evaluation_target_sha256` independently as `EXPECTED_BASELINE_EVALUATION_TARGET_SHA256`. Do not derive either release pin from the pack or report supplied to `validate_release.py`.

Serve the base model against the release Chroma collection:

```powershell
.venv\Scripts\python.exe deployment/evaluate_claim_ledger.py `
  --pack deployment/benchmarks/private_claim_ledger_v1.json `
  --lane all `
  --json-output deployment/benchmarks/claim_ledger_baseline.json `
  --markdown-output deployment/benchmarks/claim_ledger_baseline.md
```

Replace the model bundle, recreate the app container, then serve the candidate against the same collection and unchanged pack:

```powershell
.venv\Scripts\python.exe deployment/evaluate_claim_ledger.py `
  --pack deployment/benchmarks/private_claim_ledger_v1.json `
  --lane all `
  --baseline-json deployment/benchmarks/claim_ledger_baseline.json `
  --json-output deployment/benchmarks/claim_ledger_candidate.json `
  --markdown-output deployment/benchmarks/claim_ledger_candidate.md
```

The evaluator requires identical `pack_id`, canonical pack SHA-256, question identities, top-k, and case/lane inventory. The app signs endpoint, request identity, retrieval settings, evidence identity, answer, mode, cited document IDs, end-to-end server latency, and an evaluation target derived from actual mounted GGUF/mmproj/corpus bytes, actual Chroma vectors/content plus ingestion generation, running app code, the digest-pinned llama.cpp runtime/configuration, and generation settings. Local reports retain these attestations and scoring inputs but omit source excerpts and paths. Baseline and candidate must share corpus, index, app, runtime, generation-configuration, collection, and embedding identities. Release validation verifies every HMAC attestation, checks the independently pinned pack and baseline target, requires the candidate model/mmproj identity to match the export manifest, recomputes every result, aggregate, and paired comparison, then hashes all three inputs. The evaluator reports paired wins, ties, losses, critical regressions, language/task slices, mode distribution, and nearest-rank p50/p95 server latency.

## Fixed Live Smoke

The five questions in `deployment/benchmarks/default_questions.json` remain the stable release smoke:

```powershell
.venv\Scripts\python.exe deployment/evaluate_live_query.py `
  --label candidate `
  --baseline-json deployment/benchmarks/baseline.json `
  --json-output-path deployment/benchmarks/candidate.json `
  --output-path deployment/benchmarks/candidate.md
```

Require at least four of five passes, no pass-count regression, matching question identities, and safe unsupported-question behavior. This smoke does not replace the claim-ledger pack.

## Release Gates

| Gate | Requirement |
| --- | --- |
| Approved data | All eligible rows approved; dataset audit has zero errors |
| Leakage control | Nonempty disjoint train/evaluation document IDs and family IDs; `split_strategy=document_family_id` |
| Training | bf16 LoRA, assistant-only loss, full approved dataset, train loss below `1.2` |
| Evaluation loss | Final evaluation loss below baseline evaluation loss |
| Fixed smoke | At least `4/5`, safe fallback, no baseline regression |
| Claim-ledger inventory | At least 100 cases, at least one model case, refusal case, retrievable claim, and expected number, with exactly frozen, retrieval, and live results for every case |
| Claim quality | Claim accuracy and numeric accuracy each at least `0.75` |
| Grounding | Citation precision and completeness each at least `0.75` |
| Retrieval | Recall@k at least `0.75` |
| Refusal | Refusal correctness `1.0` and zero false refusals |
| Latency | p95 at most 180 seconds |
| Paired comparison | Matching identity/inventory and no critical regression |
| Provenance | Immutable base revision, clean committed training code, dataset hash, merged model, GGUF, matching mmproj, and verified SHA-256 |

Validate after both evaluations and GGUF export:

```powershell
$env:EVALUATION_ATTESTATION_KEY = "<independently managed attestation key>"
$env:EXPECTED_CLAIM_LEDGER_PACK_SHA256 = "<reviewed canonical pack SHA-256>"
$env:EXPECTED_BASELINE_EVALUATION_TARGET_SHA256 = "<approved baseline evaluation_target_sha256>"
.venv\Scripts\python.exe finetune/validate_release.py `
  --run-dir finetune/outputs/qwen35_4b_approved_v1 `
  --benchmark-json deployment/benchmarks/candidate.json `
  --claim-ledger-pack deployment/benchmarks/private_claim_ledger_v1.json `
  --claim-ledger-baseline-json deployment/benchmarks/claim_ledger_baseline.json `
  --claim-ledger-json deployment/benchmarks/claim_ledger_candidate.json
```

## Release Evidence and Privacy

The local raw reports contain private questions, answers, source excerpts, document IDs, family IDs, and per-case results. Do not commit or upload them.

A passed release includes only:

- hashes and immutable lineage;
- training and evaluation aggregates;
- document/family counts, never IDs;
- fixed-smoke aggregates;
- claim-ledger pack ID/hash, case count, lanes, aggregate metrics, and comparison counts;
- model and metadata-only dataset cards;
- artifact SHA-256 inventory.

## Decision

- Release only when every gate passes.
- Any critical regression blocks release even when aggregate scores improve.
- A safe evidence fallback prevents hallucinated output but does not count as a successful model answer.
- If the candidate fails, fix data quality, retrieval, prompting, or training based on the failing lane; do not lower thresholds to hide the failure.

## Ongoing Monitoring

Run the fixed smoke weekly:

```powershell
.venv\Scripts\python.exe deployment/evaluate_live_query.py --output-path deployment/benchmarks/weekly_report.md
```

Run the private claim ledger after model, embedding, chunking, retrieval, prompt, or grounding changes. Keep the last approved release report as the paired baseline.
