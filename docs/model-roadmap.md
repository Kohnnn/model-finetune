# Model Roadmap

## Current Failure Summary

| Issue | Severity | Root Cause |
|-------|----------|------------|
| Qwen 3.5 SFT data is dirty (hallucinated completions) | Critical | `outputs/datasets/vietcap_sft_generated.jsonl` used as training data without review |
| Missing assistant-only loss masking | High | `--disable-response-only-masking` flag was used, causing loss on prompt tokens |
| Stale GGUF provenance | High | Ollama model treated as trusted base; actual lineage unknown |
| App fallback masking | Medium | RAG app falls back to general analyst tone instead of admitting lack of evidence |
| No evaluation gates | High | No held-out benchmark to measure recovery progress |

## Sequencing

```
Phase 1: Qwen Recovery
  └── Clean dataset → assistant-only training → fresh GGUF export → benchmark

Phase 2: Gemma 4 Introduction
  └── Baseline evaluation → E2B/E4B local training → compare vs Qwen

Phase 3: Unified Release
  └── Shared benchmark gates → champion/challenger selection → deployment
```

## Champion / Challenger Strategy

| Role | Model | Justification |
|------|-------|---------------|
| **Champion** | Qwen 3.5 4B (recovered) | Already deployed, trusted RAG grounding, fast inference |
| **Challenger** | Gemma 4 E4B | Better architecture (per-layer embeddings), competitive size |

## Key Recommendations

1. **Stop training on dirty data** — do not use `outputs/datasets/vietcap_sft_generated.jsonl` until cleaned or regenerated
2. **Treat Ollama Qwen as untrusted provenance** — export fresh GGUF from a known-good Hugging Face artifact
3. **Fix Qwen first** — clean subset → assistant-only training → benchmark before starting Gemma work
4. **E2B as smoke test only** — `google/gemma-4-E2B-it` for quick local validation; E4B as main challenger

## Deliverables

- [ ] Clean Qwen SFT dataset (reviewed subset)
- [ ] Qwen 3.5 GGUF with known provenance
- [ ] Benchmark report comparing Qwen vs Gemma baselines
- [ ] Fine-tuned Gemma 4 E4B (if it outperforms Qwen)
- [ ] Unified release gates passing on both models

## Reading Order

1. `docs/model-roadmap.md` (this file)
2. `docs/qwen35-recovery-master-plan.md`
3. `docs/model-evaluation-and-release-plan.md`
4. `docs/gemma4-training-master-plan.md`
