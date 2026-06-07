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

Phase 2: Quality + Release  (Gemma 4 challenger DROPPED 2026-06)
  └── Reviewed gold SFT set → benchmark gates → single-champion deployment
```

> **2026-06 update:** the Gemma 4 challenger track has been **dropped**. The target
> task is VietCap-style equity-research prose with no tool-calling/agentic need, so
> Gemma 4's multimodal/agentic strengths add no value, and its Unsloth tokenizer
> friction was not worth the cost. Qwen3.5-4B is the sole champion. See
> `docs/model-selection-decision.md`.

## Champion / Challenger Strategy

| Role | Model | Justification |
|------|-------|---------------|
| **Champion** | Qwen 3.5 4B (recovered) | Already deployed, trusted RAG grounding, fast inference, proven Unsloth→GGUF path |
| ~~Challenger~~ | ~~Gemma 4 E4B~~ | **Dropped 2026-06** — irrelevant strengths for text-only analyst output, tokenizer friction, no quality win |

## Key Recommendations

1. **Stop training on dirty data** — do not use `outputs/datasets/vietcap_sft_generated.jsonl` until cleaned or regenerated
2. **Treat Ollama Qwen as untrusted provenance** — export fresh GGUF from a known-good Hugging Face artifact
3. **Stay on Qwen3.5-4B** — clean subset → assistant-only training → benchmark; do not reopen the Gemma track without a concrete capability gap
4. **Denoise the corpus first** — stricter exporter + report cleanup feed a cleaner SFT set (see `docs/cleanup-and-denoise-plan.md`)

## Deliverables

- [ ] Clean Qwen SFT dataset (reviewed subset)
- [x] Qwen 3.5 full-corpus draft GGUF exported and copied to `deployment/models/`
- [ ] Qwen 3.5 clean/reviewed GGUF with known provenance
- [x] Initial live Qwen benchmark report at `deployment/benchmarks/latest_report.md`
- [x] Corpus denoise pass over `markdown_reports/` (6228 → 5991, see cleanup plan)
- [ ] Unified release gates passing on the Qwen champion
- [x] ~~Gemma 4 baseline/fine-tune~~ — dropped 2026-06

## Current Status — 2026-06

- v0.1 Qwen path is operational: OCR outputs, full-corpus draft dataset, training summary, GGUF, mmproj, and live benchmark report are present locally.
- Quality gate is not complete: the draft dataset is not a reviewed gold set, and the live model frequently falls back to extractive evidence instead of producing grounded cited answers.
- Corpus denoising shipped: the markdown report exporter now strips compliance/contact/ratings noise, drops near-empty shells, and de-duplicates reports.
- Gemma 4 challenger track dropped; next work prioritizes a clean reviewed SFT set and benchmark gates on Qwen3.5-4B.

## Reading Order

1. `docs/model-roadmap.md` (this file)
2. `docs/model-selection-decision.md`
3. `docs/cleanup-and-denoise-plan.md`
4. `docs/qwen35-recovery-master-plan.md`
5. `docs/model-evaluation-and-release-plan.md`
