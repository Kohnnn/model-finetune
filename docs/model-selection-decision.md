# Model Selection Decision

_Last updated: 2026-06_

## Decision

**Champion model: `unsloth/Qwen3.5-4B`** — fine-tuned with Unsloth + LoRA, exported to
GGUF (Q4_K_M) and served locally behind the grounded RAG app.

The **Gemma 4 challenger track is dropped.**

## Why this task does not need an agentic/tool-calling model

The product is a **private equity-research analyst** that produces VietCap-style prose:
grounded, analytical commentary in Vietnamese and English, strictly based on retrieved
evidence. It does **not**:

- call tools or functions,
- plan multi-step agent workflows,
- require vision/audio multimodality for the core answer path,
- need frontier-scale general reasoning.

It **does** need:

- strong bilingual (VI/EN) instruction-following and financial-prose fluency,
- faithful long-context grounding (low hallucination on retrieved chunks),
- clean, reproducible local fine-tuning and GGUF export on a single 16 GB GPU,
- fast local inference for an interactive app.

So the selection weighting favors instruction-following + bilingual fluency + a proven
local train→deploy path over raw agent/reasoning benchmarks.

## Options considered (2026 landscape)

| Model | Verdict | Notes |
| --- | --- | --- |
| **Qwen3.5-4B** | **Selected** | Strong VI/EN, multimodal-capable but used text-only, Unsloth-supported, fits 16 GB, and the entire pipeline + deployment is already wired around it. Lowest risk, highest leverage. |
| Qwen3.5-9B | Deferred | Higher quality ceiling but heavier to train/serve locally. Revisit only if 4B fails quality gates. |
| SmolLM3-3B | Rejected | Fully open and very light, but weaker Vietnamese finance coverage than Qwen. |
| Qwen3.6-35B-A3B (MoE) | Rejected | Frontier-ish, but oversized for local 16 GB train/serve and unnecessary for prose grounding. |
| **Gemma 4 (E2B/E4B/31B/26B-A4B)** | **Dropped** | See below. |

## Why Gemma 4 was dropped

1. **Strengths are irrelevant here.** Gemma 4's headline advantages — multimodal vision/audio
   encoders, agentic capability, per-layer-embedding on-device efficiency — do not move the
   needle for text-only, retrieval-grounded analyst prose.
2. **Tooling friction.** Local fine-tuning required working around Unsloth tokenizer patches
   (the abandoned `train_gemma4.py` used a direct transformers + PEFT path specifically to
   bypass them). That is ongoing maintenance cost for no clear payoff.
3. **No demonstrated quality win** over Qwen3.5-4B on this corpus and task.
4. **Focus.** One well-tuned champion with clean data and benchmark gates beats two
   half-finished tracks.

### Removed artifacts

- `finetune/train_gemma4.py` (direct transformers + PEFT trainer)
- `test_direct_transformers.py`, `test_gemma4_load.py`, `test_gemma4_peft.py` (probe scripts)
- `download_gemma4.py` (HF download helper)
- `docs/gemma4-training-master-plan.md` (obsolete plan)

Educational background material under `assets/gemma4/` and the README "Recommended Reading"
entry are retained as general reference only.

## Reopening criteria

Reconsider a non-Qwen model only if **all** of the following hold:

- Qwen3.5-4B fails a defined quality gate (grounded-answer pass rate, hallucination rate)
  after a clean, reviewed SFT set and tuning, **and**
- a specific, measured capability gap is identified that a candidate model demonstrably
  closes, **and**
- the candidate trains and exports to GGUF on the existing 16 GB hardware without bespoke
  tooling hacks.
