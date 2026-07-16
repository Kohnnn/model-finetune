# Cleanup & Denoise Plan + Progress

_Last updated: 2026-06_

This document records the repository cleanup, secret remediation, dataset denoising,
and documentation work, plus what comes next. It is the single source of truth for the
2026-06 maintenance pass.

## Goals

1. Remove a leaked secret from the repo and its git history.
2. Tighten `.gitignore` and remove scratch/experiment clutter.
3. Abandon the Gemma 4 track and refocus on Qwen3.5-4B (see
   `docs/model-selection-decision.md`).
4. Make the markdown report corpus stricter and more training-friendly (drop
   compliance/contact/ratings noise, near-empty shells, and duplicates).
5. Add clear architecture/flow diagrams to the README.
6. Document the plan and progress here.

## Status summary

| Area | Status | Result |
| --- | --- | --- |
| Secret purge (`.env` / `HF_TOKEN`) | Done (pending force-push + token rotation) | Untracked + history rewrite prepared |
| `.gitignore` hardening | Done | Added `/.env`, `.mypy_cache/` |
| Remove tracked scratch scripts | Done | 9 root `test_*` Ollama scripts removed |
| Remove untracked scratch | Done | `check_ds.py`, `download_gemma4.py`, `trace_*.txt` deleted |
| Abandon Gemma 4 | Done | `train_gemma4.py` + 3 probes + plan doc removed |
| Exporter denoise upgrade | Done | Stricter line/page/doc filters + dedup |
| Corpus cleanup pass | Done | `markdown_reports/` 6228 → 5991 |
| README diagrams | Done | 2 generated diagrams under `docs/assets/architecture/` |
| Docs | Done | This file + roadmap + selection decision |

## 1. Secret remediation (`HF_TOKEN`)

**Problem:** `.env` containing `HF_TOKEN` was committed (commit introducing the
fine-tuning pipeline) on the **public** GitHub repo `Kohnnn/model-finetune`, and models
were pushed to Hugging Face.

**Actions:**

- `git rm --cached .env` (file kept locally, now gitignored).
- Added `.env` to `.gitignore`.
- History rewrite to purge `.env` from **all** commits using `git filter-repo`
  (installed as a pip package; invoked as `python <site-packages>/git_filter_repo.py`).
- Force-push the rewritten history to `origin` and `model-finetune`.

**Required human action:** **rotate the `HF_TOKEN`** in the Hugging Face account. History
rewriting removes it going forward, but the value must be assumed compromised because it
was public. Anyone with an existing clone must re-clone after the force-push.

## 2. `.gitignore`

Added:

- `/.env` (root environment file with secrets)
- `.mypy_cache/`

Existing ignores for venvs, caches, `raw_dataset/`, generated JSONL, `markdown_reports/`,
`finetune/outputs/`, deployment secrets/models/runtime, and the `command` notes file were
already present and retained.

## 3. Repo clutter removed

Tracked scratch scripts (`git rm`):

- `test_ollama.py`, `test_ollama2.py`, `test_ollama3.py`, `test_ollama4.py`,
  `test_ollama_extract.py`, `test_simple.py`, `test_custom_model.py`,
  `test_all_models.py`, `test_user_model.py`

These were ad-hoc `localhost:11434` pokes, not real tests. The real test suite lives in
`tests/` and is unaffected.

Untracked scratch deleted:

- `check_ds.py`, `download_gemma4.py`, `trace_dill.txt`, `trace_ds.txt`

Kept (per request, already gitignored): `command` (personal run notes).

## 4. Gemma 4 abandoned

Removed `finetune/train_gemma4.py`, `test_direct_transformers.py`,
`test_gemma4_load.py`, `test_gemma4_peft.py`, and `docs/gemma4-training-master-plan.md`.
Rationale and reopening criteria: `docs/model-selection-decision.md`.

## 5. Corpus denoise

### Exporter upgrade — `ocr_pipeline/export_markdown_reports.py`

New, stricter, training-friendly cleaning:

- **Line-level noise filter** (`LINE_NOISE_PATTERNS`): analyst certification, disclaimers,
  ratings legends (`Buy/Sell/...` and VI equivalents), contact lines (tel/fax/email/
  address), Bloomberg tickers, page numbers, copyright/footers, and orphaned pure-number
  rows.
- **Compliance-page drop** (`page_is_compliance`): pages dense with certification /
  disclosure / rating-definition markers are dropped wholesale.
- **Document-level minimum** (`--min-doc-words`, default 80): near-empty caption-only
  shells are dropped entirely.
- **De-duplication** (`content_fingerprint`, on by default; `--keep-duplicates` to disable):
  near-identical reports (e.g. PDF + PPTX + revised copies of the same document) collapse
  to one.
- **Manifest enrichment**: each row now carries `body_word_count` and
  `content_fingerprint`; the run logs drop counts (empty/short/duplicate).

### Cleanup pass — `ocr_pipeline/clean_markdown_reports.py` (new)

A fast post-filter that applies the same rules to the **already-generated**
`markdown_reports/` without re-parsing `raw_dataset/` (which takes hours). Supports
`--dry-run`. For an authoritative refresh from source, re-run the exporter instead.

**Result of the 2026-06 pass:**

```
kept=5991  rewritten=5991  |  deleted: empty=46  short=158  duplicate=33
```

`markdown_reports/`: **6228 → 5991** files; `manifest.jsonl` rebuilt in sync.

## 6. README diagrams

Generated via the 9Router image API (`cx/gpt-5.4-image`, 1792x1024 PNG) and saved to
`docs/assets/architecture/`:

- `pipeline-flow.png` — raw docs → OCR/clean → SFT dataset → Qwen3.5-4B fine-tune → GGUF
  → grounded RAG app.
- `deployment-architecture.png` — FastAPI app + Chroma + local GGUF inference, with the
  grounded-vs-fallback decision path.

Embedded at the top of the README "Architecture" section; the prior ASCII diagram is kept
as a collapsible text fallback.

> Note: the `xai/grok-2-image-1212` and `cx/gpt-5.3-image` models returned provider auth /
> account errors at generation time; `cx/gpt-5.4-image` worked.

The beginner fine-tuning journal adds 12 lifecycle diagrams under
`docs/assets/fine-tuning-journal/`. They were generated through 9Router with
`cx/gpt-5.5-image`, using the two architecture PNGs above as direct base64 style
references. Provider output was normalized to 1792x1024 PNG. No corpus content or
credentials were included in image prompts.

## Next steps (brainstorm)

Roughly in priority order:

1. **Regenerate the SFT dataset from the cleaned corpus.** The denoised
   `markdown_reports/` (and a refreshed `finetune_template.jsonl`) should feed
   `finetune/prepare_seed_dataset.py`. The current draft completions are not a reviewed
   gold set.
2. **Build a reviewed gold subset + benchmark gates.** Define a held-out eval (grounded-
   answer pass rate, hallucination/fallback rate) so training progress is measurable.
3. **Retrain Qwen3.5-4B** on the cleaned data with assistant-only loss masking, then export
   a fresh GGUF with known provenance (avoid the untrusted Ollama-derived lineage).
4. **Fix the bootstrap/compose mismatch** noted in `AGENTS.md`: `bootstrap_local.py` starts
   a `llama` service while `docker-compose.yml` defines `llama-server`/`ollama` under
   profiles. **(Done 2026-06)** — `bootstrap_local.py` now takes `--inference
   {localgguf,ollama}` (default `localgguf`), starts the correct profile-gated service,
   validates the `LLM_MODEL` GGUF for local runs, and points the app at the right backend.
   Covered by `tests/test_bootstrap_local.py`.
5. **Add an exporter regression test** under `tests/` for the new noise/dedup logic.
6. **Consider quantifying corpus quality** (token counts, language split, dedup ratio) into
   the manifest summary for dataset-card style reporting.
