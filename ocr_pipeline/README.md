# OCR Pipeline

This folder contains the local document parsing and chunk generation pipeline.

## Purpose

`process_pdfs.py` scans research files, extracts text and supported tables/charts/notes, removes confidently detected boilerplate, groups exact copies and minor revisions, creates sentence-aware chunks with source spans, and writes three JSONL outputs:

- `chroma_chunks.jsonl` for retrieval
- `finetune_template.jsonl` for later supervised fine-tuning
- `parse_manifest.jsonl` with parsed/skipped/failed status and reason codes

Despite the folder name, the current implementation is primarily text extraction rather than image OCR.

## Supported Inputs

- `.pdf`
- `.docx`
- `.pptx`

Files matching `~$*` are skipped automatically to avoid Office lock-file failures.

## Install

```bash
python -m pip install -r ocr_pipeline/requirements.txt
```

## Usage

```bash
python ocr_pipeline/process_pdfs.py \
  --input-dir raw_dataset \
  --output-dir ocr_pipeline \
  --extensions .pdf .docx .pptx
```

## PageIndex / Vectorless RAG Markdown Export

PageIndex works best when Markdown preserves the report hierarchy with `#`, `##`, and `###` headings. Generate cleaned Markdown reports with:

```bash
python ocr_pipeline/export_markdown_reports.py \
  --input-dir raw_dataset \
  --output-dir ocr_pipeline/markdown_reports \
  --extensions .pdf .docx .pptx
```

Pilot run first:

```bash
python ocr_pipeline/export_markdown_reports.py --limit 5 --sample-mode random --seed 3407
```

Outputs:

- one `.md` file per retained source report
- `ocr_pipeline/markdown_reports/manifest.jsonl` with source metadata and output paths
- `ocr_pipeline/markdown_reports/markdown_failures.log` when files fail

Cleanup behavior:

- strips common contact/disclaimer/analyst-certification tails using the same markers as `process_pdfs.py`
- removes common VietCap headers and page-number noise
- preserves natural page/slide sections as `## Page N` instead of chunking for vector search

Optional layout parser:

```bash
python -m pip install markitdown
python ocr_pipeline/export_markdown_reports.py --use-markitdown --limit 5
```

`markitdown` can improve some Office/PDF layout conversion. Scanned/image-only PDFs still need a real OCR step before PageIndex; PageIndex docs recommend their OCR for preserving PDF hierarchy before using Markdown mode.

Index a generated report with self-hosted PageIndex:

```bash
python run_pageindex.py --md_path D:/finetune/ocr_pipeline/markdown_reports/<report>.md
```

## Useful Options

```bash
python ocr_pipeline/process_pdfs.py --help
```

Key arguments:

- `--limit` for pilot runs; use a separate `--output-dir` unless intentionally passing `--allow-pilot-overwrite`
- `--sample-mode` and `--seed` for deterministic sampling
- `--chunk-words`, `--overlap-words`, `--min-chunk-words`
- `--trim-tail-pages` for explicit extra trimming; the safe default is `0`
- `--near-duplicate-distance` for grouping minor revisions into one split family
- `--keep-duplicates` only when exact duplicate documents are intentionally needed

## Output Schema

### `chroma_chunks.jsonl`

Each row contains `id`, `text`, and metadata including:

- `relative_source`, `doc_id`, `document_family_id`
- `source_file_sha256`, `content_sha256`, `parser_schema_version`
- `title`, `year`, `language`, `file_extension`
- `chunk_index`, `chunk_word_count`
- `start_page`, `end_page`, `source_page_numbers`
- `source_word_start`, `source_word_end`
- `extraction_method`, `extraction_warnings`

Absolute local source paths are not emitted.

### `finetune_template.jsonl`

Each row contains chat-format messages, chunk provenance, `context_sha256`, `source_spans`, a task type, and empty reviewer fields. Every parser row starts with `metadata.review_status="draft"` and an empty assistant placeholder.

Fill an original assistant completion, verify every claim and number against the source span, complete `reviewed_by` and timezone-aware `reviewed_at`, then set `review_status="approved"`. Run `finetune/audit_dataset.py`; training rejects any approved dataset with audit errors.

## Current Caveats

- scanned or image-only PDFs are flagged in `extraction_warnings` but are not OCRed
- PDF reading order and chart interpretation remain heuristic
- exact duplicate removal and strict SimHash grouping do not catch every revision
- some very small documents may produce no retained chunks

## Failure Handling

Failed files are written to `ocr_pipeline/parse_failures.log`.

Current full-corpus refresh completed with one remaining failure and produced `23978` cleaned chunks:

- `SIP-20231101-KQKD.docx`
