# OCR Pipeline

This folder contains the local document parsing and chunk generation pipeline.

## Purpose

`process_pdfs.py` scans research files, extracts text, removes common tail boilerplate, drops most disclaimer/contact sections, chunks documents, and writes two JSONL outputs:

- `chroma_chunks.jsonl` for retrieval
- `finetune_template.jsonl` for later supervised fine-tuning

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

## Useful Options

```bash
python ocr_pipeline/process_pdfs.py --help
```

Key arguments:

- `--limit` for pilot runs
- `--sample-mode` and `--seed` for deterministic sampling
- `--chunk-words`, `--overlap-words`, `--min-chunk-words`
- `--trim-tail-pages` for extra tail trimming after the boilerplate detector runs

## Output Schema

### `chroma_chunks.jsonl`

Each row contains:

- `id`
- `text`
- `metadata.source`
- `metadata.relative_source`
- `metadata.doc_id`
- `metadata.title`
- `metadata.year`
- `metadata.language`
- `metadata.file_extension`
- `metadata.chunk_index`
- `metadata.chunk_word_count`

### `finetune_template.jsonl`

Each row contains chat-format messages plus chunk metadata:

- `metadata.source`
- `metadata.relative_source`
- `metadata.doc_id`
- `metadata.title`
- `metadata.chunk_index`
- `metadata.chunk_word_count`
- `system`
- `user`
- `assistant` placeholder

You must fill the assistant completions before running meaningful SFT.

## Current Caveats

- scanned or image-only PDFs are not yet OCRed
- disclaimer stripping is heuristic, not perfect
- some very small documents may produce no retained chunks

## Failure Handling

Failed files are written to `ocr_pipeline/parse_failures.log`.

Current full-corpus refresh completed with one remaining failure and produced `23978` cleaned chunks:

- `SIP-20231101-KQKD.docx`
