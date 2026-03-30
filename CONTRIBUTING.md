# Contributing

Thanks for contributing.

## Development Principles

- keep changes targeted and request-scoped
- prefer small, reviewable diffs over large refactors
- do not commit private datasets, model files, local certs, or `.env`
- preserve the repository's RAG-first direction unless a change explicitly targets fine-tuning

## Setup

Recommended baseline:

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install task-specific dependencies as needed:

```bash
python -m pip install -r ocr_pipeline/requirements.txt
python -m pip install -r deployment/app/requirements.txt
python -m pip install -r finetune/requirements.txt
```

## Common Commands

Parser help:

```bash
python ocr_pipeline/process_pdfs.py --help
```

Training help:

```bash
python finetune/train.py --help
```

Bootstrap help:

```bash
python deployment/bootstrap_local.py --help
```

Quick local smoke run:

```bash
python deployment/bootstrap_local.py --ingest-limit 1024
```

Focused tests:

```bash
pytest tests/test_process_pdfs.py tests/test_rag.py -q
```

## Pull Request Checklist

- update docs when commands, defaults, or architecture change
- add or update focused tests for changed behavior
- run at least one relevant validation command before finishing
- call out any remaining manual steps or environment dependencies
