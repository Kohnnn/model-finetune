# AGENTS Guide
Instructions for agentic coding assistants in this repository.

## Project Context
- Goal: private AI analyst stack (OCR -> fine-tune -> RAG app -> OCI deployment).
- State: scaffolded code; key modules contain placeholders.
- Key paths:
  - `ocr_pipeline/process_pdfs.py`
  - `finetune/train.py`
  - `deployment/app/main.py`
  - `deployment/docker-compose.yml`
  - `FINE_TUNING_GUIDE.md`

## Rule Files Check
No additional rule files were found:
- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`
If these appear later, treat them as higher-priority policy.

## Environment Setup
Use Python 3.10+ (3.11 preferred).

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

No pinned dependency file is present; install task-specific dependencies as needed.

## Build and Run
Run OCR pipeline:

```bash
python ocr_pipeline/process_pdfs.py
```

Run fine-tuning entrypoint:

```bash
python finetune/train.py
```

Run app entrypoint:

```bash
python deployment/app/main.py
```

Start stack:

```bash
docker compose -f deployment/docker-compose.yml up -d
```

Stop stack:

```bash
docker compose -f deployment/docker-compose.yml down
```

## Lint / Format / Types
No lint/type config files were found (`pyproject.toml`, `ruff.toml`, `mypy.ini`, etc.).
Use this baseline unless the task adds repo-specific config:

```bash
ruff check .
ruff format .
mypy ocr_pipeline finetune deployment/app
```

## Testing Commands
Current state: no `tests/` directory exists.
When tests are added, use `pytest` and run the narrowest command first.

Run all tests:

```bash
pytest -q
```

Run one test file:

```bash
pytest tests/test_<module>.py -q
```

Run one test function:

```bash
pytest tests/test_<module>.py::test_<name> -q
```

Run tests by keyword:

```bash
pytest -q -k "keyword"
```

## Code Style

### Imports
- Use absolute imports where practical.
- Group order: standard library, third-party, local modules.
- Keep imports sorted; remove unused imports.
- Never use wildcard imports.

### Formatting
- Follow PEP 8 with Black/Ruff-compatible style (88-char target).
- Keep functions focused; extract helpers instead of deep nesting.
- Add comments only for non-obvious reasoning.
- Keep diffs small and request-scoped.

### Typing
- Add type hints to all new/changed function signatures.
- Prefer precise types (`list[str]`) over broad types.
- Use `TypedDict`/`dataclass` for structured objects when useful.
- Avoid implicit `None` returns in non-optional APIs.

### Naming
- `snake_case`: variables, functions, modules.
- `PascalCase`: classes.
- `UPPER_SNAKE_CASE`: constants and env vars.
- Keep domain terms consistent (`chunk`, `context`, `completion`, `adapter`, `gguf`).

### Error Handling
- Validate inputs early and fail fast.
- Do not swallow exceptions.
- Catch specific exception classes, not bare `except`.
- Provide actionable error context (path/id/operation).
- Prefer logging in runtime/library code; `print` is fine for script entrypoints.

### Configuration and Secrets
- Read settings from environment variables.
- Never hardcode secrets, tokens, or private endpoints.
- Keep safe local defaults where possible.
- Document any new env vars in docs or module headers.

### Data and I/O
- Use UTF-8 text encoding.
- JSONL outputs must be one JSON object per line.
- Use deterministic ordering when iterating files.
- Prefer streaming for large inputs.

## Testing Expectations for Changes
- Add or update tests for logic changes (create `tests/` if needed).
- For bug fixes, add a regression test.
- For integration-heavy changes (OCR/training/docker), pair mocked unit tests with one manual verification command.
- Before finishing, run at least one relevant command (focused test, lint, or type check).

## Agent Discipline
- Make minimal targeted edits; avoid unrelated refactors.
- Do not commit secrets or large generated artifacts unless explicitly requested.
- If you add tooling config, keep defaults minimal and conventional.
- In conflicts, follow explicit user instructions first, then this file.
