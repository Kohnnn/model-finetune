# Fine-Tuning Pipeline

This folder contains the Windows GPU Qwen 3.5/4B training path and supporting dataset-prep utilities.

## Purpose

Use this stage only after the retrieval system is working. RAG should carry factual knowledge first; fine-tuning should refine analyst tone, structure, and response style.

## Install

Recommended on this machine:

```powershell
./finetune/setup_gpu_env.ps1
```

Manual equivalent:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
.venv\Scripts\python.exe -m pip install -r finetune/requirements.txt
.venv\Scripts\python.exe -m pip install --upgrade "git+https://github.com/unslothai/unsloth.git" "git+https://github.com/unslothai/unsloth-zoo.git"
```

Recommended runtime:

- Python 3.11
- RTX 4060 Ti 16GB or better
- CUDA-enabled PyTorch from the `cu128` wheel index

Current local status:

- Cleaned full-corpus draft dataset generated at `finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl`
- Successful full Qwen 3.5 adapter generated at `finetune/outputs/qwen35_4b_full_corpus_draft23974/adapter`
- Successful merged Qwen 3.5 artifact generated at `finetune/outputs/qwen35_4b_full_corpus_draft23974/merged_model`
- Successful GGUF export generated at `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.Q4_K_M.gguf`
- Full-run summary generated at `finetune/outputs/qwen35_4b_full_corpus_draft23974/training_summary.json`
- Private Hugging Face repo uploaded at `Mikkkkoooo/qwen35-4b-private-analyst-full-corpus`

## Dataset Requirement

Default dataset path:

- `ocr_pipeline/finetune_template.jsonl`

Important:

- the template file is not training-ready until assistant completions are filled
- `train.py` blocks empty assistant-only datasets unless `--allow-empty-assistant` is used
- using `--allow-empty-assistant` is only useful for validation, not real training
- `prepare_seed_dataset.py` can create a synthetic bootstrap dataset when you need a local seed run

## Usage

Inspect options:

```bash
python finetune/train.py --help
```

Dry-run validation:

```bash
python finetune/train.py --dry-run --allow-empty-assistant --max-samples 10
```

Example smoke training command after labeling:

```bash
python finetune/train.py \
  --dataset-path ocr_pipeline/finetune_template.jsonl \
  --max-samples 128 \
  --num-epochs 0.2 \
  --skip-gguf-export
```

Full-corpus draft dataset generation:

```bash
python finetune/prepare_seed_dataset.py \
  --input-path ocr_pipeline/finetune_template.jsonl \
  --output-path finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl \
  --max-rows 1000000 \
  --max-context-words 450
```

Qwen 3.5/4B full-corpus run completed on this machine:

```bash
python finetune/train.py \
  --dataset-path finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl \
  --output-dir finetune/outputs/qwen35_4b_full_corpus_draft23974 \
  --max-seq-length 1024 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --num-epochs 1 \
  --eval-split 0 \
  --log-steps 100 \
  --save-steps 500 \
  --warmup-steps 100 \
  --save-merged-model \
  --skip-gguf-export \
  --disable-response-only-masking
```

Hugging Face upload helper:

```bash
python finetune/push_to_huggingface.py \
  --model-dir finetune/outputs/qwen35_4b_full_corpus_draft23974/merged_model \
  --repo-id Mikkkkoooo/qwen35-4b-private-analyst-full-corpus \
  --private
```

GGUF export helper:

```bash
python finetune/export_gguf.py \
  --model-path finetune/outputs/qwen35_4b_full_corpus_draft23974/adapter \
  --output-dir finetune/outputs/qwen35_4b_full_corpus_draft23974 \
  --gguf-name qwen3_5_4b_private_analyst_full_corpus_q4_k_m
```

Current artifact sizes on this machine:

- Qwen 3.5 full adapter: about `0.10 GB`
- Qwen 3.5 full merged model: about `8.70 GB`
- Qwen 3.5 GGUF export directory: about `3.15 GB`

## Outputs

Training artifacts are written under `finetune/outputs/`:

- `checkpoints/`
- `adapter/`
- `gguf/`
- `datasets/`
- `qwen35_4b_full_corpus_draft23974/`

## Recommended Workflow

1. validate the RAG MVP first
2. generate the `qwen35_full_corpus_draft.jsonl` draft set
3. manually review the highest-value rows if you want a higher-quality follow-up run
4. run the Qwen 3.5 4B training command on the reviewed or full-corpus file
5. save adapter + merged Hugging Face artifact
6. export GGUF for deployment when you are happy with the model

## Notes

- current deployment is model-agnostic as long as `LLAMA_MODEL_FILENAME` matches the GGUF you provide
- if you change model family or quantization, update `deployment/.env`
- `FINE_TUNING_GUIDE.md` provides the higher-level workflow narrative
- `QWEN35_TRAINING_NOTES.md` captures the full troubleshooting and execution path used here
- `HF_TOKEN` is expected to be stored locally in the user environment, not committed to the repo
- the current Hugging Face repo is private because the source corpus is private
- the current GGUF export includes the quantized model and the mmproj companion file required by this Qwen 3.5 stack
