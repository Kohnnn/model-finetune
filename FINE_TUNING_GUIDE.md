# Comprehensive Guide: Fine-Tuning and Deploying Your Private AI System

This guide explains the complete end-to-end lifecycle of your private AI data analysis system: from understanding fine-tuning concepts to processing your PDF dataset, executing the fine-tune on your local RTX 4060 Ti, and finally deploying the resulting model onto your Oracle Cloud Infrastructure (OCI) server.

---

## 1. What is Fine-Tuning?

Large Language Models (LLMs) like Qwen are trained on vast amounts of general internet text. Out of the box, they are "generalists." 

**Fine-Tuning (specifically Supervised Fine-Tuning or SFT)** is the process of teaching the model exactly *how* you want it to behave. You do this by providing hundreds of examples of ideal interactions (Prompts + Expected Completions). 

**What Fine-Tuning Does:**
- Teaches the model your unique professional tone (e.g., writing like a financial analyst).
- Enforces strict output formatting (e.g., returning JSON, adhering to specific bullet-point structures).
- Specializes the model's reasoning process to your domain.

**What Fine-Tuning Does NOT Do:**
- It does **not** inject new factual knowledge effectively. That is what your **RAG (ChromaDB)** system is for. 

**How you can fit it on an RTX 4060 Ti (16GB):**
Normally, training a 4-Billion parameter model requires massive memory. We use a technique called **QLoRA** (Quantized Low-Rank Adaptation):
1. **Quantization:** The base model is squashed down to 4-bit precision, reducing its VRAM footprint from ~9GB to ~3.5GB.
2. **LoRA:** Instead of changing all 4 Billion parameters, we freeze the base model and bolt on tiny "adapter" layers. We only train these tiny layers. This requires very little VRAM, allowing you to train seamlessly on 16GB.

---

## 2. How to Process Your Dataset

Your goal is to parse your research reports (PDFs, Word Docs, PowerPoints) and generate two parallel datasets. We use **100% local, lightweight, open-source Python libraries** for extraction to bypass API keys and complex dependencies.

### Step A: Extraction & Chunking
Your `ocr_pipeline/process_pdfs.py` script will iterate through your `raw_dataset` folder.
- **PyMuPDF / python-docx / python-pptx** will load the documents automatically.
- The script now removes many disclaimer/contact sections using explicit English and Vietnamese boilerplate markers before chunking.
- The processed text is split into chunks of **800 words**, with a **100-word overlap** so context isn't lost at the edges.

### Step B: Generating the Two Pipelines
For every text chunk parsed:

1. **`chroma_chunks.jsonl` (The RAG Database)**
   - These are just raw factual text chunks. This data is dumped straight into ChromaDB for factual retrieval at query time.
   
2. **`finetune_template.jsonl` (The Training Data)**
    - The script auto-generates the "Prompt" half of the data, representing how a user might ask a question based on that chunk.
    - The file also includes metadata per chunk so you can trace draft examples back to the source report.
    - **Recommended Job:** generate a draft review set, then manually review the best 300 to 500 examples before the full production fine-tune.
   
**Example Format for `finetune_template.jsonl`:**
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert financial analyst. Answer purely based on the given context."},
    {"role": "user", "content": "Context: [Chunk about DXG Q3 earnings...] \n\nQuery: Summarize DXG's margin compression."},
    {"role": "assistant", "content": "Driven by rising material costs, DXG experienced a 200 bps margin compression in Q3, specifically localized in..."}
  ]
}
```

---

## 3. How to Fine-Tune (The Unsloth Pipeline)

Once you have ~300 high-quality rows in `finetune_template.jsonl`, you are ready to train using your `finetune/train.py` script. We use **Unsloth**, a library that makes LoRA training 2x faster and uses 70% less memory.

For this repository, there is now also a CPU-only bootstrap path:

- `finetune/prepare_seed_dataset.py` creates a synthetic seed SFT dataset from the OCR template file.
- `finetune/train_cpu_lora.py` runs a smaller CPU LoRA fine-tune against `Qwen/Qwen2.5-0.5B-Instruct`.
- `finetune/push_to_huggingface.py` uploads the resulting adapter or merged model to the Hugging Face Hub.

This CPU path is suitable for local experimentation and Hub seeding, but the intended production workflow is still the GPU-based Unsloth run.

### Step A: Load the Model
The current validated local path uses `unsloth/Qwen3.5-4B` with `load_in_4bit=True` on your RTX 4060 Ti 16GB.

### Step B: Attach LoRA Adapters
The script applies the LoRA parameters:
- `r=16` and `lora_alpha=16` (standard values balancing capability and speed).
- We target specific neural network layers (`q_proj`, `k_proj`, `v_proj`, etc.).

### Step C: Execute Training
The current local workflow is:

1. Build a draft review set:

```bash
python finetune/prepare_seed_dataset.py \
  --input-path ocr_pipeline/finetune_template.jsonl \
  --output-path finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl \
  --max-rows 1000000 \
  --max-context-words 450
```

2. Run a smoke fine-tune on reviewed or draft data:

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

3. After manual review, rerun the same command on a more human-reviewed dataset if you want a stronger house-style checkpoint.

Validated full-corpus command used in this repository:

```bash
python finetune/train.py \
  --dataset-path finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl \
  --output-dir finetune/outputs/qwen35_4b_full_corpus_draft23974 \
  --max-seq-length 1024 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --num-epochs 1 \
  --eval-split 0 \
  --save-merged-model \
  --skip-gguf-export \
  --disable-response-only-masking
```

See `finetune/QWEN35_TRAINING_NOTES.md` for the troubleshooting log and exact runtime results.

Validated GGUF export helper:

```bash
python finetune/export_gguf.py \
  --model-path finetune/outputs/qwen35_4b_full_corpus_draft23974/adapter \
  --output-dir finetune/outputs/qwen35_4b_full_corpus_draft23974 \
  --gguf-name qwen3_5_4b_private_analyst_full_corpus_q4_k_m
```

If you are on a CPU-only machine, use the smaller fallback path instead of forcing the Unsloth script.

### Step D: GGUF Export
When training completes, Unsloth automatically merges your newly trained LoRA adapters into the base model and exports everything as a single `.gguf` file optimized for CPU execution.
- We apply `quantization_method="q4_k_m"`, shrinking the final deployed file to ~3GB.

---

## 4. How to Deploy the Model to OCI

Now that you have your `qwen3.5-4b-instruct-q4_k_m.gguf` file, it's time to move to your Oracle Cloud server (4 Arm64 OCPUs, 24 GB RAM, No GPU).

### Step A: Transfer the Model
Use `scp` or `rsync` to upload the `.gguf` file to the `./deployment/models/` folder on your OCI server.

### Step B: Understanding the Docker Compose Architecture
Your `docker-compose.yml` ties the system together entirely internally:

1. **Llama.cpp Server (`llama`)**
   - Llama.cpp is uniquely optimized to run LLMs on CPUs (specifically Arm64).
   - It boots up your `.gguf` model and exposes an OpenAI-compatible API exactly like ChatGPT (`http://llama:8080/v1`).
   - Hard capped to 3GB RAM.
   
2. **ChromaDB (`chromadb`)**
   - Holds your `chroma_chunks.jsonl` data.
   - Handles the vector math for similarity search.

3. **Your Custom App (`app`)**
   - Exists to process user requests.
   - When a user asks a question, the App:
     1. Queries ChromaDB for the Top 3 relevant 800-word chunks.
     2. Packages the User's Query + the 3 Chunks into a prompt.
     3. Sends the prompt over the internal network to `llama.cpp`.
     4. `llama.cpp` generates an answer (in the tone it learned during fine-tuning) based purely on the facts retrieved by Chroma.

4. **Nginx Reverse Proxy (`nginx`)**
   - The only container exposed to the outside internet (Port 443).
   - Secures traffic using SSL and forwards it to your Python app.
   - The Llama and Chroma endpoints remain completely invisible to hackers.

### Step C: Spin it Up
On your OCI VM, you simply run:
```bash
docker-compose up -d
```
Your private, fine-tuned, RAG-enabled AI analyst is now live.
