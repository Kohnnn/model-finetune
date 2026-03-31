# Release Notes

## v0.1.0 - Private Analyst Stack Milestone

This milestone turns the repository from a scaffold into a working private analyst workflow.

### Highlights

- built a local RAG service with FastAPI, ChromaDB, and llama.cpp
- cleaned the OCR pipeline to remove most disclaimer and contact boilerplate
- trained a full-corpus `unsloth/Qwen3.5-4B` LoRA model on the local RTX 4060 Ti 16GB machine
- published the merged model to a private Hugging Face repository
- exported a deployment-ready GGUF and wired it into the deployment stack
- documented the full engineering process in both a high-level development journal and a detailed training note set

### Included Artifacts

- Hugging Face model: `https://huggingface.co/Mikkkkoooo/qwen35-4b-private-analyst-full-corpus`
- merged local model: `finetune/outputs/qwen35_4b_full_corpus_draft23974/merged_model`
- GGUF: `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.Q4_K_M.gguf`
- deployment model copies:
  - `deployment/models/Qwen3.5-4B.Q4_K_M.gguf`
  - `deployment/models/Qwen3.5-4B.BF16-mmproj.gguf`

### Parse and Training Summary

- supported files processed: `8179 / 8180`
- cleaned chunks: `23978`
- training rows: `23974`
- train runtime: about `18.96h`
- train loss: `1.0765`

### Validation Completed

- parser, training helper, and RAG helper tests passing
- deployment compose config validated
- live `/healthz` and `/query` verified against the new deployment model

### Known Caveats

- the full dataset uses draft-generated completions rather than a fully human-reviewed SFT set
- the live app currently often falls back to extractive evidence when the served model does not return a grounded cited answer
- the source corpus and generated datasets remain private and are not suitable for public release

### Recommended Next Steps

1. build a reviewed gold SFT subset and retrain a higher-quality house-style checkpoint
2. add an evaluation harness over benchmark prompts and compare revisions quantitatively
3. refine prompting and serving so the live app returns fewer fallback-only answers
4. create a tagged release process for future model refreshes
