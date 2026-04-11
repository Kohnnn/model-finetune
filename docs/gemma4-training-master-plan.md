# Gemma 4 Training Master Plan

## Local Feasibility on 4060 Ti 16 GB

| Variant | Params | Memory Est. | Local Training? |
|---------|--------|-------------|-----------------|
| E2B-it | ~2B effective | ~6-8 GB Q4 | ✅ Yes (smoke test) |
| E4B-it | ~4B effective | ~10-12 GB Q4 | ✅ Yes (main challenger) |
| 26B-A4B | 26B total / 4B active | ~14-16 GB Q4 + experts | ⚠️ Tight, may OOM |
| 31B | 31B dense | ~18+ GB Q4 | ❌ No (OOM on 4060 Ti) |

**Recommendation**: Only plan local training for E2B and E4B variants. 26B-A4B is theoretically possible but risky. 31B is not feasible.

## Recommended Variants

| Role | Model | Justification |
|------|-------|---------------|
| **Smoke-test baseline** | `google/gemma-4-E2B-it` | Quick local validation, fast iteration |
| **Main local challenger** | `google/gemma-4-E4B-it` | Competitive with Qwen 3.5 4B, trainable locally |

## Training Phases

### Phase 1: Baseline Evaluation (Before Any Fine-Tuning)

Run both baselines through the shared benchmark (see `docs/model-evaluation-and-release-plan.md`):

```python
# Pseudocode
baselines = [
    ("qwen35-recovered", qwen35_gguf_path),
    ("gemma4-e2b", "google/gemma-4-E2B-it"),
    ("gemma4-e4b", "google/gemma-4-E4B-it"),
]
for name, model_path in baselines:
    results = run_benchmark(model_path, benchmark_queries)
    save_results(name, results)
```

Record baseline scores before any fine-tuning.

### Phase 2: E2B Fine-Tuning (Optional Smoke Test)

```bash
python finetune/train.py \
  --dataset-path finetune/outputs/datasets/qwen35_clean_sft.jsonl \
  --output-dir finetune/outputs/gemma4_e2b_analyst \
  --base-model google/gemma-4-E2B-it \
  --max-seq-length 1024 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --num-epochs 1 \
  --eval-split 0 \
  --save-merged-model \
  --skip-gguf-export
```

### Phase 3: E4B Fine-Tuning (Main Challenger)

```bash
python finetune/train.py \
  --dataset-path finetune/outputs/datasets/qwen35_clean_sft.jsonl \
  --output-dir finetune/outputs/gemma4_e4b_analyst \
  --base-model google/gemma-4-E4B-it \
  --max-seq-length 1024 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --num-epochs 1 \
  --eval-split 0 \
  --save-merged-model \
  --skip-gguf-export
```

### Phase 4: Export and Benchmark

```bash
python finetune/export_gguf.py \
  --model-path finetune/outputs/gemma4_e4b_analyst/merged_model \
  --output-dir finetune/outputs/gemma4_e4b_analyst \
  --gguf-name gemma4_e4b_analyst
```

Run shared benchmark and compare fine-tuned models.

## Benchmark Rules

1. **Always compare apples-to-apples** — run same queries against Qwen and Gemma
2. **Measure grounding, not just fluency** — Gemma may be more eloquent but Qwen may be more grounded in evidence
3. **Use held-out test set** — queries not in training data
4. **Document exact prompts** — reproducibility matters
5. **Report both pass rate and quality scores** — a model might get 80% pass but with lower quality on passed items

## Deployment Path

```
Fine-tuned Gemma E4B
       │
       ▼
Export GGUF (Q4_K_M)
       │
       ▼
Copy to deployment/models/gemma4_e4b_analyst.Q4_K_M.gguf
       │
       ▼
Update deployment/docker-compose.yml to use Gemma instead of Qwen
       │
       ▼
Run bootstrap and benchmark
       │
       ▼
Champion/Challenger decision gate
```

## What NOT To Plan For This Machine

- Local fine-tuning of 26B-A4B (memory will OOM on 4060 Ti)
- Local fine-tuning of 31B (far exceeds 16 GB)
- Training without quantization (would need 40+ GB)
