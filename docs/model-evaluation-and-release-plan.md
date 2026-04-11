# Model Evaluation and Release Plan

## Shared Benchmark Framework

### Test Query Categories

| Category | Example Query | What It Tests |
|----------|---------------|---------------|
| Factual grounding | "What were VietCapital Bank\'s key margin metrics in 2023?" | Retrieval + accuracy |
| Comparison | "Compare the NIM trends of ACB vs VCB over the past 3 years" | Multi-document reasoning |
| Temporal reasoning | "How has the NPL ratio trended since 2020?" | Time-series understanding |
| Confident refusal | "What is the capital requirement for a bank in Antarctica?" | No-grounding rejection |

### Benchmark Queries (Held-Out Set)

```text
1. "Summarize the key margin risks for a consumer lender."
2. "What specific capital adequacy requirements apply to Vietnamese commercial banks?"
3. "Compare the funding cost structure of Sacombank vs BIDV."
4. "How has the regulatory environment for digital lending evolved since 2020?"
5. "What are the main credit risk concentrations in the Vietnamese banking sector?"
```

### Evaluation Criteria

Each answer is scored on:

| Criterion | Score Range | Definition |
|-----------|-------------|------------|
| **Grounding** | 0-3 | Answer is supported by retrieved evidence |
| **Accuracy** | 0-3 | Facts stated match known information |
| **Fluency** | 0-2 | Answer is coherent and well-structured |
| **Refusal Quality** | 0-2 | Correct rejection when no evidence (bonus) |

**Pass threshold**: Total score >= 6/10 with Grounding >= 2/3

## Release Gates

| Gate | Qwen 3.5 Requirement | Gemma 4 E4B Requirement |
|------|---------------------|------------------------|
| **Gate 1: Data Clean** | Clean SFT dataset reviewed | Clean SFT dataset reviewed |
| **Gate 2: Training Loss** | Final loss < 1.2 | Final loss < 1.2 |
| **Gate 3: Benchmark Pass** | >= 4/5 queries pass | >= 4/5 queries pass |
| **Gate 4: Provenance** | GGUF from known HF commit | GGUF from known HF commit |
| **Gate 5: Fallback Behavior** | No hallucinated fallbacks | No hallucinated fallbacks |

## Champion / Challenger Decision

After passing all gates:

| Scenario | Decision |
|----------|----------|
| Qwen passes, Gemma fails | Deploy Qwen as champion; Gemma work future |
| Qwen fails, Gemma passes | Deploy Gemma as champion |
| Both pass | Deploy both; champion = Qwen (existing infra), challenger = Gemma |
| Both fail | Iterate on data quality before release |

## Documentation Requirements

For each release, document:

1. Exact base model commit (Hugging Face)
2. Training dataset commit or version
3. Training hyperparameters used
4. Final loss achieved
5. Benchmark scores per query
6. Any deviations from this plan

## Ongoing Monitoring

After deployment, run weekly benchmark queries:

```bash
python deployment/evaluate_live_query.py --output-path deployment/benchmarks/weekly_report.md
```

If champion drops below 4/5 pass rate, trigger Gemma challenger promotion review.
