# A Visual Guide to Reasoning LLMs

**Source:** [Newsletter.maartengrootendorst.com](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms)
**Author:** Maarten Grootendorst
**Date:** February 3, 2025

## Quick Summary

Reasoning LLMs use **test-time compute** to "think longer" through Chain-of-Thought before answering, enabling complex problem-solving.

## Paradigm Shift

### Old: Train-time Compute
- Scale model size, dataset, FLOPs
- Diminishing returns at scale

### New: Test-time Compute  
- Model "thinks" during inference
- Chains reasoning steps before answering
- Balances training with inference

## Techniques

### Search against Verifiers
1. **Majority Voting**: Multiple answers, pick most common
2. **Best-of-N**: Use reward model to score answers
3. **Beam Search + PRM**: Track top-scoring paths
4. **Monte Carlo Tree Search**: Exploration vs exploitation

### Modifying Proposal Distribution
1. **Prompting**: "Let's think step-by-step"
2. **STaR**: Self-taught reasoner (generate reasoning data)
3. **DeepSeek-R1**: RL-based reasoning discovery

## DeepSeek-R1 Case Study

### Training Pipeline
1. **Cold Start**: Fine-tune with high-quality reasoning data
2. **RL for Reasoning**: Group Relative Policy Optimization (GRPO)
3. **Rejection Sampling**: Generate 600K reasoning + 200K non-reasoning samples
4. **SFT**: Supervised fine-tuning with 800K samples
5. **Final RL**: Human preference alignment

### Key Insights
- No verifiers needed for R1-Zero
- Format rewards (\<think\> tags) sufficient
- Distillation to smaller models (Qwen-32B) works well

## Process Reward Model (PRM) vs Outcome Reward Model (ORM)

| Model | What it judges | Use case |
|-------|---------------|----------|
| ORM | Final answer | Best-of-N selection |
| PRM | Each reasoning step | Beam search, path tracking |

## When to Use Reasoning Models

- **Complex math** (proofs, calculations)
- **Coding challenges** (algorithm design)
- **Multi-step logic** (planning, deduction)
- **Code generation** requiring reasoning

## Further Reading

- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948)
- [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314)
- [STaR Paper](https://arxiv.org/abs/2210.03629)

## Related Concepts

- [LLM Agents](./llm-agents.md) - Planning in agents
- [Mixture of Experts](./moe.md) - Efficient large models
