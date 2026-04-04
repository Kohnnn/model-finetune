# A Visual Guide to Mixture of Experts

**Source:** [Newsletter.maartengrootendorst.com](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
**Author:** Maarten Grootendorst
**Date:** October 7, 2024

## Quick Summary

Mixture of Experts (MoE) uses **multiple specialized sub-models (experts)** with a **router** to selectively activate only some experts per token, enabling massive models with efficient inference.

## Core Components

### Experts
- FFNN sub-layers within each layer
- Each expert learns **syntax-level patterns** (not domains)
- Not specialized by topic, but by token patterns

### Router (Gate Network)
- FFNN that scores experts per input
- Selects top-k experts to activate
- Must balance load to prevent expert starvation

## Dense vs Sparse

| Type | Behavior | Compute |
|------|----------|---------|
| Dense | All parameters activated | High |
| Sparse | Only selected experts | Reduced |

## Key Challenges

### Load Balancing
- **Auxiliary Loss**: Penalizes uneven expert usage
- **Expert Capacity**: Limits tokens per expert
- **Token Overflow**: Excess tokens routed to next layer

### Expert Routing
- **Top-1**: Single expert per token (Switch Transformer)
- **Top-K**: K experts per token (Mixtral 8x7B)
- **Soft MoE**: Weighted mixture instead of hard selection

## Example: Mixtral 8x7B

```
Total parameters: 46.7B (8 × 5.6B experts + shared)
Active parameters: 12.8B (2 × 5.6B experts)
Load: Must fit 46.7B, runs at 12.8B speed
```

## Expert Specialization

Research shows experts specialize by **syntax** (token patterns), not by domain:
- Certain experts handle punctuation, keywords
- Routing creates emergent specialization
- Similar tokens route to similar experts

## Vision-MoE

MoE extends to vision models (ViT):
- **V-MoE**: Sparse MoE in encoder
- **Soft-MoE**: Continuous token mixing
- Patch routing instead of token routing

## When to Use MoE

- **Massive models** (100B+ parameters)
- **Efficiency-critical** deployments
- **Multi-modal** architectures
- **On-device** with clever quantization

## Further Reading

- [Mixtral 8x7B Paper](https://arxiv.org/abs/2401.04088)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [ST-MoE](https://arxiv.org/abs/2202.08906)

## Related Concepts

- [Gemma 4](../assets/gemma4/) - Uses MoE in 26B A4B variant
- [Quantization](./quantization.md) - Handling sparse params
