# A Visual Guide to Quantization

**Source:** [Newsletter.maartengrootendorst.com](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
**Author:** Maarten Grootendorst
**Date:** July 22, 2024

## Quick Summary

Quantization compresses models from **high precision (FP32)** to **low precision (INT8/INT4)** for reduced memory and faster inference with minimal accuracy loss.

## The Problem

70B parameter model at FP32 = **280GB VRAM** (impossible for most hardware)

## Data Types

| Type | Bits | Use case |
|------|------|----------|
| FP32 | 32 | Full precision |
| FP16 | 16 | Half precision (standard) |
| BF16 | 16 | Better range than FP16 |
| INT8 | 8 | Common quantization target |
| INT4 | 4 | Aggressive compression |

## Quantization Methods

### Symmetric (absmax)
- Range centered at zero
- Formula: `x_quant = round(x / s)`
- Scale: `s = max(|x|) / 127`

### Asymmetric (zero-point)
- Range shifted from zero
- Includes zero-point offset
- Better for activations with asymmetry

## Key Challenge: Outliers

Large outlier values reduce precision for most values:
- **Clipping**: Set dynamic range manually
- **Calibration**: Find optimal range (percentile, MSE, KL-divergence)

## Weight vs Activation Quantization

| Component | When known | Method |
|-----------|-----------|--------|
| Weights | Static (before inference) | Per-layer calibration |
| Activations | Dynamic (during inference) | Dynamic or static quantization |

## Post-Training Quantization (PTQ)

### Dynamic Quantization
- Calculate scale/zeropoint per layer during inference
- More accurate, slightly slower

### Static Quantization  
- Use calibration dataset beforehand
- Pre-compute scales
- Faster inference, less accurate

## Popular Methods

### GPTQ
- Layer-by-layer processing
- Uses inverse-Hessian for error weighting
- Good for full GPU deployment

### GGUF (llama.cpp)
- CPU + GPU offloading
- Super/sub block quantization
- Many precision levels (Q2_K, Q4_K, Q5_K, Q8_0)

### AWQ, EXL2, BitNet
- Various optimizations
- 1-bit models (BitNet)
- Performance-focused

## Quantization Aware Training (QAT)

- "Fake" quantization during training
- Explores wide minima (lower quantization error)
- Better accuracy than PTQ
- Adds training cost

## Practical Guide

| Hardware | Recommended | Why |
|----------|-------------|-----|
| High-end GPU | GPTQ Q4/Q5 | Fast, good accuracy |
| Consumer GPU | GGUF Q4_K_M | CPU offload available |
| CPU only | GGUF Q5_K_M | Balance speed/quality |
| On-device | INT4 + QAT | Maximum compression |

## Memory Calculator

```
70B params × 4 bytes (INT32) = 280 GB
70B params × 2 bytes (INT16) = 140 GB  
70B params × 1 byte (INT8)  = 70 GB
70B params × 0.5 byte (INT4) = 35 GB
```

## Further Reading

- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [BitNet Paper](https://arxiv.org/abs/2310.11453)

## Related Concepts

- [Mixture of Experts](./moe.md) - Sparse params need quantization
- [Mamba](./mamba.md) - Often quantized for deployment
