# A Visual Guide to LLM Agents

**Source:** [Newsletter.maartengrootendorst.com](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents)
**Author:** Maarten Grootendorst
**Date:** March 17, 2025

## Quick Summary

LLM Agents extend basic LLMs with **Memory**, **Tools**, and **Planning** capabilities for autonomous task completion.

## Core Components

### 1. Memory
- **Short-term**: Context window, conversation summarization
- **Long-term**: Vector databases, RAG, external storage

### 2. Tools
- **Data retrieval**: Web search, database queries
- **Actions**: Code execution, API calls
- **Function calling**: JSON-structured tool invocation

### 3. Planning
- **Reasoning**: Chain-of-Thought, ReAct
- **Reflection**: Reflexion, SELF-REFINE
- **Goal decomposition**: Task breakdown

## Multi-Agent Systems

Multiple specialized agents collaborate with a supervisor orchestrating communication.

### Frameworks
- **Agent Initialization**: How specialized agents are created
- **Agent Orchestration**: Coordination via supervisor or peer-to-peer

### Examples
- **Generative Agents**: Simulate believable human behavior
- **Modular Frameworks**: Profile, perception, memory, planning, actions

## Model Context Protocol (MCP)

Standardized API access for tools (Anthropic):
- MCP Host → LLM application
- MCP Client → Maintains 1:1 connections
- MCP Server → Provides tools/capabilities

## Tool Use Patterns

1. **Prompting**: Few-shot examples for tool format
2. **Fine-tuning**: Toolformer, ToolLLM, Gorilla
3. **Function Calling**: Native JSON generation

## When to Use Agents

- **Complex tasks** requiring multiple steps
- **Research automation** with tool access
- **Autonomous workflows** (coding, data analysis)
- **Dialogue systems** needing memory

## Further Reading

- [Toolformer Paper](https://arxiv.org/abs/2302.04761)
- [Generative Agents Paper](https://arxiv.org/abs/2304.03442)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)

## Related Concepts

- [Reasoning LLMs](./reasoning-llms.md) - Planning foundations
- [Quantization](./quantization.md) - Deploying agents efficiently
