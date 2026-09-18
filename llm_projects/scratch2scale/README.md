# Scratch to Scale: Distributed LLM Training

Progressive guide to distributed training: from a single-GPU baseline that OOMs, through DDP, to FSDP ZeRO-2 and ZeRO-3 — each script shows exactly what breaks at the previous scale and how the next strategy fixes it.

## Objective

Understand distributed training by building up from first principles. Each script is self-contained and runnable — not a tutorial, but working code that demonstrates the memory and communication tradeoffs at each level.

## Project Structure

| Script | Strategy | What it demonstrates |
|---|---|---|
| `00_dataloader_script.py` | Data loading | `DistributedSampler` for sharding data across ranks, tokenization, contiguous windowing |
| `01_DDP_baseline_OOM.py` | DDP (baseline) | Standard `DistributedDataParallel` — works for small models, OOMs when the full model doesn't fit on one GPU |
| `02_FSDP_ZeRO2_Strategy.py` | FSDP ZeRO-2 | Shards optimizer states + gradients across ranks. Model parameters stay replicated. Reduces memory ~4x vs DDP |
| `03_FSDP_ZeRO3_Strategy.py` | FSDP ZeRO-3 | Shards everything: parameters + optimizer states + gradients. All-gather before forward, reduce-scatter after backward |
| `04_FSDP_Advanced.py` | FSDP + mixed precision + activation checkpointing | Combines ZeRO-3 with FP16 compute, BF16 reduce, and selective activation recomputation for maximum memory efficiency |

## Key Concepts

- **Why DDP breaks**: full model replica per GPU — fine for ResNet, OOM for LLMs
- **ZeRO-2 tradeoff**: 4x memory savings, same communication cost as DDP
- **ZeRO-3 tradeoff**: maximum memory savings, but adds all-gather communication before every forward pass
- **Activation checkpointing**: trade compute for memory — recompute activations during backward instead of storing them

## Running

```bash
# Single node, 2 GPUs
torchrun --nproc_per_node=2 02_FSDP_ZeRO2_Strategy.py

# See commands.sh for all configurations
```
