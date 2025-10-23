#!/usr/bin/env python
"""
FSDP ZeRO‑3 + *Staff‑level* optimizations:
1) **Activation Checkpointing** → compute↔memory trade‑off; lowers VRAM by re‑computing forward during backward.
2) **auto_wrap_policy** that wraps *transformer blocks* (e.g., Qwen decoder layers) instead of the whole model.
   - This increases comm/compute overlap and shrinks per‑unit all‑gather spikes.

Notes:
- We try to detect the block class dynamically (Qwen2DecoderLayer / Qwen2Block). If unknown, fall back to a param‑count threshold policy.
- Keep the hand‑rolled training loop; only distribution strategy changes.
"""
import os, math, argparse, inspect
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed.fsdp import FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing, CheckpointImpl
)
import torch.nn as nn

class LineDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        toks = tokenizer(text, return_tensors=None)['input_ids']
        self.ids = torch.tensor(toks, dtype=torch.long)
        n_full = (self.ids.numel() // seq_len) * seq_len
        self.ids = self.ids[:n_full]
        self.seq_len = seq_len
    def __len__(self):
        return self.ids.numel() // self.seq_len - 1
    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = self.ids[s:s+self.seq_len]
        y = self.ids[s+1:s+self.seq_len+1]
        return x, y

def setup_dist():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank(); world = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank

@torch.no_grad()
def vram(msg):
    cur = torch.cuda.memory_allocated() / 1024**3
    mx = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{dist.get_rank()}] VRAM {msg}: cur={cur:.2f} GB, max={mx:.2f} GB", flush=True)

def guess_qwen_block_cls(model: nn.Module):
    # Try to locate a typical Qwen block type by name
    candidates = []
    for m in model.modules():
        cls = m.__class__.__name__
        if 'DecoderLayer' in cls or 'Block' in cls:
            # Heuristic: transformer block has attn & MLP children
            has_attn = any('attn' in c.__class__.__name__.lower() or 'attention' in c.__class__.__name__.lower() for c in m.modules())
            has_mlp  = any('mlp' in c.__class__.__name__.lower() or 'ffn' in c.__class__.__name__.lower() for c in m.modules())
            if has_attn and has_mlp:
                candidates.append(m.__class__)
    return candidates[0] if candidates else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='Qwen/Qwen2-4B')
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--seq_len', type=int, default=2048)
    ap.add_argument('--global_bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--steps', type=int, default=20)
    args = ap.parse_args()

    rank, world, local_rank = setup_dist()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model)

    # Detect transformer block class if possible; else fallback to param_size threshold
    block_cls = guess_qwen_block_cls(base)
    if block_cls is not None:
        policy = transformer_auto_wrap_policy(
            {block_cls},
            transformer_layer_cls={block_cls},
        )
        if rank == 0:
            print(f"[auto_wrap] Using detected block class: {block_cls.__name__}")
    else:
        if rank == 0:
            print("[auto_wrap] Could not detect block class; using size-based policy (~60M params per wrap)")
        def policy(module, recurse, nonwrapped_numel):
            return nonwrapped_numel >= 60_000_000

    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)

    # Wrap base with FSDP ZeRO‑3 and auto‑wrap policy (block‑wise FSDP units)
    model = FSDP(
        base.cuda(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=local_rank,
        auto_wrap_policy=policy,
    )

    # === Activation Checkpointing ===
    # Wrap matching submodules so activations are recomputed in backward instead of stored.
    check_fn = None
    if block_cls is not None:
        def check_fn(m):
            return isinstance(m, block_cls)
    else:
        # conservative: checkpoint any module with many params (heuristic)
        def check_fn(m):
            return sum(p.numel() for p in m.parameters(recurse=False)) >= 20_000_000

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=check_fn,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = LineDataset(args.data, tokenizer, args.seq_len)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=math.ceil(args.global_bs/world), sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)

    vram('after model+optimizer construction (ZeRO‑3 + ckpt + auto_wrap)')

    model.train()
    for step, (x, y) in enumerate(dl):
        if step >= args.steps: break
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        out = model(x, labels=y)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 5 == 0 and rank == 0:
            print({"step": step, "loss": float(loss)}, flush=True)

    vram('end of run (ZeRO‑3 + ckpt + auto_wrap)')

if __name__ == '__main__':
    main()