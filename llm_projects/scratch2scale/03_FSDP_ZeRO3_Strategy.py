#!/usr/bin/env python
"""
FSDP with ShardingStrategy.FULL_SHARD → **ZeRO‑3** semantics.
All three: params + grads + optimizer states are sharded across ranks.

Important dynamics:
- During forward pass, each FSDP unit performs a **just‑in‑time all‑gather** of its parameters,
  computes, then **reshards/frees** them. Peak per‑layer spikes are short‑lived; steady‑state VRAM is lowest.
- Backward shards grads; optimizer states live sharded too.
"""
import os, math, argparse
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed.fsdp import FSDP, ShardingStrategy, MixedPrecision

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

    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)

    # FULL_SHARD ~ ZeRO‑3: params+grads+opt states are sharded
    model = FSDP(
        base.cuda(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=local_rank,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = LineDataset(args.data, tokenizer, args.seq_len)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=math.ceil(args.global_bs/world), sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)

    vram('after model+optimizer construction (ZeRO‑3)')

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

    vram('end of run (ZeRO‑3)')

if __name__ == '__main__':
    main()