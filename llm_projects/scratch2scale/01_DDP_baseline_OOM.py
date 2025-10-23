#!/usr/bin/env python
"""
DDP baseline for Qwen‑3 4B that will likely OOM on 4×A100 40GB with realistic settings.
Purpose: make the *replication problem* concrete.

Key ideas:
- Each rank loads the *entire* model (params) + keeps grads + keeps Adam states (≈2× params in FP32) → huge.
- Data is split by DistributedSampler. Training loop is hand‑written.
- Intentionally uses a moderately large seq_len/global_bs to trigger OOM.

If you don't OOM immediately, bump --seq_len or --global_bs until you do. Watch nvidia‑smi.
"""
import os, math, argparse, time, json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

class LineDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        toks = tokenizer(text, return_tensors=None)['input_ids']
        # Make a simple contiguous stream → fixed windows
        self.ids = torch.tensor(toks, dtype=torch.long)
        self.seq_len = seq_len
        # drop tail
        n_full = (self.ids.numel() // seq_len) * seq_len
        self.ids = self.ids[:n_full]

    def __len__(self):
        return self.ids.numel() // self.seq_len - 1

    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = self.ids[s:s+self.seq_len]
        y = self.ids[s+1:s+self.seq_len+1]
        return x, y

def setup_dist():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend='nccl',
        device_id=torch.device(f'cuda:{local_rank}')  # Specify device explicitly
    )
    
    rank = dist.get_rank()
    world = dist.get_world_size()
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

    # Load model in FP16 and move to correct GPU
    # Load model sequentially to avoid all ranks hitting GPU 0 at once
    if rank == 0:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
        model = model.to(f'cuda:{local_rank}')
    dist.barrier()  # Wait for rank 0 to finish

    if rank != 0:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
        model = model.to(f'cuda:{local_rank}')
    dist.barrier()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Adam keeps two FP32 moment buffers per param → big multiplier
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = LineDataset(args.data, tokenizer, args.seq_len)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=math.ceil(args.global_bs/world), sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)

    vram('after model+optimizer construction')

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    for step, (x, y) in enumerate(dl):
        if step >= args.steps: break
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        out = model(x, labels=y)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 5 == 0 and rank == 0:
            print({"step": step, "loss": loss.detach().item()}, flush=True)

    vram('end of run')
    # Clean shutdown
    dist.destroy_process_group()

if __name__ == '__main__':
    main()


