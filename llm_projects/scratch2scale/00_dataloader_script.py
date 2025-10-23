#!/usr/bin/env python
"""
Dataloader inspection script - examine data distribution across ranks
"""
import os
import math
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer

# Silence warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
        
        print(f"Total tokens: {self.ids.numel()}")
        print(f"Sequence length: {seq_len}")
        print(f"Number of samples: {len(self)}")

    def __len__(self):
        return self.ids.numel() // self.seq_len - 1

    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = self.ids[s:s+self.seq_len]
        y = self.ids[s+1:s+self.seq_len+1]
        return x, y, idx  # Return index for tracking

def setup_dist():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend='nccl',
        device_id=torch.device(f'cuda:{local_rank}')
    )
    
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world, local_rank

def inspect_dataloader():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='Qwen/Qwen2-4B')
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--seq_len', type=int, default=2048)
    ap.add_argument('--global_bs', type=int, default=4)
    ap.add_argument('--num_inspect', type=int, default=3, help='Number of batches to inspect')
    args = ap.parse_args()

    rank, world, local_rank = setup_dist()

    print(f"\n[Rank {rank}] Initializing on GPU {local_rank}")
    print(f"[Rank {rank}] World size: {world}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    if rank == 0:
        print("\n" + "="*80)
        print("DATASET INFORMATION")
        print("="*80)
    
    ds = LineDataset(args.data, tokenizer, args.seq_len)
    
    # Create distributed sampler
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False)  # shuffle=False for inspection
    
    # Create dataloader
    per_rank_bs = math.ceil(args.global_bs / world)
    dl = DataLoader(
        ds, 
        batch_size=per_rank_bs, 
        sampler=sampler,
        num_workers=0,  # Set to 0 for easier debugging
        pin_memory=False,
        drop_last=True
    )

    print(f"\n[Rank {rank}] Per-rank batch size: {per_rank_bs}")
    print(f"[Rank {rank}] Number of batches: {len(dl)}")

    # Inspect first few batches
    print(f"\n[Rank {rank}] " + "="*60)
    print(f"[Rank {rank}] INSPECTING FIRST {args.num_inspect} BATCHES")
    print(f"[Rank {rank}] " + "="*60)

    for step, (x, y, indices) in enumerate(dl):
        if step >= args.num_inspect:
            break
        
        print(f"\n[Rank {rank}] --- Batch {step} ---")
        print(f"[Rank {rank}] Batch shape: {x.shape}")
        print(f"[Rank {rank}] Sample indices in this batch: {indices.tolist()}")
        
        # Decode and show first sample in batch
        if x.shape[0] > 0:
            first_x = x[0]
            first_y = y[0]
            first_idx = indices[0].item()
            
            print(f"\n[Rank {rank}] Sample index {first_idx}:")
            print(f"[Rank {rank}] Input tokens (first 20): {first_x[:20].tolist()}")
            print(f"[Rank {rank}] Label tokens (first 20): {first_y[:20].tolist()}")
            
            # Decode to text
            input_text = tokenizer.decode(first_x[:50], skip_special_tokens=True)
            label_text = tokenizer.decode(first_y[:50], skip_special_tokens=True)
            
            print(f"\n[Rank {rank}] Input text (first 50 tokens):")
            print(f"[Rank {rank}] '{input_text}'")
            print(f"\n[Rank {rank}] Label text (first 50 tokens):")
            print(f"[Rank {rank}] '{label_text}'")
            
            # Verify that label is input shifted by 1
            print(f"\n[Rank {rank}] Verification - are labels = inputs shifted by 1?")
            print(f"[Rank {rank}] Input[1:21]:  {first_x[1:21].tolist()}")
            print(f"[Rank {rank}] Label[0:20]:  {first_y[0:20].tolist()}")
            print(f"[Rank {rank}] Match: {torch.equal(first_x[1:21], first_y[0:20])}")

    # Summary across all ranks
    dist.barrier()
    if rank == 0:
        print("\n" + "="*80)
        print("INSPECTION COMPLETE")
        print("="*80)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    inspect_dataloader()