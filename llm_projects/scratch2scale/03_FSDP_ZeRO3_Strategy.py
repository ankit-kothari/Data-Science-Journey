#!/usr/bin/env python
"""
FSDP FULL_SHARD (ZeRO-3 semantics)
Only memory (VRAM) logs are printed. Other logs are silenced.
"""

import os, math, argparse
import functools
import warnings
import logging
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

# ----------------- Env + logging silencing -----------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# mute tqdm / HF bars
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# optional: reduce NCCL chatter
os.environ.setdefault("NCCL_DEBUG", "WARN")
# optional: reduce torch distributed logs
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")

# Silence HF transformers logs
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Silence PyTorch FSDP logger that prints the huge param list
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.fsdp").setLevel(logging.ERROR)

# Silence specific warnings
warnings.filterwarnings("ignore", message=r".*torch_dtype is deprecated.*")
warnings.filterwarnings("ignore", message=r".*barrier\(\): using the device under current context.*")

# ---- optional direct import for Qwen layer; dynamic fallback otherwise ----
DECODER_LAYER_CLS = None
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as _Qwen2DecoderLayer
    DECODER_LAYER_CLS = _Qwen2DecoderLayer
except Exception:
    try:
        from transformers.models.qwen import QWenBlock as _QWenBlock
        DECODER_LAYER_CLS = _QWenBlock
    except Exception:
        pass

# ----------------- Data -----------------
class LineDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        toks = tokenizer(text, return_tensors=None)["input_ids"]
        self.ids = torch.tensor(toks, dtype=torch.long)
        n_full = (self.ids.numel() // seq_len) * seq_len
        self.ids = self.ids[:n_full]
        self.seq_len = seq_len

    def __len__(self):
        return self.ids.numel() // self.seq_len - 1

    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = self.ids[s : s + self.seq_len]
        y = self.ids[s + 1 : s + self.seq_len + 1]
        return x, y

# ----------------- Dist -----------------
def setup_dist():
    if not dist.is_initialized():
        # pass device_id to avoid the barrier() warning
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank

@torch.no_grad()
def vram(msg):
    cur = torch.cuda.memory_allocated() / 1024**3
    mx = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[rank {dist.get_rank()}] VRAM {msg}: cur={cur:.2f} GB, max={mx:.2f} GB", flush=True)

def discover_decoder_layer_cls(model):
    for name, module in model.named_modules():
        if ".layers." in name:
            cn = module.__class__.__name__.lower()
            if cn.endswith(("decoderlayer", "block", "transformerlayer")):
                return module.__class__
    return None

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2-4B")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--global_bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--no-layer-wrap", action="store_true",
                    help="Disable per-layer auto-wrap (use size-based policy instead).")
    ap.add_argument("--wrap-min-params", type=int, default=10_000_000)
    ap.add_argument("--mem-log-every", type=int, default=5,
                    help="Log VRAM every N steps (1 = every step).")
    args = ap.parse_args()

    rank, world, local_rank = setup_dist()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model load: use dtype= to avoid deprecation
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)

    # mixed precision
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)

    # wrap policy
    block_cls = DECODER_LAYER_CLS or discover_decoder_layer_cls(base)
    if not args.no_layer_wrap and block_cls is not None:
        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={block_cls})
    else:
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=args.wrap_min_params)

    # FSDP construct
    model = FSDP(
        base.cuda(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp,
        device_id=local_rank,
        use_orig_params=True,
    )

    vram("after model+optimizer construction")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = LineDataset(args.data, tokenizer, args.seq_len)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(
        ds,
        batch_size=math.ceil(args.global_bs / world),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    try:
        model.train()
        for step, (x, y) in enumerate(dl):
            if step >= args.steps:
                break
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            out = model(x, labels=y)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            # Memory logging cadence only
            if args.mem_log_every > 0 and (step % args.mem_log_every == 0):
                vram(f"after step {step}")

        vram("end of run")
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
