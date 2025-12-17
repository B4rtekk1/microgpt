"""
Distributed Training Script for 8xB200 with SFT, Math & Code data.

Usage:
    torchrun --nproc_per_node=8 train_distributed.py --stage pretrain --hours 3
    torchrun --nproc_per_node=8 train_distributed.py --stage sft --hours 1
    torchrun --nproc_per_node=8 train_distributed.py --stage full --hours 5
"""

import os, math, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from functools import partial

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from layers.rmsnorm import RMSNorm
from layers.rotary_embeddings import RotaryEmbedding
from layers.fused_qkv_linear import FusedQKVLinear
from layers.swiglu import SwiGLU
from modules.layer_scale import LayerScale
from model_config import ModelConfig
from training import WarmupCosineScheduler, ZLossWrapper, initialize_weights, mark_residual_layers
from tokenizer_wrapper import Tokenizer

# Distributed utils
def setup_distributed():
    if 'RANK' in os.environ:
        rank, local_rank, world_size = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

def is_main(): return not dist.is_initialized() or dist.get_rank() == 0

# Dataset configs
DATASETS = {
    "pretrain": [("wikitext", "wikitext-103-raw-v1", "text")],
    "math": [("gsm8k", "main", "question"), ("hendrycks/competition_math", None, "problem")],
    "code": [("bigcode/starcoderdata", "python", "content")],
    "sft": [("OpenAssistant/oasst_top1_2023-08-25", None, "text")],
}

# Model
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qkv = FusedQKVLinear(cfg)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.rope = RotaryEmbedding(cfg)
    def forward(self, x):
        B,T,C = x.size()
        q,k,v = [t.transpose(1,2) for t in self.qkv(x)]
        cos,sin = self.rope(x,None)
        q,k = RotaryEmbedding.apply_rotary_pos_emb(q,k,cos,sin)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(y.transpose(1,2).contiguous().view(B,T,C))

class Block(nn.Module):
    def __init__(self, cfg, idx=0):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(cfg.n_embd), RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = SwiGLU(cfg)
        self.ls1 = LayerScale(cfg.n_embd, cfg.layer_scale_init) if cfg.layer_scale_init else nn.Identity()
        self.ls2 = LayerScale(cfg.n_embd, cfg.layer_scale_init) if cfg.layer_scale_init else nn.Identity()
    def forward(self, x):
        x = x + self.ls1(self.attn(self.ln1(x)))
        return x + self.ls2(self.mlp(self.ln2(x)))

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout_rate)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_layers)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.head.weight
        self.apply(lambda m: initialize_weights(m, n_layers=cfg.num_layers))
        mark_residual_layers(self)
    def forward(self, x, targets=None):
        h = self.drop(self.wte(x))
        for b in self.blocks: h = b(h)
        return self.head(self.ln_f(h)) if targets is not None else self.head(self.ln_f(h)[:,-1:])
    @torch.no_grad()
    def generate(self, idx, n=100, temp=0.8, top_k=50):
        for _ in range(n):
            logits = self(idx[:,-self.cfg.sequence_length:])[:,-1,:]/temp
            if top_k:
                v,_ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,-1:]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits,-1),1)], 1)
        return idx

def load_data(stage, tokenizer, seq_len, max_mb=500):
    """Load and tokenize data for given stage."""
    from datasets import load_dataset
    all_tokens = []
    for ds_path, ds_name, field in DATASETS.get(stage, []):
        if is_main(): print(f"Loading {ds_path}...")
        try:
            ds = load_dataset(ds_path, ds_name, split="train", streaming=True, trust_remote_code=True)
            size = 0
            for item in ds:
                txt = item.get(field, "") or ""
                if stage == "sft" and "messages" in item:
                    msgs = item["messages"]
                    txt = "".join([f"<|{m.get('role','user')}|>{m.get('content','')}" for m in msgs])
                if txt:
                    tokens = tokenizer.encode(txt)
                    all_tokens.extend(tokens)
                    size += len(txt.encode('utf-8'))
                    if size > max_mb * 1024 * 1024: break
        except Exception as e:
            if is_main(): print(f"Error loading {ds_path}: {e}")
    if dist.is_initialized(): dist.barrier()
    return np.array(all_tokens, dtype=np.uint16)

def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+seq_len].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def save_plots(history, path):
    if not HAS_MATPLOTLIB or not history: return
    path.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    steps = [h['step'] for h in history]
    ax1.plot(steps, [h['loss'] for h in history], label='Loss')
    ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(steps, [h['lr'] for h in history], color='green')
    ax2.set_xlabel('Step'); ax2.set_ylabel('LR'); ax2.grid(True, alpha=0.3)
    fig.savefig(path / 'training.png', dpi=150); plt.close(fig)

def train_stage(model, optimizer, data, cfg, args, stage_name, target_steps, device, world_size):
    """Train for one stage."""
    scheduler = WarmupCosineScheduler(optimizer, int(target_steps*0.05), target_steps, min_lr=1e-5)
    criterion = ZLossWrapper(z_coefficient=5e-4)
    history = []
    
    pbar = tqdm(range(target_steps), disable=not is_main(), desc=f"{stage_name}", miniters=1000)
    for step in pbar:
        x, y = get_batch(data, cfg.sequence_length, args.batch_size, device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x, y)
            loss, ce, zl = criterion(logits, y, return_components=True)
        (loss / args.grad_accum).backward()
        
        if (step+1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        
        if step % 1000 == 0:
            lr = scheduler.get_lr()
            history.append({'step': step, 'loss': loss.item(), 'lr': lr, 'stage': stage_name})
            if is_main(): pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})
    
    return history

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="pretrain", choices=["pretrain", "math", "code", "sft", "full"])
    parser.add_argument("--model", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--data-mb", type=int, default=500)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()
    
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # Dirs
    ckpt_dir = Path("./checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    plots_dir = ckpt_dir / "plots"; plots_dir.mkdir(exist_ok=True)
    
    # Config
    presets = {"small": ModelConfig.small, "medium": ModelConfig.medium, "large": ModelConfig.large}
    cfg = presets[args.model](vocab_size=args.vocab_size)
    
    if is_main():
        print("="*60)
        print(f"Stage: {args.stage} | Model: {args.model} | GPUs: {world_size}")
        print(f"Params: ~{cfg.num_parameters_estimate/1e6:.0f}M | Hours: {args.hours}")
        print("="*60)
    
    # Tokenizer
    tokenizer = Tokenizer()
    if not Path("tokenizer_vocab.txt").exists():
        if is_main():
            print("Training tokenizer on pretrain data...")
            data = load_data("pretrain", tokenizer, cfg.sequence_length, 100)
            # Need to train first
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            texts = [item['text'] for i, item in zip(range(10000), ds) if item['text'].strip()]
            with open("corpus.txt", "w") as f: f.write("\n".join(texts))
            tokenizer.train("corpus.txt", vocab_size=args.vocab_size)
            tokenizer.save()
    if dist.is_initialized(): dist.barrier()
    tokenizer = Tokenizer()
    cfg.vocab_size = tokenizer.vocab_size
    
    # Model
    model = GPT(cfg).to(device)
    if args.resume and Path(args.resume).exists():
        if is_main(): print(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['model'])
    
    # FSDP
    if world_size > 1:
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, mixed_precision=mp,
                     auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
                     device_id=local_rank)
        if is_main(): print("FSDP enabled")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, fused=True)
    
    # Estimate speed
    if is_main(): print("Estimating speed...")
    warmup_data = load_data("pretrain", tokenizer, cfg.sequence_length, 10)
    t0 = time.time()
    for _ in range(5):
        x, y = get_batch(warmup_data, cfg.sequence_length, args.batch_size, device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x,y).view(-1, cfg.vocab_size), y.view(-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    steps_per_sec = 5 / (time.time() - t0)
    if is_main(): print(f"Speed: {steps_per_sec:.2f} steps/sec")
    
    # Stages
    stages = [args.stage] if args.stage != "full" else ["pretrain", "math", "code", "sft"]
    hours_per_stage = args.hours / len(stages)
    all_history = []
    
    for stage in stages:
        if is_main(): print(f"\n{'='*60}\nSTAGE: {stage.upper()}\n{'='*60}")
        
        data = load_data(stage, tokenizer, cfg.sequence_length, args.data_mb)
        if len(data) < cfg.sequence_length + 1:
            if is_main(): print(f"Skipping {stage} - not enough data")
            continue
        
        target_steps = int(steps_per_sec * hours_per_stage * 3600)
        if is_main(): print(f"Target steps: {target_steps:,}")
        
        history = train_stage(model, optimizer, data, cfg, args, stage, target_steps, device, world_size)
        all_history.extend(history)
        
        # Save checkpoint
        if is_main():
            torch.save({'model': model.state_dict() if not isinstance(model, FSDP) else 
                       {k: v.cpu() for k,v in model.state_dict().items()},
                       'config': cfg.to_dict(), 'stage': stage}, ckpt_dir / f"{stage}.pt")
    
    # Final
    if is_main():
        torch.save({'model': model.state_dict() if not isinstance(model, FSDP) else 
                   {k: v.cpu() for k,v in model.state_dict().items()},
                   'config': cfg.to_dict(), 'history': all_history}, ckpt_dir / "final.pt")
        with open(ckpt_dir / "history.json", "w") as f: json.dump(all_history, f)
        save_plots(all_history, plots_dir)
        
        print("\n" + "="*60)
        print("GENERATING SAMPLE")
        print("="*60)
        model.eval()
        prompt = torch.zeros((1,1), dtype=torch.long, device=device)
        gen = model.generate(prompt, n=100)
        print(tokenizer.decode(gen[0].tolist()))
    
    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    train()
