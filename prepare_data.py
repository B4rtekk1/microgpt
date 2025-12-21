"""
Prepare and cache datasets for training.
Run this locally before uploading to Lambda.ai to save time.

Usage:
    python prepare_data.py --stages base,math-sft,sft-ultra,math-sft-plus --sizes 30,1,1,1
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from dataset_loader import DatasetLoader
from tokenizer_wrapper import Tokenizer
from sft_normalizer import SFTNormalizer

current_dir = Path(__file__).parent.absolute()

def prepare_stage(
    stage: str, 
    data_size_gb: float, 
    tokenizer: Tokenizer,
    train_tokenizer: bool = False,
    vocab_size: int = 32000
) -> tuple:
    """Prepare and cache a single stage dataset."""
    
    data_dir = current_dir / "data_cache" / stage
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    
    # Check cache
    if train_bin.exists() and val_bin.exists():
        print(f"[{stage}] Cache already exists at {data_dir}")
        train_data = np.fromfile(train_bin, dtype=np.uint16)
        val_data = np.fromfile(val_bin, dtype=np.uint16)
        print(f"[{stage}] Loaded {len(train_data) + len(val_data):,} tokens from cache")
        return train_data, val_data
    
    print(f"\n{'='*60}")
    print(f"PREPARING: {stage.upper()} ({data_size_gb} GB)")
    print(f"{'='*60}")
    
    loader = DatasetLoader()
    sft_normalizer = SFTNormalizer(add_eos=False)
    
    ds = loader.load(stage, streaming=True)
    
    target_size_bytes = int(data_size_gb * 1024 * 1024 * 1024)
    current_size = 0
    collected_texts = []
    
    iterator = ds
    if isinstance(ds, dict):
        iterator = ds.get('train', next(iter(ds.values())))
    
    pbar = tqdm(desc=f"Collecting {data_size_gb}GB", unit="MB")
    last_pbar_val = 0
    
    for item in iterator:
        if "sft" in stage.lower() or "math" in stage.lower():
            text = sft_normalizer.normalize(item, dataset_name=stage)
        else:
            text = item.get('text', str(item))
        
        text_bytes = text.encode('utf-8')
        size = len(text_bytes)
        
        collected_texts.append(text)
        current_size += size
        
        current_mb = current_size / (1024 * 1024)
        if int(current_mb) > last_pbar_val:
            pbar.update(int(current_mb) - last_pbar_val)
            last_pbar_val = int(current_mb)
        
        if current_size >= target_size_bytes:
            break
    pbar.close()
    
    actual_size_gb = current_size / (1024**3)
    print(f"[{stage}] Collected {len(collected_texts):,} samples, {actual_size_gb:.2f} GB")
    
    # Train tokenizer if needed (only for first stage)
    if train_tokenizer:
        tokenizer_corpus_path = current_dir / "tokenizer_corpus.txt"
        print(f"[{stage}] Writing tokenizer corpus...")
        with open(tokenizer_corpus_path, "w", encoding="utf-8") as f:
            for text in collected_texts:
                if len(text.strip()) > 0:
                    f.write(text + "\n")
        
        print(f"[{stage}] Training tokenizer (vocab size: {vocab_size})...")
        tokenizer.train(str(tokenizer_corpus_path), vocab_size=vocab_size)
        tokenizer.save()
        print(f"[{stage}] Tokenizer saved (actual vocab size: {tokenizer.vocab_size})")
    
    # Tokenize
    print(f"[{stage}] Tokenizing...")
    all_tokens = []
    for text in tqdm(collected_texts, desc="Tokenizing"):
        if len(text) > 0:
            encoded = tokenizer.encode(text)
            all_tokens.extend(encoded)
    
    data = np.array(all_tokens, dtype=np.uint16)
    print(f"[{stage}] Total tokens: {len(data):,}")
    
    # Split
    val_split = 0.1
    n = int((1 - val_split) * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Cache
    print(f"[{stage}] Caching to {data_dir}...")
    train_data.tofile(train_bin)
    val_data.tofile(val_bin)
    
    print(f"[{stage}] Done! Train: {len(train_data):,}, Val: {len(val_data):,} tokens")
    
    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    parser.add_argument("--stages", type=str, default="base,math-sft,sft-ultra,math-sft-plus",
                       help="Comma-separated list of stages")
    parser.add_argument("--sizes", type=str, default="30,1,1,1",
                       help="Comma-separated list of sizes in GB (matching stages)")
    parser.add_argument("--vocab-size", type=int, default=32000,
                       help="Vocabulary size for tokenizer")
    args = parser.parse_args()
    
    stages = args.stages.split(",")
    sizes = [float(s) for s in args.sizes.split(",")]
    
    if len(stages) != len(sizes):
        raise ValueError(f"Number of stages ({len(stages)}) must match number of sizes ({len(sizes)})")
    
    print("="*60)
    print("DATA PREPARATION SCRIPT")
    print("="*60)
    print(f"Stages: {stages}")
    print(f"Sizes (GB): {sizes}")
    print(f"Vocab size: {args.vocab_size}")
    print("="*60)
    
    # Initialize tokenizer
    tokenizer = Tokenizer()
    
    total_tokens = 0
    for i, (stage, size) in enumerate(zip(stages, sizes)):
        train_data, val_data = prepare_stage(
            stage=stage,
            data_size_gb=size,
            tokenizer=tokenizer,
            train_tokenizer=(i == 0),  # Only train tokenizer on first stage
            vocab_size=args.vocab_size
        )
        total_tokens += len(train_data) + len(val_data)
    
    print("\n" + "="*60)
    print("PREPARATION COMPLETE!")
    print("="*60)
    print(f"Total tokens prepared: {total_tokens:,}")
    print(f"\nFiles to upload to Lambda.ai:")
    print(f"  1. data_cache/ folder")
    print(f"  2. tokenizer.model (or your tokenizer files)")
    print("="*60)


if __name__ == "__main__":
    main()
