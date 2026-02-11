"""
Download NuminaMath-CoT, train tokenizer on it, and tokenize to binary.

Usage:
    python prepare_cot.py
    python prepare_cot.py --vocab-size 16000 --max-samples 50000
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

from tokenizer_wrapper import Tokenizer

# Must match tokenizer/src/lib.rs SPECIAL_TOKENS
INST_START = "[INST]"
INST_END = "[/INST]"
EOS_TOKEN = "<EOS>"

ROOT = Path(__file__).parent.absolute()
DATASET_DIR = ROOT / "dataset"
CACHE_DIR = ROOT / "data_cache" / "cot"
CORPUS_FILE = ROOT / "cot_corpus.txt"
VOCAB_FILE = ROOT / "tokenizer_vocab.txt"


def format_cot(item: dict) -> str:
    """Format a NuminaMath-CoT item as [INST] problem [/INST] solution <EOS>."""
    problem = item.get("problem", "").strip()
    solution = item.get("solution", "").strip()
    if not problem or not solution:
        return ""
    instruction = f"Solve this problem step by step:\n\n{problem}"
    return f"{INST_START} {instruction} {INST_END} {solution} {EOS_TOKEN}"


def download_cot(max_samples: int | None = None) -> list[str]:
    """Download NuminaMath-CoT and return formatted texts."""
    jsonl_path = DATASET_DIR / "math-sft-plus.jsonl"

    # If already downloaded as JSONL, load from disk
    if jsonl_path.exists():
        print(f"Loading from local cache: {jsonl_path}")
        texts = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                item = json.loads(line)
                text = item.get("text", "")
                if not text:
                    text = format_cot(item)
                if text:
                    texts.append(text)
        print(f"Loaded {len(texts):,} samples from local JSONL")
        return texts

    # Otherwise stream from HuggingFace
    print("Downloading NuminaMath-CoT from HuggingFace...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)

    texts = []
    skipped = 0
    for i, item in enumerate(tqdm(ds, desc="Downloading CoT")):
        if max_samples and i >= max_samples:
            break
        text = format_cot(item)
        if text:
            texts.append(text)
        else:
            skipped += 1

    print(f"Downloaded {len(texts):,} samples (skipped {skipped} empty)")

    # Save JSONL for reuse
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Saved JSONL to {jsonl_path}")

    return texts


def train_tokenizer(texts: list[str], vocab_size: int, min_freq: int) -> Tokenizer:
    """Write corpus, train tokenizer, save vocab."""
    print(f"\nWriting corpus ({len(texts):,} samples)...")
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    corpus_mb = CORPUS_FILE.stat().st_size / (1024 * 1024)
    print(f"Corpus: {corpus_mb:.1f} MB")

    tokenizer = Tokenizer()
    print(f"Training tokenizer (vocab_size={vocab_size}, min_freq={min_freq})...")
    tokenizer.train(str(CORPUS_FILE), vocab_size=vocab_size, min_freq=min_freq)
    tokenizer.save(str(VOCAB_FILE))
    print(f"Tokenizer saved to {VOCAB_FILE} (vocab: {tokenizer.vocab_size})")

    return tokenizer


def tokenize_dataset(texts: list[str], tokenizer: Tokenizer, val_split: float = 0.05):
    """Tokenize all texts and save train/val binary splits."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTokenizing {len(texts):,} samples...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

    data = np.array(all_tokens, dtype=np.uint16)
    total = len(data)
    print(f"Total tokens: {total:,}")

    # Split
    n = int((1 - val_split) * total)
    train_data = data[:n]
    val_data = data[n:]

    train_bin = CACHE_DIR / "train.bin"
    val_bin = CACHE_DIR / "val.bin"
    train_data.tofile(train_bin)
    val_data.tofile(val_bin)

    train_mb = train_bin.stat().st_size / (1024 * 1024)
    val_mb = val_bin.stat().st_size / (1024 * 1024)
    print(f"Train: {len(train_data):,} tokens ({train_mb:.1f} MB) -> {train_bin}")
    print(f"Val:   {len(val_data):,} tokens ({val_mb:.1f} MB) -> {val_bin}")

    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description="Download CoT, train tokenizer, tokenize")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Tokenizer vocab size")
    parser.add_argument("--min-freq", type=int, default=2, help="Min frequency for tokenizer training")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (None = all)")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing JSONL")
    parser.add_argument("--skip-train-tokenizer", action="store_true", help="Skip tokenizer training, use existing vocab")
    args = parser.parse_args()

    print("=" * 60)
    print("PREPARE COT PIPELINE")
    print("=" * 60)
    print(f"  Vocab size:   {args.vocab_size}")
    print(f"  Min freq:     {args.min_freq}")
    print(f"  Max samples:  {args.max_samples or 'all'}")
    print(f"  Val split:    {args.val_split}")
    print("=" * 60)

    # 1) Download
    print("\n[1/3] DOWNLOAD")
    texts = download_cot(max_samples=args.max_samples)

    # 2) Train tokenizer
    print("\n[2/3] TRAIN TOKENIZER")
    if args.skip_train_tokenizer and VOCAB_FILE.exists():
        print(f"Skipping — loading existing vocab from {VOCAB_FILE}")
        tokenizer = Tokenizer(str(VOCAB_FILE))
    else:
        tokenizer = train_tokenizer(texts, args.vocab_size, args.min_freq)

    # 3) Tokenize
    print("\n[3/3] TOKENIZE")
    train_data, val_data = tokenize_dataset(texts, tokenizer, args.val_split)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"  Vocab:   {tokenizer.vocab_size} tokens  ({VOCAB_FILE})")
    print(f"  Train:   {len(train_data):,} tokens")
    print(f"  Val:     {len(val_data):,} tokens")
    print(f"  Cache:   {CACHE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
