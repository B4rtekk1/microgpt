"""
Download and prepare datasets for training.
- Downloads 10GB of FineWeb2 for pretraining
- Downloads 50MB of each SFT dataset (math-sft, sft-ultra, math-sft-plus)

Usage:
    python download_datasets.py
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Output directory
DATASET_DIR = Path(__file__).parent / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASETS = {
    "fineweb2": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",  # English educational content
        "split": "train",
        "size_bytes": 1 * 1024 * 1024 * 1024,  # 1GB
        "text_key": "text"
    },
    "math-sft": {
        "path": "openai/gsm8k",
        "name": "main", 
        "split": "train",
        "size_bytes": 50 * 1024 * 1024,  # 50MB
        "format": "qa",  # question-answer format
        "question_key": "question",
        "answer_key": "answer"
    },
    "sft-ultra": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "name": "default",
        "split": "train_sft",
        "size_bytes": 50 * 1024 * 1024,  # 50MB
        "format": "messages",
        "messages_key": "messages"
    },
    "math-sft-plus": {
        "path": "AI-MO/NuminaMath-CoT",
        "name": None,
        "split": "train",
        "size_bytes": 50 * 1024 * 1024,  # 50MB
        "format": "qa",
        "question_key": "problem",
        "answer_key": "solution"
    }
}


def format_item(item: dict, config: dict) -> str:
    """Format a dataset item into text."""
    format_type = config.get("format", "text")
    
    if format_type == "text":
        text_key = config.get("text_key", "text")
        return item.get(text_key, str(item))
    
    elif format_type == "qa":
        q_key = config.get("question_key", "question")
        a_key = config.get("answer_key", "answer")
        question = item.get(q_key, "")
        answer = item.get(a_key, "")
        return f"### Question:\n{question}\n\n### Answer:\n{answer}"
    
    elif format_type == "messages":
        messages_key = config.get("messages_key", "messages")
        messages = item.get(messages_key, [])
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"### User:\n{content}")
            elif role == "assistant":
                formatted.append(f"### Assistant:\n{content}")
            elif role == "system":
                formatted.append(f"### System:\n{content}")
        return "\n\n".join(formatted)
    
    return str(item)


def download_dataset(name: str, config: dict):
    """Download a single dataset."""
    output_file = DATASET_DIR / f"{name}.jsonl"
    
    # Check if already exists
    if output_file.exists():
        size = output_file.stat().st_size
        target_size = config["size_bytes"]
        if size >= target_size * 0.9:  # Allow 10% tolerance
            print(f"[{name}] Already downloaded ({size / (1024*1024):.1f}MB)")
            return
    
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Source: {config['path']}")
    target_mb = config["size_bytes"] / (1024 * 1024)
    target_gb = config["size_bytes"] / (1024 * 1024 * 1024)
    if target_gb >= 1:
        print(f"Target size: {target_gb:.1f}GB")
    else:
        print(f"Target size: {target_mb:.1f}MB")
    print(f"{'='*60}")
    
    # Load dataset with streaming
    load_args = {
        "path": config["path"],
        "split": config["split"],
        "streaming": True
    }
    
    if config.get("name"):
        load_args["name"] = config["name"]
    
    try:
        dataset = load_dataset(**load_args)
    except Exception as e:
        print(f"[{name}] Error loading dataset: {e}")
        return
    
    # Collect data
    current_size = 0
    target_size = config["size_bytes"]
    samples = []
    
    pbar_unit = "GB" if target_size >= 1024*1024*1024 else "MB"
    pbar_divisor = 1024*1024*1024 if pbar_unit == "GB" else 1024*1024
    pbar = tqdm(
        total=target_size / pbar_divisor,
        desc=f"Downloading {name}",
        unit=pbar_unit,
        bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f}{unit} [{elapsed}<{remaining}]'
    )
    last_pbar_val = 0
    
    for item in dataset:
        text = format_item(item, config)
        text_bytes = text.encode('utf-8')
        size = len(text_bytes)
        
        samples.append({"text": text})
        current_size += size
        
        # Update progress
        current_unit = current_size / pbar_divisor
        if current_unit - last_pbar_val >= 0.01:
            pbar.update(current_unit - last_pbar_val)
            last_pbar_val = current_unit
        
        if current_size >= target_size:
            break
    
    pbar.close()
    
    # Save as JSONL
    print(f"[{name}] Saving {len(samples):,} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    actual_size = output_file.stat().st_size
    actual_mb = actual_size / (1024 * 1024)
    actual_gb = actual_size / (1024 * 1024 * 1024)
    
    if actual_gb >= 1:
        print(f"[{name}] Done! Saved {actual_gb:.2f}GB ({len(samples):,} samples)")
    else:
        print(f"[{name}] Done! Saved {actual_mb:.2f}MB ({len(samples):,} samples)")


def main():
    print("="*60)
    print("DATASET DOWNLOAD SCRIPT")
    print("="*60)
    print(f"Output directory: {DATASET_DIR}")
    print(f"Datasets to download:")
    for name, config in DATASETS.items():
        size_mb = config["size_bytes"] / (1024 * 1024)
        size_gb = config["size_bytes"] / (1024 * 1024 * 1024)
        if size_gb >= 1:
            print(f"  - {name}: {size_gb:.1f}GB")
        else:
            print(f"  - {name}: {size_mb:.1f}MB")
    print("="*60)
    
    for name, config in DATASETS.items():
        try:
            download_dataset(name, config)
        except Exception as e:
            print(f"[{name}] Failed: {e}")
            continue
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    
    # Show summary
    total_size = 0
    for file in DATASET_DIR.glob("*.jsonl"):
        size = file.stat().st_size
        total_size += size
        size_mb = size / (1024 * 1024)
        size_gb = size / (1024 * 1024 * 1024)
        if size_gb >= 1:
            print(f"  {file.name}: {size_gb:.2f}GB")
        else:
            print(f"  {file.name}: {size_mb:.2f}MB")
    
    total_gb = total_size / (1024 * 1024 * 1024)
    print(f"\nTotal: {total_gb:.2f}GB")
    print(f"Location: {DATASET_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
