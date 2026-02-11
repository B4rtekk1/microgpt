"""
Training configuration for NVIDIA H200 (141GB VRAM)
Optimized for ~1.3B parameter model with 50 hours of training time.

Usage:
    python train_h200.py
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import platform
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import sys
import json
from tqdm import tqdm
from dataclasses import dataclass, field

# Import from main train.py
from train import (
    Trainer,
    TrainingConfig,
    PipelineStage,
    GPT,
    CausalSelfAttention,
    MLP,
    Block,
)

from model_config import ModelConfig
from config import HardwareInfo

# =============================================================================
# H200 OPTIMIZED CONFIGURATION
# =============================================================================

# Model Configuration - ~1.3B parameters
VOCAB_SIZE = 32000              # Standard LLM vocab size
SEQUENCE_LENGTH = 2048          # Good context length for training
NUM_LAYERS = 24                 # Depth for capability
N_HEADS = 16                    # 128 dim/head (optimal for Flash Attention)
N_KV_HEADS = 4                  # GQA - 4x memory savings on KV cache
N_EMBD = 2048                   # Width
DROPOUT_RATE = 0.0              # No dropout with large data
MAX_POSITION_EMBEDDINGS = 4096  # Support for longer context at inference
ROPE_BASE = 10000

# Advanced optimizations
USE_CHECKPOINTS = True          # Gradient checkpointing for memory
USE_DROP_PATH = False           # Off for pretraining
DROP_PATH_RATE = 0.0
LAYER_SCALE_INIT = 1e-5         # LayerScale for training stability

# Position encodings
USE_RELATIVE_POSITION_BIAS = False
REL_POS_NUM_BUCKETS = 32
REL_POS_MAX_DISTANCE = 128
REL_POS_BIDIRECTIONAL = False

# RoPE scaling (None for standard training)
ROPE_SCALING = None
ROPE_SCALING_FACTOR = 1.0

# =============================================================================
# TRAINING CONFIGURATION FOR 50 HOURS
# =============================================================================

TRAINING_MODE = "pipeline"      # Multi-stage training

# H200 can handle large batches!
BATCH_SIZE = 128                # Large batch for H200
GRAD_ACCUM_STEPS = 4            # Effective batch = 512
LEARNING_RATE = 3e-4            # Standard for 1B+ models
OPTIMIZER_TYPE = "sophia_g"     # Or "adamw"
GRAD_CLIP = 1.0

# No early stopping for full training
EARLY_STOPPING_PATIENCE = 0

# Resume from checkpoint if interrupted
RESUME_FROM = None

# Logging
LOG_DIR = "./logs_h200"
DEBUG_MODE = False

# =============================================================================
# TRAINING PIPELINE - 50 HOURS TOTAL
# =============================================================================

# Time allocation:
# - Pretraining: 30 hours (~60,000 steps)  [FineWeb-Edu: language & knowledge]
# - Basic Math: 2.5 hours (~5,000 steps)   [MetaMathQA: algebra, arithmetic, geometry]
# - Math Reasoning: 8 hours (~16,000 steps) [OpenMathInstruct-2: 5M GSM8K+MATH problems with CoT]
# - Diverse Math: 2.5 hours (~5,000 steps)  [MathInstruct: 262k from 13 sources]
# - SFT: 4 hours (~8,000 steps)
# - CoT: 3 hours (~6,000 steps)

TRAINING_PIPELINE = [
    # Stage 1: Pretraining on FineWeb-Edu (Language & Knowledge)
    # ~30 hours, ~10 epochs over 25GB data
    PipelineStage(
        name="Pretraining",
        dataset="local:fineweb",
        steps=60000,
        lr=3e-4,
        data_size_gb=25.0,
        eval_prompt="The theory of relativity states that"
    ),
    
    # Stage 2: Basic Math (MetaMathQA - algebra, arithmetic, geometry, word problems)
    # ~2.5 hours, ~5 epochs over ~400MB (~100M tokens)
    PipelineStage(
        name="Basic Mathematics",
        dataset="local:math-basic",
        steps=5000,
        lr=1e-4,
        data_size_gb=0.4,
        eval_prompt="[INST] What is 17 * 23? [/INST]"
    ),
    
    # Stage 3: Math Reasoning (OpenMathInstruct-2 - 5M problems by Llama-405B)
    # ~8 hours, ~1.3 epochs over ~4.5GB (~1.1B tokens)
    PipelineStage(
        name="Math Reasoning",
        dataset="local:math-openmath2",
        steps=16000,
        lr=8e-5,
        data_size_gb=4.5,
        eval_prompt="[INST] Solve for x: 3x^2 - 12x + 9 = 0 [/INST]"
    ),
    
    # Stage 4: Diverse Math (MathInstruct - 262k from 13 sources)
    # ~2.5 hours, ~8 epochs over ~212MB (~53M tokens)
    # Includes: TheoremQA, AQuA-RAT, NumGLUE, MathQA
    PipelineStage(
        name="Diverse Mathematics",
        dataset="local:math-instruct",
        steps=5000,
        lr=5e-5,
        data_size_gb=0.25,
        eval_prompt="[INST] If a fair die is rolled 3 times, what is the probability of getting exactly 2 sixes? [/INST]"
    ),
    
    # Stage 5: Instruction Fine-tuning (SFT)
    # ~4 hours, ~32 epochs over 1GB data
    PipelineStage(
        name="Instruction Tuning",
        dataset="local:sft-ultra",
        steps=8000,
        lr=3e-5,
        data_size_gb=1.0,
        eval_prompt="[INST] Explain quantum computing in simple terms. [/INST]"
    ),
    
    # Stage 6: Chain-of-Thought Fine-tuning (NuminaMath-CoT)
    # ~3 hours, ~18 epochs over ~500MB
    PipelineStage(
        name="Chain-of-Thought",
        dataset="local:math-sft-plus",
        steps=6000,
        lr=2e-5,
        data_size_gb=0.5,
        eval_prompt="[INST] Solve this problem step by step:\n\nFind the limit of (x^2 - 1)/(x - 1) as x approaches 1. [/INST]"
    ),
]

# =============================================================================
# ALTERNATIVE CONFIGURATIONS
# =============================================================================

# For smaller training run (10 hours):
QUICK_PIPELINE = [
    PipelineStage(
        name="Pretraining",
        dataset="local:fineweb2",
        steps=15000,                    # ~8 hours
        lr=3e-4,
        data_size_gb=5.0,
        eval_prompt="The"
    ),
    PipelineStage(
        name="SFT",
        dataset="local:sft-ultra",
        steps=3000,                     # ~2 hours
        lr=5e-5,
        data_size_gb=0.5,
        eval_prompt="[INST] Hello! [/INST]"
    ),
]

# For 700M model (more conservative):
MODEL_700M = {
    "num_layers": 20,
    "n_heads": 16,
    "n_kv_heads": 4,
    "n_embd": 1536,
}


def create_model_config(use_700m: bool = False) -> ModelConfig:
    """Create model configuration."""
    if use_700m:
        return ModelConfig(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            num_layers=MODEL_700M["num_layers"],
            n_heads=MODEL_700M["n_heads"],
            n_kv_heads=MODEL_700M["n_kv_heads"],
            n_embd=MODEL_700M["n_embd"],
            dropout_rate=DROPOUT_RATE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            base=ROPE_BASE,
            use_checkpoints=USE_CHECKPOINTS,
            use_drop_path=USE_DROP_PATH,
            drop_path_rate=DROP_PATH_RATE,
            layer_scale_init=LAYER_SCALE_INIT,
        )
    
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        num_layers=NUM_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        n_embd=N_EMBD,
        dropout_rate=DROPOUT_RATE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        base=ROPE_BASE,
        use_checkpoints=USE_CHECKPOINTS,
        use_drop_path=USE_DROP_PATH,
        drop_path_rate=DROP_PATH_RATE,
        layer_scale_init=LAYER_SCALE_INIT,
        use_relative_position_bias=USE_RELATIVE_POSITION_BIAS,
        rel_pos_num_buckets=REL_POS_NUM_BUCKETS,
        rel_pos_max_distance=REL_POS_MAX_DISTANCE,
        rel_pos_bidirectional=REL_POS_BIDIRECTIONAL,
        rope_scaling=ROPE_SCALING,
        rope_scaling_factor=ROPE_SCALING_FACTOR,
    )


def create_training_config(quick_mode: bool = False) -> TrainingConfig:
    """Create training configuration."""
    pipeline = QUICK_PIPELINE if quick_mode else TRAINING_PIPELINE
    
    return TrainingConfig(
        mode=TRAINING_MODE,
        max_steps=100000,               # Will be overridden by pipeline
        max_epochs=1,
        target_hours=50.0 if not quick_mode else 10.0,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        optimizer_type=OPTIMIZER_TYPE,
        max_grad_norm=GRAD_CLIP,
        dataset_stage="pipeline",
        pipeline=pipeline,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        resume_from=RESUME_FROM,
        log_dir=LOG_DIR,
        debug=DEBUG_MODE,
    )


def print_config_summary(model_config: ModelConfig, training_config: TrainingConfig):
    """Print configuration summary."""
    print("=" * 70)
    print("H200 TRAINING CONFIGURATION")
    print("=" * 70)
    
    # Model info
    params = model_config.num_parameters_estimate
    print(f"\n📦 MODEL:")
    print(f"   Parameters: {params / 1e9:.2f}B ({params / 1e6:.0f}M)")
    print(f"   Layers: {model_config.num_layers}")
    print(f"   Heads: {model_config.n_heads} (KV: {model_config.n_kv_heads})")
    print(f"   Embedding dim: {model_config.n_embd}")
    print(f"   Sequence length: {model_config.sequence_length}")
    print(f"   Vocab size: {model_config.vocab_size}")
    
    # Training info
    effective_batch = training_config.batch_size * training_config.gradient_accumulation_steps
    print(f"\n⚙️  TRAINING:")
    print(f"   Batch size: {training_config.batch_size} (effective: {effective_batch})")
    print(f"   Learning rate: {training_config.learning_rate:.2e}")
    print(f"   Optimizer: {training_config.optimizer_type}")
    print(f"   Gradient clipping: {training_config.max_grad_norm}")
    
    # Pipeline info
    print(f"\n📊 PIPELINE:")
    total_steps = 0
    for i, stage in enumerate(training_config.pipeline, 1):
        total_steps += stage.steps
        print(f"   {i}. {stage.name}: {stage.steps:,} steps, LR={stage.lr:.2e}, Data={stage.data_size_gb}GB")
    print(f"   TOTAL: {total_steps:,} steps")
    
    # Estimates
    tokens_per_step = effective_batch * model_config.sequence_length
    total_tokens = total_steps * tokens_per_step
    print(f"\n📈 ESTIMATES:")
    print(f"   Tokens per step: {tokens_per_step:,}")
    print(f"   Total tokens: {total_tokens / 1e9:.1f}B")
    
    vram_estimate = model_config.estimate_vram_gb(
        batch_size=training_config.batch_size,
        include_optimizer_states=True
    )
    print(f"   Est. VRAM usage: {vram_estimate:.1f}GB (H200 has 141GB)")
    
    print("=" * 70)


def main():
    """Main training function for H200."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train picoGPT on H200")
    parser.add_argument("--quick", action="store_true", help="Use quick 10-hour pipeline")
    parser.add_argument("--small", action="store_true", help="Use 700M model instead of 1.3B")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()
    
    # Create configurations
    model_config = create_model_config(use_700m=args.small)
    training_config = create_training_config(quick_mode=args.quick)
    
    # Print summary
    print_config_summary(model_config, training_config)
    
    if args.dry_run:
        print("\n[DRY RUN] Exiting without training.")
        return
    
    # Check for datasets
    dataset_dir = Path(__file__).parent / "dataset"
    required_datasets = ["fineweb.jsonl", "math-basic.jsonl", "math-openmath2.jsonl", "math-instruct.jsonl", "sft-ultra.jsonl", "math-sft-plus.jsonl"]
    missing = [d for d in required_datasets if not (dataset_dir / d).exists()]
    
    if missing:
        print(f"\n⚠️  Missing datasets: {missing}")
        print("   Run 'python download_datasets.py' first!")
        print("   For H200 training, edit download_datasets.py to increase fineweb2 size to 25GB")
        return
    
    # Start training
    print("\n🚀 Starting H200 training...")
    trainer = Trainer(model_config, training_config)
    trainer.train()
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
