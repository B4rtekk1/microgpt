"""
Fully automated training script for picoGPT.

This script integrates all custom modules from the project and provides
an automated training pipeline based on ModelConfig architecture settings.

Features:
- Automatic hardware detection and optimization
- Integrated logging with TrainingLogger, MetricsLogger, ArchitectureLogger
- Custom Sophia optimizer with ZLoss
- Gradient accumulation and learning rate scheduling
- Automatic batch size and step estimation
- Checkpoint saving and model evaluation
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

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import custom layers and modules
from layers.rmsnorm import RMSNorm
from layers.rotary_embeddings import RotaryEmbedding
from layers.fused_qkv_linear import FusedQKVLinear
from layers.swiglu import SwiGLU
from layers.relative_position_bias import RelativePositionBias

# Import modules
from modules.layer_scale import LayerScale
from modules.drop_path import DropPath
from modules.kv_cache import KVCache

# Import model config
from model_config import ModelConfig

# Import training utilities (ALL from training/__init__.py)
from training import (
    create_sophia_optimizer,
    create_lr_scheduler,
    WarmupCosineScheduler,
    GradientAccumulator,
    ZLoss,
    ZLossWrapper,
    compute_z_loss,
    initialize_weights,
    mark_residual_layers,
    count_params,
    model_size_mb,
    SophiaG,
    SophiaH,
)

# Import loggers (ALL from logger/__init__.py)
from logger.logger import (
    Logger,
    TrainingLogger,
    MetricsLogger,
    ArchitectureLogger,
    MetricBuffer,
)

# Import dataset and tokenizer
from dataset_loader import DatasetLoader
from tokenizer_wrapper import Tokenizer

# Import hardware detection
from config import HardwareInfo, GPUArchitecture

current_dir = Path(__file__).parent.absolute()

# =============================================================================
# Training Configuration Dataclass
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for training parameters.
    
    This is separate from ModelConfig - this controls HOW we train,
    while ModelConfig controls WHAT we train.
    """
    # Training duration
    mode: str = "fixed"  # "5h" for time-based, "fixed" for step-based, "epochs" for epoch-based
    max_steps: int = 1000
    max_epochs: int = 10
    target_hours: float = 5.0
    
    # Batch settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Optimizer settings
    optimizer_type: str = "sophia_g"  # "sophia_g", "sophia_h", "adamw"
    learning_rate: float = 8e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Loss settings
    use_z_loss: bool = True
    z_loss_coefficient: float = 5e-4
    label_smoothing: float = 0.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500
    log_dir: str = "./logs"
    
    # Data settings
    data_size_mb: int = 1000
    val_split: float = 0.1
    dataset_stage: str = "base"  # "base", "sft", or "pre+sft" for two-stage training
    
    # Two-stage training settings (for pre+sft mode)
    pre_steps: int = 500  # Steps for pretraining stage
    sft_steps: int = 500  # Steps for SFT stage
    sft_lr: float = 2e-5  # Lower learning rate for SFT (fine-tuning)
    
    # Early stopping
    early_stopping_patience: int = 0  # 0 = disabled
    early_stopping_min_delta: float = 0.001
    
    # Resume training
    resume_from: Optional[str] = None
    
    # Debug/test mode
    debug: bool = False
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    def auto_configure(self, model_config: ModelConfig) -> None:
        """
        Auto-configure training parameters based on model size.
        Larger models get lower learning rates, more gradient accumulation, etc.
        """
        # Estimate parameters
        estimated_params = model_config.num_parameters_estimate
        
        # Scale learning rate inversely with model size
        if estimated_params > 1e9:  # > 1B params
            self.learning_rate = 3e-4
            self.gradient_accumulation_steps = max(4, self.gradient_accumulation_steps)
        elif estimated_params > 500e6:  # > 500M params
            self.learning_rate = 5e-4
            self.gradient_accumulation_steps = max(2, self.gradient_accumulation_steps)
        elif estimated_params > 100e6:  # > 100M params
            self.learning_rate = 6e-4
        else:  # Small models
            self.learning_rate = 8e-4
            
        # Adjust batch size based on sequence length
        if model_config.sequence_length > 2048:
            self.batch_size = max(8, self.batch_size // 2)
            
        # More warmup for larger models
        if estimated_params > 500e6:
            self.warmup_ratio = 0.1

# =============================================================================
# Model Components (same as before but using all project features)
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention with optional Relative Position Bias.
    
    Uses:
    - FusedQKVLinear for efficient projection
    - RotaryEmbedding for positional encoding
    - Flash Attention 2 / SDPA based on config
    - Optional RelativePositionBias (T5-style)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_heads
        self.n_kv_head = config.n_kv_heads
        self.head_dim = config.head_dim
        
        # Use FusedQKVLinear
        self.qkv_proj = FusedQKVLinear(config)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Positional encoding
        self.rotary = RotaryEmbedding(config)
        
        # Optional relative position bias
        if config.use_relative_position_bias:
            self.rel_pos_bias = RelativePositionBias(
                num_heads=config.n_heads,
                num_buckets=config.rel_pos_num_buckets or 32,
                max_distance=config.rel_pos_max_distance or 128,
                bidirectional=config.rel_pos_bidirectional or False
            )
        else:
            self.rel_pos_bias = None
            
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None, 
        kv_cache: Optional[KVCache] = None
    ) -> torch.Tensor:
        B, T, C = x.size()

        # QKV Projection
        q, k, v = self.qkv_proj.forward(x)

        # Transpose for attention: (B, nh, T, hs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary(x, position_ids)
        q, k = RotaryEmbedding.apply_rotary_pos_emb(q, k, cos, sin)

        # KV Cache Logic
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        
        # Get relative position bias if enabled
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(T, T, x.device)
        
        # Attention Dispatch
        if self.config.attention_implementation == "flash_attention_2":
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=(attn_bias is None)
                    )
            except RuntimeError:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=(attn_bias is None)
                )
        elif self.config.attention_implementation == "sdpa":
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=(attn_bias is None)
            )
        else:
            # Manual implementation (Eager)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # Add relative position bias if available
            if attn_bias is not None:
                att = att + attn_bias
            
            # Causal mask
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            y = att @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y


class MLP(nn.Module):
    """MLP block using SwiGLU activation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.swiglu = SwiGLU(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


class Block(nn.Module):
    """
    Transformer block with:
    - Pre-LayerNorm (RMSNorm)
    - LayerScale
    - DropPath (stochastic depth)
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Attention block
        self.ln1 = RMSNorm(config.n_embd, eps=config.variance_epsilon)
        self.attn = CausalSelfAttention(config)
        
        # LayerScale (optional based on config)
        if config.layer_scale_init is not None:
            self.ls1 = LayerScale(config.n_embd, config.layer_scale_init)
            self.ls2 = LayerScale(config.n_embd, config.layer_scale_init)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()
        
        # DropPath (stochastic depth, increases with layer depth)
        if config.use_drop_path and config.drop_path_rate:
            # Linear increase in drop rate with depth
            drop_rate = config.drop_path_rate * layer_idx / max(config.num_layers - 1, 1)
            self.drop_path1 = DropPath(drop_rate)
            self.drop_path2 = DropPath(drop_rate)
        else:
            self.drop_path1 = nn.Identity()
            self.drop_path2 = nn.Identity()
        
        # MLP block
        self.ln2 = RMSNorm(config.n_embd, eps=config.variance_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        # Attention with residual
        attn_out = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + self.drop_path1(self.ls1(attn_out))
        
        # MLP with residual
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.drop_path2(self.ls2(mlp_out))
        return x


class GPT(nn.Module):
    """
    GPT model using all project components.
    
    Features:
    - RMSNorm for layer normalization
    - RoPE for positional encoding
    - Fused QKV projection
    - SwiGLU activation
    - LayerScale and DropPath
    - Weight tying
    - Gradient checkpointing support
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout_rate),
            h=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.num_layers)]),
            ln_f=RMSNorm(config.n_embd, eps=config.variance_epsilon),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight  # type: ignore

        # Initialize weights using custom initialization
        self.apply(lambda m: initialize_weights(m, n_layers=config.num_layers))
        mark_residual_layers(self)
        
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        return count_params(self, trainable_only=True)
    
    def get_model_size_mb(self, dtype_bytes: int = 2) -> float:
        """Get model size in MB."""
        return model_size_mb(self, dtype_bytes=dtype_bytes)

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None, 
        kv_caches: Optional[List[KVCache]] = None,
        use_checkpoint: bool = False
    ) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        
        # Token embeddings
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        
        if kv_caches is None:
            kv_caches = [None] * len(self.transformer.h)  # type: ignore
        
        # Forward through blocks
        for block, cache in zip(self.transformer.h, kv_caches):  # type: ignore
            if use_checkpoint and self.config.use_checkpoints and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cache, use_reentrant=False)
            else:
                x = block(x, kv_cache=cache)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
        else:
            # Inference optimization: only compute last token
            logits = self.lm_head(x[:, [-1], :])

        return logits
    
    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text with optional top-k and top-p (nucleus) sampling."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =============================================================================
# Trainer Class - Main Training Logic
# =============================================================================

class Trainer:
    """
    Unified trainer that uses all project components.
    """
    
    def __init__(
        self, 
        model_config: ModelConfig,
        training_config: TrainingConfig,
        checkpoint_dir: Optional[Path] = None
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.checkpoint_dir = checkpoint_dir or (current_dir / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-configure training based on model
        training_config.auto_configure(model_config)
        
        # Initialize loggers
        self.logger = Logger(name="Trainer", log_dir=training_config.log_dir)
        self.training_logger = TrainingLogger(
            name="TrainingLogger", 
            log_dir=training_config.log_dir,
            log_to_file=True
        )
        self.metrics_logger = MetricsLogger(
            name="Metrics",
            use_wandb=False,
            use_tensorboard=True,
            log_dir=training_config.log_dir
        )
        self.arch_logger = ArchitectureLogger(
            name="Architecture",
            log_dir=training_config.log_dir
        )
        
        # Hardware info
        self.hardware_info = HardwareInfo.detect()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data and tokenizer (to be initialized)
        self.train_data = None
        self.val_data = None
        self.tokenizer = None
        
        # Model and optimizer (to be initialized)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.accumulator = None
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # History for plots
        self.loss_history = []
        self.val_loss_history = []
        self.lr_history = []
        
        # Performance tracking
        self.tokens_processed = 0
        self.start_time = None
        
        # GradScaler for mixed precision
        self.scaler = None
        
    def _log_system_info(self):
        """Log system and hardware information."""
        self.logger.info("=" * 60)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        
        if self.hardware_info.cuda_available:
            self.logger.info(f"CUDA version: {self.hardware_info.cuda_version}")
            self.logger.info(f"GPU count: {self.hardware_info.gpu_count}")
            for i, name in enumerate(self.hardware_info.gpu_names or []):
                self.logger.info(f"  GPU {i}: {name}")
            self.logger.info(f"Total GPU memory: {self.hardware_info.total_gpu_memory / 1024:.2f} GB")
            self.logger.info(f"Supports BF16: {self.hardware_info.supports_bf16}")
            self.logger.info(f"Supports Flash Attention: {self.hardware_info.supports_flash_attention}")
        
        self.logger.info("=" * 60)
        
    def _log_config(self):
        """Log model and training configurations."""
        self.training_logger.log_config(self.model_config, title="Model Configuration")
        self.training_logger.log_config(self.training_config, title="Training Configuration")
        
        # Log optimization summary
        self.logger.info("\n" + self.model_config.get_optimization_summary())
        
    def _format_sft_sample(self, item: Dict[str, Any]) -> str:
        """
        Format SFT dataset item into instruction format.
        Uses [INST] and [/INST] tokens for instruction wrapping.
        
        OpenAssistant format: has 'text' field with conversation.
        """
        # OpenAssistant oasst_top1 dataset has 'text' field with formatted conversation
        if 'text' in item:
            return item['text']
        
        # Fallback for other SFT formats (instruction/output)
        instruction = item.get('instruction', item.get('prompt', ''))
        output = item.get('output', item.get('response', item.get('completion', '')))
        
        if instruction and output:
            return f"[INST] {instruction} [/INST] {output}"
        elif instruction:
            return f"[INST] {instruction} [/INST]"
        else:
            return str(item)
    
    def prepare_data(self, train_tokenizer: bool = True):
        """Load and prepare data and tokenizer.
        
        Args:
            train_tokenizer: If True, train a new tokenizer on the data.
                           If False, use the existing tokenizer (for SFT after pretraining).
        """
        with self.logger.timer("Data Preparation"):
            loader = DatasetLoader()
            stage = self.training_config.dataset_stage
            data_dir = current_dir / "data_cache" / stage
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Paths for cached data
            train_bin = data_dir / "train.bin"
            val_bin = data_dir / "val.bin"
            
            # Check if cached data exists
            if train_bin.exists() and val_bin.exists():
                self.logger.info(f"Loading cached {stage} dataset from {data_dir}...")
                train_data = np.fromfile(train_bin, dtype=np.uint16)
                val_data = np.fromfile(val_bin, dtype=np.uint16)
                
                # Validation: check if we have enough data for at least one batch
                min_required = self.model_config.sequence_length + 1
                if len(train_data) < min_required or len(val_data) < 2:
                    self.logger.warning(f"Cached data is too small (train: {len(train_data)}, val: {len(val_data)}). Minimum required for sequence_length={self.model_config.sequence_length}: {min_required}. Discarding cache.")
                else:
                    self.train_data = train_data
                    self.val_data = val_data
                    self.logger.info(f"Loaded {len(self.train_data) + len(self.val_data):,} tokens from cache.")
                    
                    # Still need to load the tokenizer
                    try:
                        self.tokenizer = Tokenizer()
                        self.logger.info("Loaded existing tokenizer.")
                        # Update config vocab size
                        if self.tokenizer.vocab_size != self.model_config.vocab_size:
                            self.logger.info(f"Updating vocab_size from tokenizer: {self.model_config.vocab_size} -> {self.tokenizer.vocab_size}")
                            self.model_config.vocab_size = self.tokenizer.vocab_size
                    except Exception as e:
                        self.logger.warning(f"Could not load existing tokenizer: {e}. If this is a fresh run, delete data_cache.")
                    
                    return

            # If not cached, download and tokenize
            if stage == "sft":
                self.logger.info("Loading SFT dataset (OpenAssistant)...")
            else:
                self.logger.info("Loading WikiText dataset for pretraining...")
            
            ds = loader.load(stage, streaming=True)
            
            # Collect target amount of data
            target_size_bytes = self.training_config.data_size_mb * 1024 * 1024
            current_size = 0
            collected_texts = []
            
            iterator = ds
            if isinstance(ds, dict):
                iterator = ds.get('train', next(iter(ds.values())))
            
            pbar = tqdm(desc=f"Collecting {self.training_config.data_size_mb}MB", unit="MB")
            last_pbar_val = 0
            
            for item in iterator:
                # Format text based on dataset stage
                if stage == "sft":
                    text = self._format_sft_sample(item)
                else:
                    text = item.get('text', str(item))
                
                text_bytes = text.encode('utf-8')
                size = len(text_bytes)
                
                collected_texts.append(text)
                current_size += size
                
                # Update progress bar
                current_mb = current_size / (1024 * 1024)
                if int(current_mb) > last_pbar_val:
                    pbar.update(int(current_mb) - last_pbar_val)
                    last_pbar_val = int(current_mb)
                
                if current_size >= target_size_bytes:
                    break
            pbar.close()
                    
            actual_size_mb = current_size / (1024*1024)
            self.logger.info(f"Collected {len(collected_texts)} samples, {actual_size_mb:.2f} MB")
            
            if actual_size_mb < self.training_config.data_size_mb - 1:
                self.logger.warning(f"Requested {self.training_config.data_size_mb}MB but only found {actual_size_mb:.2f}MB in the dataset.")
            
            # Train tokenizer only if requested (typically only during pretraining)
            if train_tokenizer:
                # Prepare tokenizer corpus
                tokenizer_corpus_path = current_dir / "tokenizer_corpus.txt"
                self.logger.info(f"Writing tokenizer corpus...")
                with open(tokenizer_corpus_path, "w", encoding="utf-8") as f:
                    for text in collected_texts:
                        if len(text.strip()) > 0:
                            f.write(text + "\n")
                
                # Train new tokenizer
                try:
                    self.tokenizer = Tokenizer()
                except ImportError as e:
                    self.logger.error(f"Could not import custom Tokenizer: {e}")
                    raise
                
                self.logger.info(f"Training tokenizer (vocab size: {self.model_config.vocab_size})...")
                self.tokenizer.train(str(tokenizer_corpus_path), vocab_size=self.model_config.vocab_size)
                self.tokenizer.save()
                
                # Update config vocab size if needed
                real_vocab_size = self.tokenizer.vocab_size
                if real_vocab_size != self.model_config.vocab_size:
                    self.logger.info(f"Updating vocab_size: {self.model_config.vocab_size} -> {real_vocab_size}")
                    self.model_config.vocab_size = real_vocab_size
            else:
                # Reuse existing tokenizer (for SFT after pretraining)
                if self.tokenizer is None:
                    # Try loading it if it exists
                    try:
                        self.tokenizer = Tokenizer()
                        self.logger.info(f"Loaded existing tokenizer (vocab size: {self.tokenizer.vocab_size})")
                    except Exception:
                        raise ValueError("Cannot reuse tokenizer - no tokenizer has been trained yet and none found on disk!")
                else:
                    self.logger.info(f"Reusing existing tokenizer (vocab size: {self.tokenizer.vocab_size})")
            
            # Tokenize dataset with current tokenizer
            self.logger.info(f"Tokenizing {stage.upper()} dataset...")
            all_tokens = []
            for text in tqdm(collected_texts, desc="Tokenizing"):
                if len(text) > 0:
                    encoded = self.tokenizer.encode(text)
                    all_tokens.extend(encoded)
            
            data = np.array(all_tokens, dtype=np.uint16)
            self.logger.info(f"Total tokens: {len(data):,}")
            
            # Split train/val
            n = int((1 - self.training_config.val_split) * len(data))
            self.train_data = data[:n]
            self.val_data = data[n:]
            
            # Save to cache
            self.logger.info(f"Caching tokenized data to {data_dir}...")
            self.train_data.tofile(train_bin)
            self.val_data.tofile(val_bin)
            self.logger.info("Successfully cached dataset.")
                
    def build_model(self):
        """Build and configure the model."""
        with self.logger.timer("Model Building"):
            self.model = GPT(self.model_config)
            self.model.to(self.device)
            
            # Log architecture
            self.arch_logger.log_model_summary(self.model, depth=3)
            
            # Log parameter count
            num_params = self.model.get_num_params()
            size_mb = self.model.get_model_size_mb()
            self.logger.info(f"Model parameters: {num_params / 1e6:.2f}M")
            self.logger.info(f"Model size: {size_mb:.2f} MB")
            
            # Estimate VRAM
            estimated_vram = self.model_config.estimate_vram_gb(
                batch_size=self.training_config.batch_size,
                include_optimizer_states=True
            )
            self.logger.info(f"Estimated VRAM usage: {estimated_vram:.2f} GB")
            
            # Apply optimizations
            self._apply_optimizations()
            
    def _apply_optimizations(self):
        """Apply hardware-based optimizations."""
        # Mixed precision context
        ptdtype = torch.float32
        if self.model_config.mixed_precision == "bf16":
            ptdtype = torch.bfloat16
        elif self.model_config.mixed_precision == "fp16":
            ptdtype = torch.float16
        
        self.autocast_dtype = ptdtype
        self.autocast_ctx = (
            torch.nullcontext() if self.device.type == 'cpu' 
            else torch.amp.autocast(device_type=self.device.type, dtype=ptdtype)
        )
        
        self.logger.info(f"Training precision: {self.model_config.mixed_precision} ({ptdtype})")
        
        # TF32
        if self.model_config.use_tf32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("TF32 enabled")
        
        # Torch compile (NOT supported on Windows - Triton is Linux-only)
        is_windows = platform.system() == "Windows"
        if self.model_config.use_torch_compile and not is_windows:
            self.logger.info("Compiling model with torch.compile...")
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}")
        elif is_windows and self.model_config.use_torch_compile:
            self.logger.info("Skipping torch.compile (not supported on Windows - Triton requires Linux)")
        
        # Initialize GradScaler for mixed precision training
        if self.device.type == 'cuda' and self.model_config.mixed_precision in ['fp16', 'bf16']:
            # Note: GradScaler is mainly for fp16, bf16 doesn't need it
            if self.model_config.mixed_precision == 'fp16':
                self.scaler = torch.amp.GradScaler('cuda')
                self.logger.info("GradScaler enabled for FP16 training")
                
    def build_optimizer(self):
        """Build optimizer, scheduler, and loss function."""
        tc = self.training_config
        
        # Create optimizer based on config
        if tc.optimizer_type in ["sophia_g", "sophia_h"]:
            self.optimizer = create_sophia_optimizer(
                self.model,
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay,
                variant=tc.optimizer_type
            )
            self.logger.info(f"Using {tc.optimizer_type.upper()} optimizer")
        else:
            # Fallback to AdamW
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay
            )
            self.logger.info("Using AdamW optimizer")
        
        # Loss function
        if tc.use_z_loss:
            self.criterion = ZLossWrapper(z_coefficient=tc.z_loss_coefficient)
            self.logger.info(f"Using ZLoss with coefficient {tc.z_loss_coefficient}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Gradient accumulator
        self.accumulator = GradientAccumulator(
            accumulation_steps=tc.gradient_accumulation_steps
        )
        
        self.logger.info(f"Gradient accumulation steps: {tc.gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {tc.effective_batch_size}")
        
    def _get_max_steps(self) -> int:
        """Calculate maximum training steps based on mode."""
        tc = self.training_config
        
        if tc.mode == "fixed":
            return tc.max_steps
        elif tc.mode == "epochs":
            steps_per_epoch = len(self.train_data) // (self.model_config.sequence_length * tc.batch_size)
            return steps_per_epoch * tc.max_epochs
        elif tc.mode == "5h":
            # Estimate based on warmup runs
            return self._estimate_steps_for_duration(tc.target_hours * 3600)
        else:
            return tc.max_steps
            
    def _estimate_steps_for_duration(self, target_seconds: float) -> int:
        """Estimate steps possible within target duration using warmup."""
        tc = self.training_config
        
        self.logger.info(f"Estimating training speed...")
        warmup_iters = 10
        
        start_time = time.time()
        for _ in range(warmup_iters):
            xb, yb = self._get_batch('train')
            with self.autocast_ctx:
                logits = self.model(xb, yb)
                if tc.use_z_loss:
                    loss, _, _ = self.criterion(logits, yb, return_components=True)
                else:
                    loss = self.criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        warmup_duration = time.time() - start_time
        steps_per_sec = warmup_iters / warmup_duration
        max_steps = int(steps_per_sec * target_seconds)
        
        self.logger.info(f"Estimated speed: {steps_per_sec:.2f} steps/second")
        self.logger.info(f"Estimated total steps: {max_steps:,}")
        
        return max_steps
        
    def _get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data."""
        data = self.train_data if split == 'train' else self.val_data
        block_size = self.model_config.sequence_length
        batch_size = self.training_config.batch_size
        
        if len(data) <= block_size:
            raise ValueError(f"Data length ({len(data)}) <= block_size ({block_size})")
        
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        
        if self.device.type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            
        return x, y
    
    @torch.no_grad()
    def evaluate(self, num_batches: int = 10) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        for _ in range(num_batches):
            xb, yb = self._get_batch('val')
            with self.autocast_ctx:
                logits = self.model(xb, yb)
                if self.training_config.use_z_loss:
                    loss, ce_loss, z_loss = self.criterion(logits, yb, return_components=True)
                else:
                    loss = self.criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                    ce_loss = loss
                    z_loss = torch.tensor(0.0)
            total_loss += ce_loss.item()
        
        self.model.train()
        return {"val_loss": total_loss / num_batches}
    
    def save_checkpoint(self, name: str = "latest"):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model_config.to_dict(),
            'training_config': vars(self.training_config),
            'step': self.current_step,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping_counter,
            'tokens_processed': self.tokens_processed,
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'lr_history': self.lr_history,
        }
        
        if hasattr(self.scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.training_logger.log_checkpoint(self.current_epoch, self.current_step, str(checkpoint_path))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint and resume training state."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if available
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore training state
        self.current_step = checkpoint.get('step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        self.tokens_processed = checkpoint.get('tokens_processed', 0)
        
        # Restore history
        self.loss_history = checkpoint.get('loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        self.lr_history = checkpoint.get('lr_history', [])
        
        self.logger.info(f"Resumed from step {self.current_step}, epoch {self.current_epoch}")
        self.logger.info(f"Best val loss: {self.best_val_loss:.4f}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        history = {
            'loss': self.loss_history,
            'val_loss': self.val_loss_history,
            'lr': self.lr_history,
        }
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        self.logger.info(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Generate and save training plots."""
        if not HAS_MATPLOTLIB:
            self.logger.warning("matplotlib not installed, skipping plots")
            return
        
        plots_dir = self.checkpoint_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot 1: Loss curves
        fig, ax = plt.subplots(figsize=(12, 6))
        if self.loss_history:
            steps = [h['step'] for h in self.loss_history]
            losses = [h['loss'] for h in self.loss_history]
            ax.plot(steps, losses, label='Train Loss', alpha=0.7)
        if self.val_loss_history:
            val_steps = [h['step'] for h in self.val_loss_history]
            val_losses = [h['val_loss'] for h in self.val_loss_history]
            ax.plot(val_steps, val_losses, label='Val Loss', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(plots_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Plot 2: Learning rate
        if self.lr_history:
            fig, ax = plt.subplots(figsize=(12, 4))
            lr_steps = [h['step'] for h in self.lr_history]
            lrs = [h['lr'] for h in self.lr_history]
            ax.plot(lr_steps, lrs, color='green')
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)
            fig.savefig(plots_dir / 'lr_curve.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Plot 3: Combined plot
        if self.loss_history:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Loss
            steps = [h['step'] for h in self.loss_history]
            losses = [h['loss'] for h in self.loss_history]
            ax1.plot(steps, losses, label='Train', alpha=0.7)
            if self.val_loss_history:
                val_steps = [h['step'] for h in self.val_loss_history]
                val_losses = [h['val_loss'] for h in self.val_loss_history]
                ax1.plot(val_steps, val_losses, label='Val', linewidth=2)
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Training Progress')
            
            # LR
            if self.lr_history:
                lr_steps = [h['step'] for h in self.lr_history]
                lrs = [h['lr'] for h in self.lr_history]
                ax2.plot(lr_steps, lrs, color='green')
                ax2.set_ylabel('LR')
                ax2.set_xlabel('Step')
                ax2.grid(True, alpha=0.3)
            
            fig.tight_layout()
            fig.savefig(plots_dir / 'training_summary.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        self.logger.info(f"Training plots saved to {plots_dir}")
        
        
    def train(self):
        """Main training entry point - handles single or two-stage training."""
        tc = self.training_config
        
        if tc.dataset_stage == "pre+sft":
            # Two-stage training: pretraining then SFT
            self._train_two_stage()
        else:
            # Single stage training
            self._train_single_stage()
    
    def _train_two_stage(self):
        """Two-stage training: pretraining on base dataset, then SFT."""
        tc = self.training_config
        overall_start_time = time.time()
        
        self._log_system_info()
        self._log_config()
        
        self.logger.info("=" * 60)
        self.logger.info("TWO-STAGE TRAINING: PRETRAINING + SFT")
        self.logger.info(f"Stage 1 (Pretraining): {tc.pre_steps:,} steps")
        self.logger.info(f"Stage 2 (SFT): {tc.sft_steps:,} steps")
        self.logger.info("=" * 60)
        
        # ==================== STAGE 1: PRETRAINING ====================
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STAGE 1: PRETRAINING (Base Dataset)")
        self.logger.info("=" * 60)
        
        # Set stage to base for data preparation
        original_stage = tc.dataset_stage
        tc.dataset_stage = "base"
        self.prepare_data()
        self.build_model()
        self.build_optimizer()
        
        # Train pretraining stage
        self._run_training_loop(
            max_steps=tc.pre_steps,
            stage_name="Pretraining",
            learning_rate=tc.learning_rate
        )
        
        # Save pretraining checkpoint
        self.save_checkpoint("pretrained")
        self.logger.info("Pretraining stage completed. Checkpoint saved.")
        
        # Generate sample after pretraining
        self.logger.info("\n--- Sample after PRETRAINING ---")
        self._generate_sample(prompt="The")
        
        # ==================== STAGE 2: SFT ====================
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STAGE 2: SUPERVISED FINE-TUNING (SFT)")
        self.logger.info("=" * 60)
        
        # Prepare SFT data - IMPORTANT: reuse tokenizer from pretraining to avoid catastrophic forgetting!
        tc.dataset_stage = "sft"
        self.prepare_data(train_tokenizer=False)  # Use existing tokenizer
        
        # Reinitialize optimizer with lower learning rate for fine-tuning
        tc.learning_rate = tc.sft_lr
        self.build_optimizer()
        
        # Reset step counter for SFT stage
        sft_start_step = self.current_step
        
        # Train SFT stage
        self._run_training_loop(
            max_steps=tc.sft_steps,
            stage_name="SFT",
            learning_rate=tc.sft_lr
        )
        
        # Restore original stage
        tc.dataset_stage = original_stage
        
        # Final checkpoint
        self.save_checkpoint("final")
        
        # Save history and plots
        self.save_history()
        self.plot_training_curves()
        
        # Training summary
        total_time = time.time() - overall_start_time
        self.training_logger.training_summary()
        
        self.logger.info(f"\nTwo-stage training completed in {total_time / 3600:.2f} hours")
        self.logger.info(f"Total steps: {self.current_step} (Pre: {tc.pre_steps}, SFT: {tc.sft_steps})")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Generate sample text
        self._generate_sample()
    
    def _run_training_loop(self, max_steps: int, stage_name: str, learning_rate: float):
        """Run a training loop for specified number of steps."""
        tc = self.training_config
        
        # Initialize scheduler for this stage
        warmup_steps = int(max_steps * tc.warmup_ratio)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=max_steps,
            min_lr=tc.min_lr
        )
        
        self.logger.info(f"Starting {stage_name}...")
        self.logger.info(f"Max steps: {max_steps:,}, LR: {learning_rate:.2e}")
        self.logger.info(f"Warmup steps: {warmup_steps:,}")
        
        start_step = self.current_step
        self.start_time = time.time()  # Initialize start_time for tokens/sec calculation
        pbar = tqdm(range(max_steps), desc=stage_name)
        
        for local_step in pbar:
            step = start_step + local_step
            self.current_step = step
            
            # Initialize grad_norm for logging (will be updated when should_update)
            grad_norm = torch.tensor(0.0)
            
            # Get batch
            xb, yb = self._get_batch('train')
            
            # Forward pass with gradient accumulation
            with self.accumulator.accumulate(self.model):
                with self.autocast_ctx:
                    logits = self.model(xb, yb)
                    if tc.use_z_loss:
                        loss, ce_loss, z_loss = self.criterion(logits, yb, return_components=True)
                    else:
                        loss = self.criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                        ce_loss = loss
                        z_loss = torch.tensor(0.0)
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / tc.gradient_accumulation_steps
                
                # Backward with optional GradScaler
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                if self.accumulator.should_update():
                    # Unscale gradients for clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        tc.max_grad_norm
                    )
                    
                    # Hessian update for Sophia
                    if hasattr(self.optimizer, 'update_hessian') and local_step % 10 == 0:
                        self.optimizer.update_hessian()
                    
                    # Optimizer step with optional scaler
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
            
            # Track tokens processed
            self.tokens_processed += tc.batch_size * self.model_config.sequence_length
            
            # Logging
            if local_step % tc.log_interval == 0:
                lr = self.scheduler.get_lr()
                ppl = math.exp(min(ce_loss.item(), 20))  # Perplexity with cap
                
                # Calculate tokens/second
                elapsed = time.time() - self.start_time if self.start_time else 1.0
                tokens_per_sec = self.tokens_processed / max(elapsed, 1.0)
                
                metrics = {
                    "loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "z_loss": z_loss.item() if tc.use_z_loss else 0.0,
                    "perplexity": ppl,
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    "stage": stage_name,
                }
                
                self.loss_history.append({'step': step, 'loss': loss.item(), 'ppl': ppl, 'stage': stage_name})
                self.lr_history.append({'step': step, 'lr': lr})
                
                self.metrics_logger.log_metrics(metrics, step=step)
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{ppl:.1f}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                    "lr": f"{lr:.2e}"
                })
            
            # Evaluation
            if local_step > 0 and local_step % tc.eval_interval == 0:
                val_metrics = self.evaluate()
                val_metrics['val_ppl'] = math.exp(min(val_metrics['val_loss'], 20))
                self.metrics_logger.log_metrics(val_metrics, step=step, prefix="val/")
                
                self.val_loss_history.append({
                    'step': step, 
                    'val_loss': val_metrics['val_loss'], 
                    'val_ppl': val_metrics['val_ppl'],
                    'stage': stage_name
                })
                
                self.training_logger.log_validation(
                    epoch=self.current_epoch,
                    metrics=val_metrics,
                    is_best=val_metrics["val_loss"] < self.best_val_loss
                )
                
                # Check for improvement
                if val_metrics["val_loss"] < self.best_val_loss - tc.early_stopping_min_delta:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.early_stopping_counter = 0
                    self.save_checkpoint(f"best_{stage_name.lower()}")
                else:
                    self.early_stopping_counter += 1
                    if tc.early_stopping_patience > 0 and self.early_stopping_counter >= tc.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered after {self.early_stopping_counter} evaluations without improvement")
                        break
                
                self.training_logger.log_gpu_memory()
            
            # Checkpoint
            if local_step > 0 and local_step % tc.checkpoint_interval == 0:
                self.save_checkpoint(f"latest_{stage_name.lower()}")
        
        # Update current step to include all steps from this stage
        self.current_step = start_step + max_steps

    def _train_single_stage(self):
        """Single stage training loop."""
        tc = self.training_config
        
        # Initialize all components
        self._log_system_info()
        self._log_config()
        self.prepare_data()
        self.build_model()
        self.build_optimizer()
        
        # Calculate max steps
        max_steps = self._get_max_steps()
        
        # Initialize scheduler
        warmup_steps = int(max_steps * tc.warmup_ratio)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=max_steps,
            min_lr=tc.min_lr
        )
        
        # Resume from checkpoint if specified
        start_step = 0
        if tc.resume_from and Path(tc.resume_from).exists():
            self.load_checkpoint(tc.resume_from)
            start_step = self.current_step
            self.logger.info(f"Resuming training from step {start_step}")
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING")
        self.logger.info(f"Max steps: {max_steps:,}")
        self.logger.info(f"Start step: {start_step:,}")
        self.logger.info(f"Warmup steps: {warmup_steps:,}")
        self.logger.info(f"Eval interval: {tc.eval_interval}")
        self.logger.info("=" * 60)
        
        # Training loop
        start_time = time.time()
        target_seconds = tc.target_hours * 3600 if tc.mode == "5h" else None
        
        self.training_logger.log_epoch_start(1, max_steps // tc.eval_interval)
        
        pbar = tqdm(range(max_steps), desc="Training")
        for step in pbar:
            self.current_step = step
            
            # Initialize grad_norm for logging (will be updated when should_update)
            grad_norm = torch.tensor(0.0)
            
            # Get batch
            xb, yb = self._get_batch('train')
            
            # Forward pass with gradient accumulation
            with self.accumulator.accumulate(self.model):
                with self.autocast_ctx:
                    logits = self.model(xb, yb)
                    if tc.use_z_loss:
                        loss, ce_loss, z_loss = self.criterion(logits, yb, return_components=True)
                    else:
                        loss = self.criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                        ce_loss = loss
                        z_loss = torch.tensor(0.0)
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / tc.gradient_accumulation_steps
                
                # Backward with optional GradScaler
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                if self.accumulator.should_update():
                    # Unscale gradients for clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        tc.max_grad_norm
                    )
                    
                    # Hessian update for Sophia
                    if hasattr(self.optimizer, 'update_hessian') and step % 10 == 0:
                        self.optimizer.update_hessian()
                    
                    # Optimizer step with optional scaler
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
            
            # Check time limit
            if target_seconds and (step % 100 == 0) and (time.time() - start_time) > target_seconds:
                self.logger.info("Time limit reached. Stopping training.")
                break
            
            # Track tokens processed
            self.tokens_processed += tc.batch_size * self.model_config.sequence_length
            
            # Logging
            if step % tc.log_interval == 0:
                lr = self.scheduler.get_lr()
                ppl = math.exp(min(ce_loss.item(), 20))  # Perplexity with cap
                
                # Calculate tokens/second
                elapsed = time.time() - start_time
                tokens_per_sec = self.tokens_processed / max(elapsed, 1.0)
                
                metrics = {
                    "loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "z_loss": z_loss.item() if tc.use_z_loss else 0.0,
                    "perplexity": ppl,
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                }
                
                # Save to history
                self.loss_history.append({'step': step, 'loss': loss.item(), 'ppl': ppl})
                self.lr_history.append({'step': step, 'lr': lr})
                
                self.metrics_logger.log_metrics(metrics, step=step)
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{ppl:.1f}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                    "lr": f"{lr:.2e}"
                })
            
            # Evaluation
            if step > 0 and step % tc.eval_interval == 0:
                val_metrics = self.evaluate()
                val_metrics['val_ppl'] = math.exp(min(val_metrics['val_loss'], 20))
                self.metrics_logger.log_metrics(val_metrics, step=step, prefix="val/")
                
                # Save to history
                self.val_loss_history.append({
                    'step': step, 
                    'val_loss': val_metrics['val_loss'],
                    'val_ppl': val_metrics['val_ppl']
                })
                
                self.training_logger.log_validation(
                    epoch=self.current_epoch,
                    metrics=val_metrics,
                    is_best=val_metrics["val_loss"] < self.best_val_loss
                )
                
                # Check for improvement
                if val_metrics["val_loss"] < self.best_val_loss - tc.early_stopping_min_delta:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.early_stopping_counter = 0
                    self.save_checkpoint("best")
                else:
                    self.early_stopping_counter += 1
                    if tc.early_stopping_patience > 0 and self.early_stopping_counter >= tc.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered after {self.early_stopping_counter} evaluations without improvement")
                        break
                
                self.training_logger.log_gpu_memory()
            
            # Checkpoint
            if step > 0 and step % tc.checkpoint_interval == 0:
                self.save_checkpoint("latest")
        
        # Final checkpoint
        self.save_checkpoint("final")
        
        # Save history and plots
        self.save_history()
        self.plot_training_curves()
        
        # Training summary
        total_time = time.time() - start_time
        self.training_logger.training_summary()
        
        self.logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        self.logger.info(f"Final step: {self.current_step}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Generate sample text
        self._generate_sample()
        
    def _generate_sample(self, num_tokens: int = 100, prompt: Optional[str] = None):
        """Generate sample text from the model.
        
        Args:
            num_tokens: Number of tokens to generate.
            prompt: Optional prompt to start generation. If None, uses empty context for base
                   or a sample instruction for SFT models.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("GENERATING SAMPLE TEXT")
        self.logger.info("=" * 60)
        
        self.model.eval()
        
        # Determine prompt based on training stage
        tc = self.training_config
        if prompt is None:
            if tc.dataset_stage == "sft" or tc.dataset_stage == "pre+sft":
                # For SFT models, use an instruction prompt
                prompt = "[INST] Hello, how are you? [/INST]"
                self.logger.info(f"Using SFT prompt: {prompt}")
            else:
                # For base models, start with "The"
                prompt = "The"
                self.logger.info(f"Using base prompt: {prompt}")
        
        # Encode the prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        context = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            generated = self.model.generate(context, max_new_tokens=num_tokens, temperature=0.8, top_k=50)
        
        text = self.tokenizer.decode(generated[0].tolist())
        self.logger.info(f"Generated text:\n{text}")
        self.logger.info("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

# =============================================================================
# CONFIGURATION - All Hardcoded Parameters
# =============================================================================

# ----------------------
# MODEL ARCHITECTURE
# ----------------------
VOCAB_SIZE = 4000
SEQUENCE_LENGTH = 512
NUM_LAYERS = 6
N_HEADS = 6
N_KV_HEADS = 6  # For GQA, set lower than N_HEADS (e.g., 4 or 8)
N_EMBD = 384
DROPOUT_RATE = 0.1
MAX_POSITION_EMBEDDINGS = 512
ROPE_BASE = 10000

# Advanced model options
USE_CHECKPOINTS = True  # Gradient checkpointing
USE_DROP_PATH = False
DROP_PATH_RATE = 0.0
LAYER_SCALE_INIT = 1e-5  # Set to None to disable

# Relative position bias (T5-style)
USE_RELATIVE_POSITION_BIAS = False
REL_POS_NUM_BUCKETS = 32
REL_POS_MAX_DISTANCE = 128
REL_POS_BIDIRECTIONAL = False

# RoPE scaling
ROPE_SCALING = None  # Options: None, "linear", "dynamic"
ROPE_SCALING_FACTOR = 1.0

# ----------------------
# TRAINING MODE
# ----------------------
TRAINING_MODE = "fixed"  # Options: "5h" (time-based), "fixed" (step-based), "epochs"
MAX_STEPS = 1000
MAX_EPOCHS = 10
TARGET_HOURS = 5.0

# ----------------------
# DATASET SETTINGS
# ----------------------
DATASET_STAGE = "base"  # Options: "base", "mid", "sft", "pre+sft"
DATA_SIZE_MB = 1000

# Two-stage training settings (for "pre+sft" mode)
PRE_STEPS = 500
SFT_STEPS = 500
SFT_LR = 2e-5

# ----------------------
# TRAINING HYPERPARAMETERS
# ----------------------
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 8e-4
OPTIMIZER_TYPE = "sophia_g"  # Options: "sophia_g", "sophia_h", "adamw"
GRAD_CLIP = 1.0

# ----------------------
# EARLY STOPPING
# ----------------------
EARLY_STOPPING_PATIENCE = 0  # 0 = disabled

# ----------------------
# RESUME TRAINING
# ----------------------
RESUME_FROM = None  # Path to checkpoint or None

# ----------------------
# LOGGING
# ----------------------
LOG_DIR = "./logs"
DEBUG_MODE = False

# =============================================================================


def main():
    # Create model config from hardcoded values (no presets)
    model_config = ModelConfig(
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
    
    # Create training config from hardcoded values
    training_config = TrainingConfig(
        mode=TRAINING_MODE,
        max_steps=MAX_STEPS,
        max_epochs=MAX_EPOCHS,
        target_hours=TARGET_HOURS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        optimizer_type=OPTIMIZER_TYPE,
        max_grad_norm=GRAD_CLIP,
        dataset_stage=DATASET_STAGE,
        pre_steps=PRE_STEPS,
        sft_steps=SFT_STEPS,
        sft_lr=SFT_LR,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        resume_from=RESUME_FROM,
        data_size_mb=DATA_SIZE_MB if not DEBUG_MODE else 10,
        log_dir=LOG_DIR,
        debug=DEBUG_MODE,
    )
    
    # Create trainer and run
    trainer = Trainer(model_config, training_config)
    trainer.train()


if __name__ == "__main__":
    main()
