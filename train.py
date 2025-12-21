"""
Automated training script for picoGPT.
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
    matplotlib.use('Agg')
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
from modules.layer_scale import LayerScale
from modules.drop_path import DropPath
from modules.kv_cache import KVCache

# Import model config
from model_config import ModelConfig

# Import training utilities
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

# Import loggers
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
from sft_normalizer import SFTNormalizer

# Import hardware detection
from config import HardwareInfo, GPUArchitecture

current_dir = Path(__file__).parent.absolute()

@dataclass
class PipelineStage:
    """Settings for a single stage in the training pipeline."""
    name: str
    dataset: str
    steps: int
    lr: float
    data_size_gb: Optional[float] = None
    eval_prompt: Optional[str] = None

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    mode: str = "fixed"
    max_steps: int = 1000
    max_epochs: int = 10
    target_hours: float = 5.0
    
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    optimizer_type: str = "sophia_g"
    learning_rate: float = 8e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    
    use_z_loss: bool = True
    z_loss_coefficient: float = 5e-4
    label_smoothing: float = 0.0
    
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500
    log_dir: str = "./logs"
    
    data_size_mb: int = 1000
    val_split: float = 0.1
    dataset_stage: str = "base"
    
    pipeline: List[PipelineStage] = field(default_factory=list)
    
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.001
    
    resume_from: Optional[str] = None
    debug: bool = False
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    def auto_configure(self, model_config: ModelConfig) -> None:
        """Auto-configure training parameters based on model size."""
        estimated_params = model_config.num_parameters_estimate
        
        if estimated_params > 1e9:
            self.learning_rate = 3e-4
            self.gradient_accumulation_steps = max(4, self.gradient_accumulation_steps)
        elif estimated_params > 500e6:
            self.learning_rate = 5e-4
            self.gradient_accumulation_steps = max(2, self.gradient_accumulation_steps)
        elif estimated_params > 100e6:
            self.learning_rate = 6e-4
        else:
            self.learning_rate = 8e-4
            
        if model_config.sequence_length > 2048:
            self.batch_size = max(8, self.batch_size // 2)
            
        if estimated_params > 500e6:
            self.warmup_ratio = 0.1

class CausalSelfAttention(nn.Module):
    """Causal Self-Attention with optional Relative Position Bias."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_heads
        self.n_kv_head = config.n_kv_heads
        self.head_dim = config.head_dim
        
        self.qkv_proj = FusedQKVLinear(config)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rotary = RotaryEmbedding(config)
        
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

        q, k, v = self.qkv_proj.forward(x)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rotary(x, position_ids)
        q, k = RotaryEmbedding.apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(T, T, x.device)
        
        if self.config.attention_implementation == "flash_attention_2":
            try:
                # Use new API for PyTorch 2.5+
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=(attn_bias is None)
                    )
            except (ImportError, RuntimeError):
                # Fallback for older PyTorch versions or unsupported hardware
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=(attn_bias is None)
                )
        elif self.config.attention_implementation == "sdpa":
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=(attn_bias is None)
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            if attn_bias is not None:
                att = att + attn_bias
            
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
    """Transformer block with Pre-LayerNorm, LayerScale, and DropPath."""
    
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.ln1 = RMSNorm(config.n_embd, eps=config.variance_epsilon)
        self.attn = CausalSelfAttention(config)
        
        if config.layer_scale_init is not None:
            self.ls1 = LayerScale(config.n_embd, config.layer_scale_init)
            self.ls2 = LayerScale(config.n_embd, config.layer_scale_init)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()
        
        if config.use_drop_path and config.drop_path_rate:
            drop_rate = config.drop_path_rate * layer_idx / max(config.num_layers - 1, 1)
            self.drop_path1 = DropPath(drop_rate)
            self.drop_path2 = DropPath(drop_rate)
        else:
            self.drop_path1 = nn.Identity()
            self.drop_path2 = nn.Identity()
        
        self.ln2 = RMSNorm(config.n_embd, eps=config.variance_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        attn_out = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + self.drop_path1(self.ls1(attn_out))
        
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.drop_path2(self.ls2(mlp_out))
        return x


class GPT(nn.Module):
    """GPT model using all project components."""
    
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
        self.transformer.wte.weight = self.lm_head.weight  # type: ignore

        self.apply(lambda m: initialize_weights(m, n_layers=config.num_layers))
        mark_residual_layers(self)
        
    def get_num_params(self, non_embedding: bool = True) -> int:
        return count_params(self, trainable_only=True)
    
    def get_model_size_mb(self, dtype_bytes: int = 2) -> float:
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
        
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        
        if kv_caches is None:
            kv_caches = [None] * len(self.transformer.h)  # type: ignore
        
        for block, cache in zip(self.transformer.h, kv_caches):  # type: ignore
            if use_checkpoint and self.config.use_checkpoints and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cache, use_reentrant=False)
            else:
                x = block(x, kv_cache=cache)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
        else:
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
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
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


class Trainer:
    """Unified trainer that uses all project components."""
    
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
        
        training_config.auto_configure(model_config)
        
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
        
        self.hardware_info = HardwareInfo.detect()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_data = None
        self.val_data = None
        self.tokenizer = None
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.accumulator = None
        
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        self.loss_history = []
        self.val_loss_history = []
        self.lr_history = []
        
        self.tokens_processed = 0
        self.start_time = None
        self.scaler = None
        
        # SFT dataset normalizer for unified formatting
        self.sft_normalizer = SFTNormalizer(add_eos=False)
        
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
        self.logger.info("\n" + self.model_config.get_optimization_summary())
        
    def _format_sft_sample(self, item: Dict[str, Any], dataset_name: Optional[str] = None) -> str:
        """
        Format SFT dataset item into instruction format using SFTNormalizer.
        
        Args:
            item: Dataset item dictionary
            dataset_name: Optional dataset name for format detection
            
        Returns:
            Formatted instruction string
        """
        return self.sft_normalizer.normalize(item, dataset_name)
    
    def prepare_data(self, train_tokenizer: bool = True):
        """Load and prepare data and tokenizer."""
        with self.logger.timer("Data Preparation"):
            loader = DatasetLoader()
            stage = self.training_config.dataset_stage
            # Sanitize stage name for Windows (no colons in folder names)
            cache_stage_name = stage.replace(":", "_")
            data_dir = current_dir / "data_cache" / cache_stage_name
            data_dir.mkdir(parents=True, exist_ok=True)
            
            train_bin = data_dir / "train.bin"
            val_bin = data_dir / "val.bin"
            
            if train_bin.exists() and val_bin.exists():
                self.logger.info(f"Loading cached {stage} dataset from {data_dir}...")
                train_data = np.fromfile(train_bin, dtype=np.uint16)
                val_data = np.fromfile(val_bin, dtype=np.uint16)
                
                min_required = self.model_config.sequence_length + 1
                if len(train_data) < min_required or len(val_data) < 2:
                    self.logger.warning(f"Cached data is too small. Discarding cache.")
                else:
                    self.train_data = train_data
                    self.val_data = val_data
                    self.logger.info(f"Loaded {len(self.train_data) + len(self.val_data):,} tokens from cache.")
                    
                    try:
                        self.tokenizer = Tokenizer()
                        self.logger.info("Loaded existing tokenizer.")
                        if self.tokenizer.vocab_size != self.model_config.vocab_size:
                            self.logger.info(f"Updating vocab_size from tokenizer: {self.model_config.vocab_size} -> {self.tokenizer.vocab_size}")
                            self.model_config.vocab_size = self.tokenizer.vocab_size
                    except Exception as e:
                        self.logger.warning(f"Could not load existing tokenizer: {e}")
                    
                    return

            if stage == "sft":
                self.logger.info("Loading SFT dataset (OpenAssistant)...")
            elif stage == "base":
                self.logger.info("Loading WikiText dataset for pretraining...")
            else:
                self.logger.info(f"Loading {stage.upper()} dataset for pretraining...")
            
            ds = loader.load(stage, streaming=True)
            
            target_size_bytes = self.training_config.data_size_mb * 1024 * 1024
            current_size = 0
            collected_texts = []
            
            iterator = ds
            if isinstance(ds, dict):
                iterator = ds.get('train', next(iter(ds.values())))
            
            pbar = tqdm(desc=f"Collecting {self.training_config.data_size_mb}MB", unit="MB")
            last_pbar_val = 0
            
            for item in iterator:
                # Use SFT formatting for all SFT-related datasets
                if "sft" in stage.lower() or "math" in stage.lower():
                    text = self._format_sft_sample(item, dataset_name=stage)
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
                    
            actual_size_mb = current_size / (1024*1024)
            self.logger.info(f"Collected {len(collected_texts)} samples, {actual_size_mb:.2f} MB")
            
            if actual_size_mb < self.training_config.data_size_mb - 1:
                self.logger.warning(f"Requested {self.training_config.data_size_mb}MB but only found {actual_size_mb:.2f}MB.")
            
            if train_tokenizer:
                tokenizer_corpus_path = current_dir / "tokenizer_corpus.txt"
                self.logger.info(f"Writing tokenizer corpus...")
                with open(tokenizer_corpus_path, "w", encoding="utf-8") as f:
                    for text in collected_texts:
                        if len(text.strip()) > 0:
                            f.write(text + "\n")
                
                try:
                    self.tokenizer = Tokenizer()
                except ImportError as e:
                    self.logger.error(f"Could not import custom Tokenizer: {e}")
                    raise
                
                self.logger.info(f"Training tokenizer (vocab size: {self.model_config.vocab_size})...")
                self.tokenizer.train(str(tokenizer_corpus_path), vocab_size=self.model_config.vocab_size)
                self.tokenizer.save()
                
                real_vocab_size = self.tokenizer.vocab_size
                if real_vocab_size != self.model_config.vocab_size:
                    self.logger.info(f"Updating vocab_size: {self.model_config.vocab_size} -> {real_vocab_size}")
                    self.model_config.vocab_size = real_vocab_size
            else:
                if self.tokenizer is None:
                    try:
                        self.tokenizer = Tokenizer()
                        self.logger.info(f"Loaded existing tokenizer (vocab size: {self.tokenizer.vocab_size})")
                    except Exception:
                        raise ValueError("Cannot reuse tokenizer - no tokenizer found!")
                else:
                    self.logger.info(f"Reusing existing tokenizer (vocab size: {self.tokenizer.vocab_size})")
            
            self.logger.info(f"Tokenizing {stage.upper()} dataset...")
            all_tokens = []
            for text in tqdm(collected_texts, desc="Tokenizing"):
                if len(text) > 0:
                    encoded = self.tokenizer.encode(text)
                    all_tokens.extend(encoded)
            
            data = np.array(all_tokens, dtype=np.uint16)
            self.logger.info(f"Total tokens: {len(data):,}")
            
            n = int((1 - self.training_config.val_split) * len(data))
            self.train_data = data[:n]
            self.val_data = data[n:]
            
            self.logger.info(f"Caching tokenized data to {data_dir}...")
            self.train_data.tofile(train_bin)
            self.val_data.tofile(val_bin)
            self.logger.info("Successfully cached dataset.")
                
    def build_model(self):
        """Build and configure the model."""
        with self.logger.timer("Model Building"):
            self.model = GPT(self.model_config)
            self.model.to(self.device)
            
            self.arch_logger.log_model_summary(self.model, depth=3)
            
            num_params = self.model.get_num_params()
            size_mb = self.model.get_model_size_mb()
            self.logger.info(f"Model parameters: {num_params / 1e6:.2f}M")
            self.logger.info(f"Model size: {size_mb:.2f} MB")
            
            estimated_vram = self.model_config.estimate_vram_gb(
                batch_size=self.training_config.batch_size,
                include_optimizer_states=True
            )
            self.logger.info(f"Estimated VRAM usage: {estimated_vram:.2f} GB")
            
            self._apply_optimizations()
            
    def _apply_optimizations(self):
        """Apply hardware-based optimizations."""
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
        
        if self.model_config.use_tf32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("TF32 enabled")
        
        is_windows = platform.system() == "Windows"
        if self.model_config.use_torch_compile and not is_windows:
            self.logger.info("Compiling model with torch.compile...")
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}")
        elif is_windows and self.model_config.use_torch_compile:
            self.logger.info("Skipping torch.compile (not supported on Windows)")
        
        if self.device.type == 'cuda' and self.model_config.mixed_precision in ['fp16', 'bf16']:
            if self.model_config.mixed_precision == 'fp16':
                self.scaler = torch.amp.GradScaler('cuda')
                self.logger.info("GradScaler enabled for FP16 training")
                
    def build_optimizer(self):
        """Build optimizer, scheduler, and loss function."""
        tc = self.training_config
        
        if tc.optimizer_type in ["sophia_g", "sophia_h"]:
            self.optimizer = create_sophia_optimizer(
                self.model,
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay,
                variant=tc.optimizer_type
            )
            self.logger.info(f"Using {tc.optimizer_type.upper()} optimizer")
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay
            )
            self.logger.info("Using AdamW optimizer")
        
        if tc.use_z_loss:
            self.criterion = ZLossWrapper(z_coefficient=tc.z_loss_coefficient)
            self.logger.info(f"Using ZLoss with coefficient {tc.z_loss_coefficient}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.accumulator = GradientAccumulator(
            accumulation_steps=tc.gradient_accumulation_steps
        )
        
        self.logger.info(f"Gradient accumulation steps: {tc.gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {tc.effective_batch_size}")
            
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_step = checkpoint.get('step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        self.tokens_processed = checkpoint.get('tokens_processed', 0)
        
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
        
        self.logger.info(f"Training plots saved to {plots_dir}")
        
        
    def train(self):
        """Main training entry point."""
        tc = self.training_config
        
        if tc.dataset_stage == "pipeline":
            self._train_pipeline()
        else:
            self.logger.error("Only pipeline mode is supported in this version!")
    
    def _train_pipeline(self):
        """Multi-stage training pipeline."""
        tc = self.training_config
        overall_start_time = time.time()
        
        if not tc.pipeline:
            self.logger.error("Pipeline mode selected but no pipeline defined!")
            return
            
        self._log_system_info()
        self._log_config()
        
        self.logger.info("=" * 60)
        self.logger.info(f"MULTI-STAGE PIPELINE TRAINING ({len(tc.pipeline)} stages)")
        for i, stage in enumerate(tc.pipeline):
            self.logger.info(f"Stage {i+1}: {stage.name} ({stage.steps:,} steps, LR: {stage.lr:.2e}, Dataset: {stage.dataset})")
        self.logger.info("=" * 60)
        
        # IMPORTANT: Prepare data FIRST to get the correct vocab_size from tokenizer
        # This must happen before building the model!
        first_stage = tc.pipeline[0]
        tc.dataset_stage = first_stage.dataset
        if first_stage.data_size_gb is not None:
            tc.data_size_mb = int(first_stage.data_size_gb * 1024)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PREPARING INITIAL DATA AND TOKENIZER")
        self.logger.info("=" * 60)
        self.prepare_data(train_tokenizer=True)
        
        # NOW build model with correct vocab_size from tokenizer
        self.build_model()
        
        for i, stage in enumerate(tc.pipeline):
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"STARTING STAGE {i+1}/{len(tc.pipeline)}: {stage.name.upper()}")
            self.logger.info("=" * 60)
            
            tc.dataset_stage = stage.dataset
            tc.learning_rate = stage.lr
            if stage.data_size_gb is not None:
                tc.data_size_mb = int(stage.data_size_gb * 1024)
            
            # For first stage, data is already prepared above
            if i > 0:
                self.prepare_data(train_tokenizer=False)
            
            self.build_optimizer()
            
            self._run_training_loop(
                max_steps=stage.steps,
                stage_name=stage.name,
                learning_rate=stage.lr
            )
            
            safe_name = stage.name.lower().replace(" ", "_")
            self.save_checkpoint(f"stage_{i+1}_{safe_name}")
            self.logger.info(f"Stage {i+1} completed. Checkpoint saved.")
            
            self.logger.info(f"\n--- Samples after {stage.name} ---")
            self._generate_sample(prompt=stage.eval_prompt)
            
        total_time = time.time() - overall_start_time
        self.save_checkpoint("final_pipeline")
        self.save_history()
        self.plot_training_curves()
        self.training_logger.training_summary()
        
        self.logger.info(f"\nMulti-stage pipeline completed in {total_time / 3600:.2f} hours")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def _run_training_loop(self, max_steps: int, stage_name: str, learning_rate: float):
        """Run a training loop for specified number of steps."""
        tc = self.training_config
        
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
        self.start_time = time.time()
        pbar = tqdm(range(max_steps), desc=stage_name)
        
        for local_step in pbar:
            step = start_step + local_step
            self.current_step = step
            
            grad_norm = torch.tensor(0.0)
            xb, yb = self._get_batch('train')
            
            with self.accumulator.accumulate(self.model):
                with self.autocast_ctx:
                    logits = self.model(xb, yb)
                    if tc.use_z_loss:
                        loss, ce_loss, z_loss = self.criterion(logits, yb, return_components=True)
                    else:
                        loss = self.criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                        ce_loss = loss
                        z_loss = torch.tensor(0.0)
                
                scaled_loss = loss / tc.gradient_accumulation_steps
                
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                if self.accumulator.should_update():
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        tc.max_grad_norm
                    )
                    
                    if hasattr(self.optimizer, 'update_hessian') and local_step % 10 == 0:
                        self.optimizer.update_hessian()
                    
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
            
            self.tokens_processed += tc.batch_size * self.model_config.sequence_length
            
            if local_step % tc.log_interval == 0:
                lr = self.scheduler.get_lr()
                ppl = math.exp(min(ce_loss.item(), 20))
                
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
                
                if val_metrics["val_loss"] < self.best_val_loss - tc.early_stopping_min_delta:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.early_stopping_counter = 0
                    self.save_checkpoint(f"best_{stage_name.lower()}")
                else:
                    self.early_stopping_counter += 1
                    if tc.early_stopping_patience > 0 and self.early_stopping_counter >= tc.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered")
                        break
                
                self.training_logger.log_gpu_memory()
            
            if local_step > 0 and local_step % tc.checkpoint_interval == 0:
                self.save_checkpoint(f"latest_{stage_name.lower()}")
        
        self.current_step = start_step + max_steps

    def _generate_sample(self, num_tokens: int = 100, prompt: Optional[str] = None):
        """Generate sample text from the model."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("GENERATING SAMPLE TEXT")
        self.logger.info("=" * 60)
        
        self.model.eval()
        
        tc = self.training_config
        if prompt is None:
            if tc.dataset_stage == "sft" or tc.dataset_stage == "pre+sft":
                prompt = "[INST] Hello, how are you? [/INST]"
                self.logger.info(f"Using SFT prompt: {prompt}")
            else:
                prompt = "The"
                self.logger.info(f"Using base prompt: {prompt}")
        
        prompt_tokens = self.tokenizer.encode(prompt)
        context = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            generated = self.model.generate(context, max_new_tokens=num_tokens, temperature=0.8, top_k=50)
        
        text = self.tokenizer.decode(generated[0].tolist())
        self.logger.info(f"Generated text:\n{text}")
        self.logger.info("=" * 60)


# =============================================================================
# CONFIGURATION
# =============================================================================

VOCAB_SIZE = 4000
SEQUENCE_LENGTH = 512
NUM_LAYERS = 6
N_HEADS = 6
N_KV_HEADS = 6
N_EMBD = 384
DROPOUT_RATE = 0.1
MAX_POSITION_EMBEDDINGS = 512
ROPE_BASE = 10000

USE_CHECKPOINTS = True
USE_DROP_PATH = False
DROP_PATH_RATE = 0.0
LAYER_SCALE_INIT = 1e-5

USE_RELATIVE_POSITION_BIAS = False
REL_POS_NUM_BUCKETS = 32
REL_POS_MAX_DISTANCE = 128
REL_POS_BIDIRECTIONAL = False

ROPE_SCALING = None
ROPE_SCALING_FACTOR = 1.0

TRAINING_MODE = "fixed"
MAX_STEPS = 1000
MAX_EPOCHS = 10
TARGET_HOURS = 5.0

DATASET_STAGE = "pipeline"
DATA_SIZE_GB = None  # Use per-stage sizes in pipeline
DATA_SIZE_MB = 100   # Default: 100MB (for quick tests)

# Pipeline with LOCAL datasets (use after running download_datasets.py)
# Change "local:" prefix to use HuggingFace directly if needed
TRAINING_PIPELINE = [
    # Pretraining on FineWeb2 (10GB available locally)
    PipelineStage(
        name="Language & Facts", 
        dataset="local:fineweb2",  # Use local dataset
        steps=50000, 
        lr=8e-4, 
        data_size_gb=0.1,  # 100MB - change to 10.0 for full training
        eval_prompt="in 1996"
    ),
    # Math reasoning (50MB available locally)
    PipelineStage(
        name="Mathematics", 
        dataset="local:math-sft",  # Use local dataset
        steps=10000, 
        lr=4e-4, 
        data_size_gb=0.05,  # 50MB
        eval_prompt="2+2*2="
    ),
    # Instruction tuning (50MB available locally)
    PipelineStage(
        name="SFT", 
        dataset="local:sft-ultra",  # Use local dataset
        steps=10000, 
        lr=2e-5, 
        data_size_gb=0.05,  # 50MB
        eval_prompt="[INST] What is the capital of France? [/INST]"
    ),
    # Chain-of-thought (50MB available locally)
    PipelineStage(
        name="Chain-of-Thought", 
        dataset="local:math-sft-plus",  # Use local dataset
        steps=10000, 
        lr=1e-5, 
        data_size_gb=0.05,  # 50MB
        eval_prompt="[INST] Solve 15*13. <|thought|> [/INST]"
    ),
]

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 8e-4
OPTIMIZER_TYPE = "sophia_g"
GRAD_CLIP = 1.0

EARLY_STOPPING_PATIENCE = 0

RESUME_FROM = None

LOG_DIR = "./logs"
DEBUG_MODE = False


def main():
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
        pipeline=TRAINING_PIPELINE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        resume_from=RESUME_FROM,
        data_size_mb=DATA_SIZE_MB if DATA_SIZE_GB is None else int(DATA_SIZE_GB * 1024),
        log_dir=LOG_DIR,
        debug=DEBUG_MODE,
    )
    
    trainer = Trainer(model_config, training_config)
    trainer.train()


if __name__ == "__main__":
    main()
