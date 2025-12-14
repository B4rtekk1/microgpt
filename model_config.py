from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch
from config import HardwareInfo
import json

@dataclass
class ModelConfig:
    """
    Configuration class for the Transformer model.

    Attributes:
        vocab_size: Size of the vocabulary.
        sequence_length: Maximum sequence length (context window).
        num_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads (for Grouped Query Attention).
        n_embd: Embedding dimension.
        dropout_rate: Dropout probability.
        max_position_embeddings: Maximum absolute position index (for learnable embeddings).
        base: Base limit for RoPE theta calculations.
        variance_epsilon: Epsilon value for layer normalization.
        use_checkpoints: Whether to use gradient checkpointing.
        use_drop_path: Whether to use stochastic depth (drop path).
        use_relative_position_bias: Whether to use relative position bias.
        rel_pos_num_buckets: Number of buckets for relative position bias.
        rel_pos_max_distance: Maximum distance for relative position bias.
        rel_pos_bidirectional: Whether relative position bias is bidirectional.
        sliding_window_size: Size of the sliding window attention (if applicable).
        rope_scaling: Type of RoPE scaling ('linear', 'dynamic', or None).
        rope_scaling_factor: Factor for RoPE scaling.
        use_fused_qkv: Whether to use fused Query-Key-Value projection.
        attention_implementation: Method for attention ('flash_attention', 'sdpa', 'standard', 'auto').
        distibuted_strategy: Distributed training strategy ('ddp', 'fdsp', 'none', 'auto').
        use_torch_compile: Whether to compile the model using torch.compile.
        mixed_precision: Mixed precision mode ('bf16', 'fp16', 'fp32', 'auto').
        use_tf32: Whether to use TF32 on Ampere+ GPUs.
        use_cuda_graphs: Whether to use CUDA Graphs.
        fdsp_sharding_strategy: Sharding strategy for FSDP.
        fdsp_cpu_offload: Whether to offload FSDP params to CPU.
        fdsp_backward_prefetch: Prefetching policy for FSDP backward pass.
        world_size: Number of distributed processes.
    """
    vocab_size: int
    sequence_length: int
    num_layers: int
    n_heads: int
    n_kv_heads: int
    n_embd: int

    dropout_rate: float
    max_position_embeddings: int
    base: int

    variance_epsilon: float = 1e-9
    use_checkpoints: Optional[bool] = True
    use_drop_path: Optional[bool] = False

    use_relative_position_bias: Optional[bool] = False
    rel_pos_num_buckets: Optional[int] = 32
    rel_pos_max_distance: Optional[int] = 128
    rel_pos_bidirectional: Optional[bool] = False

    sliding_window_size: Optional[int] = 256

    rope_scaling: Optional[str] = None  # e.g., "linear", "dynamic"
    rope_scaling_factor: Optional[float] = 1.0 # Scaling factor for RoPE (e.g., 2.0 for doubling the context length)

    use_fused_qkv: Optional[bool] = True
    attention_implementation: Optional[str] = "auto"  # e.g., "flash_attention", "sdpa", "standard"
    distributed_strategy: Optional[str] = "auto"  # e.g., "ddp", "fdsp", "deepspeed", "none"
    use_torch_compile: Optional[bool] = None # Whether to use torch.compile (if None, auto-detect based on hardware)
    mixed_precision: Optional[str] = "auto"  # e.g., "bf16", "fp16", "fp32", "auto"
    use_tf32: Optional[bool] = False  # Whether to use TF32 for matrix multiplications on Ampere GPUs
    use_cuda_graphs: Optional[bool] = None # Whether to use CUDA Graphs (if None, auto-detect based on hardware)
    fdsp_sharding_strategy: Optional[str] = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fdsp_cpu_offload: Optional[bool] = False  # Whether to offload parameters to CPU in FDSP
    fdsp_backward_prefetch: Optional[str] = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST, NONE
    world_size: Optional[int] = 1  # Number of processes for distributed training

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_dim = self.n_embd // self.n_heads

        if self.sequence_length <= 0 or self.sequence_length > self.max_position_embeddings:
            raise ValueError("sequence_length must be positive and less than or equal to max_position_embeddings, got {self.sequence_length}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.n_kv_heads <= 0 or self.n_kv_heads > self.n_heads:
            raise ValueError(f"n_kv_heads must be positive and less than or equal to n_heads, got {self.n_kv_heads}")
        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be positive, got {self.n_embd}")
        
        if self.dropout_rate < 0.0 or self.dropout_rate > 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self.dropout_rate}")
        
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads must be divisible by n_kv_heads, got n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}")
        if self.n_embd % self.n_heads != 0:
            raise ValueError(f"n_embd must be divisible by n_heads, got n_embd={self.n_embd}, n_heads={self.n_heads}")
        if self.n_embd % self.n_kv_heads != 0:
            raise ValueError(f"n_embd must be divisible by n_kv_heads, got n_embd={self.n_embd}, n_kv_heads={self.n_kv_heads}")
        if self.rope_scaling not in (None, "linear", "dynamic"):
            raise ValueError(f"rope_scaling must be one of None, 'linear', or 'dynamic', got {self.rope_scaling}")
        if self.attention_implementation not in ("auto", "flash_attention", "sdpa", "standard"):
            raise ValueError(f"attention_implementation must be one of 'auto', 'flash_attention', 'sdpa', or 'standard', got {self.attention_implementation}")
        if self.distributed_strategy not in ("auto", "ddp", "fdsp", "deepspeed", "none"):
            raise ValueError(f"distributed_strategy must be one of 'auto', 'ddp', 'fdsp', 'deepspeed', or 'none', got {self.distributed_strategy}")
        if self.mixed_precision not in ("bf16", "fp16", "fp32", "auto"):
            raise ValueError(f"mixed_precision must be one of 'bf16', 'fp16', 'fp32', or 'auto', got {self.mixed_precision}")
        if self.fdsp_sharding_strategy not in ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"):
            raise ValueError(f"fdsp_sharding_strategy must be one of 'FULL_SHARD', 'SHARD_GRAD_OP', or 'NO_SHARD', got {self.fdsp_sharding_strategy}")
        if self.fdsp_backward_prefetch not in ("BACKWARD_PRE", "BACKWARD_POST", "NONE"):
            raise ValueError(f"fdsp_backward_prefetch must be one of 'BACKWARD_PRE', 'BACKWARD_POST', or 'NONE', got {self.fdsp_backward_prefetch}")
        if self.rope_scaling and self.rope_scaling_factor <= 0.0: #type: ignore
            raise ValueError(f"rope_scaling_factor must be positive when rope_scaling is set, got {self.rope_scaling_factor}")
        if self.sliding_window_size <= 0: #type: ignore
            raise ValueError(f"sliding_window_size must be positive, got {self.sliding_window_size}")
        
        self._configure_hardware_optimizations()
    
    def _configure_hardware_optimizations(self):
        """Auto-configure optimization flags based on detected hardware."""
        self.hardware_info = HardwareInfo.detect()
        self.world_size = int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1
        if self.attention_implementation == "auto":
            self.attention_implementation = self.hardware_info.get_attention_implementation()
        valid_attention = ["flash_attention_2", "sdpa", "eager"]
        if self.attention_implementation not in valid_attention:
            raise ValueError(f"Invalid attention implementation selected: {self.attention_implementation}. Valid options are: {valid_attention}")
        if self.attention_implementation == "flash_attention_2" and not self.hardware_info.supports_flash_attention:
            raise ValueError("Flash Attention 2 is not supported on the current hardware configuration.")
        if self.head_dim > 256 and self.attention_implementation == "flash_attention_2":
            raise ValueError(f"Flash Attention 2 supports head_dim up to 256, got {self.head_dim}")
        if self.distributed_strategy == "auto":
            if self.world_size > 1:
                self.distributed_strategy = "fdsp" if self.hardware_info.supports_fdsp else "ddp"
            else:
                self.distributed_strategy = "none"
        valid_dist = ["ddp", "fdsp", "deepspeed", "none"]
        if self.distributed_strategy not in valid_dist:
            raise ValueError(f"Invalid distributed strategy selected: {self.distributed_strategy}. Valid options are: {valid_dist}")
        if self.use_torch_compile is None:
            self.use_torch_compile = self.hardware_info.supports_torch_compile
            if self.mixed_precision == "auto":
                self.mixed_precision = "bf16" if self.hardware_info.supports_bf16 else "fp16"
            else:
                self.mixed_precision = "float32"
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        if self.use_cuda_graphs is None:
            self.use_cuda_graphs = self.hardware_info.supports_cuda_graphs
    
    def get_optimization_summary(self) -> str:
        """Get a summary of selected optimization strategies.
        
        Returns:
            A formatted string containing device, attention implementation,
            distributed strategy, precision, and other optimization flags.
        """
        summary = (
            f"Device: {self.device}\n"
            f"Attention Implementation: {self.attention_implementation}\n"
            f"Distributed Strategy: {self.distributed_strategy}\n"
            f"Mixed Precision: {self.mixed_precision}\n"
            f"Use Torch Compile: {self.use_torch_compile}\n"
            f"Use CUDA Graphs: {self.use_cuda_graphs}\n"
        )
        return summary
    
    @property
    def num_parameters_estimate(self) -> int:
        """Estimate total number of parameters in the model.

        Returns:
            The estimated count of parameters (including embeddings).
        """
        embed_params = self.vocab_size * self.n_embd
        
        # Per layer: attention + FFN
        # Attention: 3 projections (Q, K, V) + output projection
        attn_params = 4 * self.n_embd * self.n_embd  # Simplified
        
        # FFN (SwiGLU): 3 projections * hidden_dim
        hidden_dim = int(2 * (4 * self.n_embd) / 3)
        ffn_params = 3 * self.n_embd * hidden_dim
        
        # Layer norms (2 per layer) - small
        norm_params = 2 * self.n_embd
        
        layer_params = attn_params + ffn_params + norm_params
        total_layers = self.num_layers * layer_params
        
        # Output projection (usually shared with embedding)
        output_params = self.vocab_size * self.n_embd
        
        return embed_params + total_layers + output_params
    
    @property
    def estimated_flops_per_token(self) -> float:
        """Estimate FLOPs required per token during forward pass.

        Returns:
            Estimated FLOPs per token (approximate).
        """
        attn_flops = 2 * self.n_embd * self.sequence_length * self.sequence_length
        ffn_flops = 8 * self.n_embd * (2 * (4 * self.n_embd) / 3)
        total_flops_per_layer = attn_flops + ffn_flops
        return self.num_layers * total_flops_per_layer
    
    @property
    def estimated_flops_per_sequence(self) -> float:
        """Estimate FLOPs required for a full sequence forward pass.

        Returns:
            Estimated FLOPs per sequence.
        """
        return self.estimated_flops_per_token * self.sequence_length
    
    def estimate_vram_gb(
            self,
            batch_size: int,
            include_optimizer_states: bool = False,
    ):
        """Estimate VRAM usage in Gigabytes.

        Args:
            batch_size: The batch size for training/inference.
            include_optimizer_states: Whether to include optimizer overhead (e.g., for Adam).

        Returns:
            Estimated VRAM usage in GB.
        """
        bytes_per_param = 2 if self.mixed_precision in ("fp16", "bf16") else 4
        model_params = self.num_parameters_estimate * bytes_per_param
        grad_bytes = model_params if include_optimizer_states else 0
        activation_bytes = (
            batch_size
            * self.sequence_length
            * self.n_embd
            * bytes_per_param
            * 4  # Approximation factor for activations
        )
        if self.use_checkpoints:
            import math
            effective_layers = max(1, int(math.sqrt(self.num_layers)))
        else:
            effective_layers = self.num_layers
        activation_bytes = activation_bytes * effective_layers
        total_bytes = model_params + grad_bytes + activation_bytes
        total_gb = total_bytes / (1024 ** 3)
        return total_gb
    
    def to_dict(self, include_hardware: bool = False) -> Dict[str, Any]:
        """
        Export configuration to dictionary (for checkpoints/serialization).
        
        Args:
            include_hardware: Include hardware-specific settings
        
        Returns:
            Dictionary with configuration values
        """
        # Fields to always exclude (computed at runtime)
        exclude = {'device', 'head_dim', 'hardware_info', 'torch_dtype'}
        
        if not include_hardware:
            # Also exclude hardware-dependent auto-configured values
            exclude.update({
                'attention_implementation',
                'distributed_strategy', 
                'use_torch_compile',
                'mixed_precision_dtype',
                'use_tf32',
                'use_cuda_graphs',
                'world_size'
            })
        
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_') or key in exclude:
                continue
            # Convert non-serializable types
            if hasattr(value, 'value'):  # Enum
                value = value.value
            elif hasattr(value, '__dict__'):  # Complex objects
                continue
            result[key] = value
        
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
        
        Returns:
            ModelConfig instance
        """
        # Fields that are computed, not passed to __init__
        exclude = {
            'device', 'head_dim', 'hardware_info', 'torch_dtype',
            'num_parameters_estimate', 'estimated_flops_per_token',
            'estimated_flops_per_sequence'
        }
        
        # Filter valid fields
        valid_fields = {k: v for k, v in config_dict.items() if k not in exclude}
        
        return cls(**valid_fields)
    
    def save(self, path: str, include_hardware: bool = False) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save the configuration
            include_hardware: Include hardware-specific settings
        """
        config_dict = self.to_dict(include_hardware=include_hardware)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to the configuration file
        
        Returns:
            ModelConfig instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    # =========================================================================
    # Preset Configurations
    # =========================================================================
    
    @classmethod
    def gpt2_small(cls, **kwargs) -> "ModelConfig":
        """
        GPT-2 Small configuration (~124M parameters).
        
        Architecture:
            - 12 layers, 12 heads, 768 embedding dim
            - Standard Multi-Head Attention (no GQA)
        
        Returns:
            ModelConfig for GPT-2 Small
        """
        defaults = {
            'vocab_size': 50257,
            'sequence_length': 1024,
            'num_layers': 12,
            'n_heads': 12,
            'n_kv_heads': 12,
            'n_embd': 768,
            'dropout': 0.1,
            'max_position_embeddings': 1024,
            'base': 10000,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def gpt2_medium(cls, **kwargs) -> "ModelConfig":
        """
        GPT-2 Medium configuration (~350M parameters).
        
        Architecture:
            - 24 layers, 16 heads, 1024 embedding dim
        
        Returns:
            ModelConfig for GPT-2 Medium
        """
        defaults = {
            'vocab_size': 50257,
            'sequence_length': 1024,
            'num_layers': 24,
            'n_heads': 16,
            'n_kv_heads': 16,
            'n_embd': 1024,
            'dropout': 0.1,
            'max_position_embeddings': 1024,
            'base': 10000,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def gpt2_large(cls, **kwargs) -> "ModelConfig":
        """
        GPT-2 Large configuration (~774M parameters).
        
        Architecture:
            - 36 layers, 20 heads, 1280 embedding dim
        
        Returns:
            ModelConfig for GPT-2 Large
        """
        defaults = {
            'vocab_size': 50257,
            'sequence_length': 1024,
            'num_layers': 36,
            'n_heads': 20,
            'n_kv_heads': 20,
            'n_embd': 1280,
            'dropout': 0.1,
            'max_position_embeddings': 1024,
            'base': 10000,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def gpt2_xl(cls, **kwargs) -> "ModelConfig":
        """
        GPT-2 XL configuration (~1.5B parameters).
        
        Architecture:
            - 48 layers, 25 heads, 1600 embedding dim
        
        Returns:
            ModelConfig for GPT-2 XL
        """
        defaults = {
            'vocab_size': 50257,
            'sequence_length': 1024,
            'num_layers': 48,
            'n_heads': 25,
            'n_kv_heads': 25,
            'n_embd': 1600,
            'dropout': 0.1,
            'max_position_embeddings': 1024,
            'base': 10000,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def llama_7b(cls, **kwargs) -> "ModelConfig":
        """
        LLaMA 7B-style configuration (~7B parameters).
        
        Architecture:
            - 32 layers, 32 heads, 4096 embedding dim
            - Uses Grouped Query Attention (8 KV heads)
            - No dropout (as in original LLaMA)
        
        Returns:
            ModelConfig for LLaMA-7B style model
        """
        defaults = {
            'vocab_size': 32000,
            'sequence_length': 2048,
            'num_layers': 32,
            'n_heads': 32,
            'n_kv_heads': 8,  # GQA
            'n_embd': 4096,
            'dropout': 0.0,
            'max_position_embeddings': 4096,
            'base': 10000,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def llama_13b(cls, **kwargs) -> "ModelConfig":
        """
        LLaMA 13B-style configuration (~13B parameters).
        
        Architecture:
            - 40 layers, 40 heads, 5120 embedding dim
            - Uses Grouped Query Attention (8 KV heads)
        
        Returns:
            ModelConfig for LLaMA-13B style model
        """
        defaults = {
            'vocab_size': 32000,
            'sequence_length': 2048,
            'num_layers': 40,
            'n_heads': 40,
            'n_kv_heads': 8,  # GQA
            'n_embd': 5120,
            'dropout': 0.0,
            'max_position_embeddings': 4096,
            'base': 10000,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def mistral_7b(cls, **kwargs) -> "ModelConfig":
        """
        Mistral 7B-style configuration (~7B parameters).
        
        Architecture:
            - 32 layers, 32 heads, 4096 embedding dim
            - Uses Grouped Query Attention (8 KV heads)
            - Sliding Window Attention (4096 tokens)
        
        Returns:
            ModelConfig for Mistral-7B style model
        """
        defaults = {
            'vocab_size': 32000,
            'sequence_length': 4096,
            'num_layers': 32,
            'n_heads': 32,
            'n_kv_heads': 8,  # GQA
            'n_embd': 4096,
            'dropout': 0.0,
            'max_position_embeddings': 32768,
            'base': 10000,
            'sliding_window_size': 4096,  # Sliding window attention
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def tiny(cls, **kwargs) -> "ModelConfig":
        """
        Tiny model for testing and debugging (~10M parameters).
        
        Architecture:
            - 4 layers, 4 heads, 256 embedding dim
            - Fast to train, useful for code verification
        
        Returns:
            ModelConfig for tiny test model
        """
        defaults = {
            'vocab_size': 32000,
            'sequence_length': 512,
            'num_layers': 4,
            'n_heads': 4,
            'n_kv_heads': 4,
            'n_embd': 256,
            'dropout': 0.1,
            'max_position_embeddings': 512,
            'base': 10000,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def small(cls, **kwargs) -> "ModelConfig":
        """
        Small model for efficient training (~50M parameters).
        
        Architecture:
            - 8 layers, 8 heads, 512 embedding dim
            - Good balance of speed and capability
        
        Returns:
            ModelConfig for small model
        """
        defaults = {
            'vocab_size': 32000,
            'sequence_length': 1024,
            'num_layers': 8,
            'n_heads': 8,
            'n_kv_heads': 8,
            'n_embd': 512,
            'dropout': 0.1,
            'max_position_embeddings': 1024,
            'base': 10000,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def medium(cls, **kwargs) -> "ModelConfig":
        """
        Medium model (~200M parameters).
        
        Architecture:
            - 16 layers, 16 heads, 1024 embedding dim
            - Uses GQA for efficiency
        
        Returns:
            ModelConfig for medium model
        """
        defaults = {
            'vocab_size': 32000,
            'sequence_length': 2048,
            'num_layers': 16,
            'n_heads': 16,
            'n_kv_heads': 4,  # GQA
            'n_embd': 1024,
            'dropout': 0.0,
            'max_position_embeddings': 2048,
            'base': 10000,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def large(cls, **kwargs) -> "ModelConfig":
        """
        Large model (~500M parameters).
        
        Architecture:
            - 24 layers, 24 heads, 1536 embedding dim
            - Uses GQA for efficiency
        
        Returns:
            ModelConfig for large model
        """
        defaults = {
            'vocab_size': 32000,
            'sequence_length': 2048,
            'num_layers': 24,
            'n_heads': 24,
            'n_kv_heads': 6,  # GQA
            'n_embd': 1536,
            'dropout': 0.0,
            'max_position_embeddings': 4096,
            'base': 10000,
        }
        defaults.update(kwargs)
        return cls(**defaults)
