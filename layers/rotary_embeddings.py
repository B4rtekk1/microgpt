import torch
import torch.nn as nn
import model_config as mc

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).

    Applies absolute position information by rotating pairs of query and key elements 
    in the embedding space. This allows the model to extrapolate to sequence lengths 
    longer than those seen during training and maintains relative position information.

    Attributes:
        config: Model configuration.
        max_position_embeddings: Maximum sequence length supported for pre-computed caches.
        base: Base theta value for frequency calculation.
        head_dim: Dimension of each attention head.
        inv_freq: Inverse frequency tensor.
        cos_cached: Cached cosine values.
        sin_cached: Cached sine values.
    
    References:
        - "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
        - https://arxiv.org/abs/2104.09864
    """
    def __init__(self, model_config: mc.ModelConfig) -> None:
        """Initialize RotaryEmbedding so that inv_freq is computed.

        Args:
            model_config: Model configuration object.
        """
        super().__init__()
        self.config = model_config
        self.max_position_embeddings = model_config.max_position_embeddings
        self.base = model_config.base
        self.head_dim = model_config.n_embd // model_config.n_heads

        dim_indicies = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (dim_indicies / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(self.max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """Pre-compute cosine and sine values for rotation.

        Args:
            seq_len: Sequence length to compute cache for.
        """
        t = torch.arange(seq_len, dtype=self.inv_freq, device=self.inv_freq.device) #type: ignore
        freqs = torch.outer(t, self.inv_freq) #type: ignore
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple:
        """Return cached cosine and sine values for the given input sequence.

        Args:
            x: Input tensor (to determine sequence length).
            position_ids: Optional position IDs if not sequential.

        Returns:
            Tuple of (cos, sin) tensors tailored to the input sequence length.
        """
        seq_len = x.shape[1]
        if seq_len > self.cos_cached.shape[0]: #type: ignore
            self._set_cos_sin_cache(seq_len)
        if position_ids is not None:
            cos = self.cos_cached[position_ids] #type: ignore
            sin = self.sin_cached[position_ids] #type: ignore
        else:
            cos = self.cos_cached[:seq_len] #type: ignore
            sin = self.sin_cached[:seq_len] #type: ignore
        return cos, sin
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=1)
    
    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> tuple:
        """Apply RoPE to query and key tensors.

        Args:
            q: Query tensor.
            k: Key tensor.
            cos: Cosine tensor.
            sin: Sine tensor.

        Returns:
            Tuple of (q_embed, k_embed) with rotary embeddings applied.
        """
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        
        q_embed = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_pos={self.max_position_embeddings}, base={self.base}"
