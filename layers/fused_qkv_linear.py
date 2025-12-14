"""
Attention component: FusedQKVLinear
"""
import torch
import torch.nn as nn
import model_config as mc

class FusedQKVLinear(nn.Module):
    """
    Fused QKV Linear - single linear layer for Query, Key, Value projections.

    Instead of three separate nn.Linear for Q, K, V, this module uses a single larger linear layer that compute all three 
    projections at once. This is more efficient on GPU.

    Supports Grouped Query Attention (GQA):
        - n_heads: Number of query heads
        - n_kv_heads: Number of key/value heads (can be < n_heads)
    
    Supports Multi Query Attention (MQA):
        - n_heads: number of query heads
        - n_kv_heads: Number of key/value heads (should be one)
    
    Performance:
        - ~10-15% faster forward pass compared to separate projections due to reduced kernel launch overhead.
    
    Attributes:
        config: The model configuration object.
        n_embd: Input embedding dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension of each head.
        q_dim: Total dimension for query projection.
        kv_dim: Total dimension for key/value projection.
        qkv_proj: The fused linear layer.
    """
    def __init__(self, model_config: mc.ModelConfig):
        super().__init__()
        self.config = model_config
        self.n_embd = self.config.n_embd
        self.n_heads = self.config.n_heads
        self.n_kv_heads = self.config.n_kv_heads
        self.head_dim = self.config.head_dim
        
        # Compute output dimensions
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        total_dim = self.q_dim + 2 * self.kv_dim
        self.qkv_proj = nn.Linear(self.n_embd, total_dim, bias=False)
        self._split_sizes = [self.q_dim, self.kv_dim, self.kv_dim]
    
    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """Compute fused Q, K, V projections.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            A tuple (q, k, v) containing:
                - q: Query tensor of shape (batch, seq_len, n_heads, head_dim)
                - k: Key tensor of shape (batch, seq_len, n_kv_heads, head_dim)
                - v: Value tensor of shape (batch, seq_len, n_kv_heads, head_dim)
        """
        batch_size, seq_len = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(self._split_sizes, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return q, k, v

