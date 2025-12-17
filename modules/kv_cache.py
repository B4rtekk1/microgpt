import torch

class KVCache:
    """
    Key-Value Cache for efficient autoregressive generation in Transformers.
    
    Stores the Key and Value tensors for past tokens to avoid recomputing them
    at each generation step. This significantly speeds up decoding.
    
    Concept: "Fast Transformer Decoding: One Write-Head is All You Need"
    Link: https://arxiv.org/abs/1911.02150 (discusses decoding efficiency)
    General idea from: "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
    """
    def __init__(
            self,
            max_batch_size: int,
            max_seq_len: int,
            n_kv_heads: int,
            head_dim: int,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
            dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the Key-Value Cache.

        Args:
            max_batch_size (int): Maximum batch size to support.
            max_seq_len (int): Maximum sequence length to cache.
            n_kv_heads (int): Number of Key/Value heads.
            head_dim (int): Dimension of each head.
            device (torch.device): Device to store the cache on.
            dtype (torch.dtype): Data type of the cache.
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        cache_shape = (max_batch_size, n_kv_heads, max_seq_len, head_dim)
        # k_cache shape: [max_batch_size, n_kv_heads, max_seq_len, head_dim]
        self.k_cache = torch.zeros(cache_shape, device=self.device, dtype=self.dtype)
        # v_cache shape: [max_batch_size, n_kv_heads, max_seq_len, head_dim]
        self.v_cache = torch.zeros(cache_shape, device=self.device, dtype=self.dtype)

        self.current_len = 0
    
    def update(self, k_new: torch.Tensor, v_new: torch.Tensor, position: int | None = None) -> tuple:
        """
        Update the cache with new key/value pairs.

        Args:
            k_new (torch.Tensor): New keys. Shape: [batch, n_kv_heads, new_seq_len, head_dim]
            v_new (torch.Tensor): New values. Shape: [batch, n_kv_heads, new_seq_len, head_dim]
            position (int | None): Starting position for update. If None, appends to current end.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The updated cache up to the current length.
                - k_cache: [batch, n_kv_heads, current_len, head_dim]
                - v_cache: [batch, n_kv_heads, current_len, head_dim]
        """
        batch_size, _, new_seq_len, _ = k_new.shape
        
        if position is None:
            position = self.current_len
        
        end_pos = position + new_seq_len
        if end_pos > self.max_seq_len:
            raise ValueError(f"Cache overflow: position {position} + new_len {new_seq_len} > max_seq_len {self.max_seq_len}")
        
        self.k_cache[:batch_size, :, position:end_pos, :] = k_new
        self.v_cache[:batch_size, :, position:end_pos, :] = v_new
        
        self.current_len = max(self.current_len, end_pos)
        
        return (
            self.k_cache[:batch_size, :, :self.current_len, :],
            self.v_cache[:batch_size, :, :self.current_len, :]
        )

    def get(self, batch_size: int | None = None) -> tuple:
        """
        Get the current content of the cache.
        
        Args:
            batch_size: Optional batch size to slice the cache (default: max_batch_size).

        Returns:
            Tuple of:
            - k: [batch_size, n_kv_heads, current_len, head_dim]
            - v: [batch_size, n_kv_heads, current_len, head_dim]
        """
        bs = batch_size or self.max_batch_size
        return(
            self.k_cache[:bs, :, :self.current_len, :],
            self.v_cache[:bs, :, :self.current_len, :]
        )

    def reset(self) -> None:
        """
        Reset the cache pointer to 0. Effectively clears the cache.
        """
        self.current_len = 0
    
    def resize(self, new_max_seq_len: int) -> None:
        """
        Resize the cache to a new maximum sequence length.
        
        Args:
            new_max_seq_len (int): New maximum sequence length.
        """
        if new_max_seq_len < self.max_seq_len:
            return
        
        # New shape: [max_batch_size, n_kv_heads, new_max_seq_len, head_dim]
        new_shape = (self.max_batch_size, self.n_kv_heads, new_max_seq_len, self.head_dim)
        new_k_cache = torch.zeros(new_shape, device=self.device, dtype=self.dtype)
        new_v_cache = torch.zeros(new_shape, device=self.device, dtype=self.dtype)

        if self.current_len > 0:
            new_k_cache[:, :, :self.current_len, :] = self.k_cache[:, :, :self.current_len, :]
            new_v_cache[:, :, :self.current_len, :] = self.v_cache[:, :, :self.current_len, :]
        
        self.k_cache = new_k_cache
        self.v_cache = new_v_cache
        self.max_seq_len = new_max_seq_len
    
    @property
    def memory_usage_mb(self) -> float:
        bytes_per_element = self.k_cache.element_size()
        total_elements = 2 * self.k_cache.numel()
        return total_elements * bytes_per_element / (1024 * 1024)
    
    def __repr__(self) -> str:
        return (
            f"KVCache(batch={self.max_batch_size}, seq_len={self.max_seq_len}, "
            f"n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
            f"current_len={self.current_len}, memory={self.memory_usage_mb:.1f}MB)"
        )