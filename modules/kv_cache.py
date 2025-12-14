import torch

class KVCache:
    def __init__(
            self,
            max_batch_size: int,
            max_seq_len: int,
            n_kv_heads: int,
            head_dim: int,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
            dtype: torch.dtype = torch.float16
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        cache_shape = (max_batch_size, n_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=self.device, dtype=self.dtype)
        self.v_cache = torch.zeros(cache_shape, device=self.device, dtype=self.dtype)

        self.current_len = 0
    
    def update(self, k_new: torch.Tensor, v_new: torch.Tensor, position: int | None = None) -> tuple:
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
        bs = batch_size or self.max_batch_size
        return(
            self.k_cache[:bs, :, :self.current_len, :],
            self.v_cache[:bs, :, :self.current_len, :]
        )

    def reset(self) -> None:
        self.current_len = 0
    
    def resize(self, new_max_seq_len: int) -> None:
        if new_max_seq_len < self.max_seq_len:
            return
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