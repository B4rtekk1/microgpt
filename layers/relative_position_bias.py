import torch
import torch.nn as nn
import model_config as mc

class RelativePositionBias(nn.Module):
    """
    Relative Position Bias - learnable bias based on relative distance.

    This module computes a learnable bias that is added to attention scores based on the relative position between query and key tokens.

    Uses bucketing for longer distances (Inspired by T5).

    Attributes:
        num_heads: Number of attention heads.
        num_buckets: Number of unique position buckets.
        max_distance: Maximum relative distance to consider.
        bidirectional: Whether to use bidirectional (encoder) or causal (decoder) bias.
    
    References:
        - "Exploring the limits of Transfer Learning with T5"
        - https://arxiv.org/abs/1910.10683
    """

    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128, bidirectional: bool = False) -> None:
        """Initialize RelativePositionBias.

        Args:
            num_heads: Number of attention heads.
            num_buckets: Number of bucket positions.
            max_distance: Maximum distance to consider for bucketing.
            bidirectional: If True, uses bidirectional relative positions (for encoders).
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        nn.init.normal_(self.relative_attention_bias.weight, mean=0.0, std = 0.02)
    
    def _relative_position_buckets(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Map relative positions to bucket indices.

        Args:
            relative_position: Tensor of relative positions.

        Returns:
            Tensor of bucket indices with same shape as input.
        """
        relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            relative_buckets += (relative_position < 0).long() * num_buckets
            relative_position = relative_position.abs()
        else:
            relative_position = -torch.clamp(relative_position, max=0)
            num_buckets = self.num_buckets
        
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / torch.log(torch.tensor(self.max_distance / max_exact))
            * (num_buckets - max_exact)).long()
        relative_position_if_large = torch.clamp(relative_position_if_large, max=num_buckets-1)
        relative_buckets += torch.where(is_small, relative_position.long(), relative_position_if_large)
        return relative_buckets
    
    def forward(self, query_length: int, key_length: int, device: torch.device):
        """Compute relative position bias.

        Args:
            query_length: Length of query sequence.
            key_length: Length of key sequence.
            device: Device to create tensors on.

        Returns:
            Bias tensor of shape (1, num_heads, query_length, key_length).
        """
        if device is None:
            device = self.relative_attention_bias.weight.device
        
        query_position = torch.arange(query_length, dtype=torch.long, device=device)
        key_position = torch.arange(key_length, dtype=torch.long, device=device)

        relative_position = query_position[:, None] - key_position[None, :]
        relative_position_bucket = self._relative_position_buckets(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, num_buckets={self.num_buckets}, "
            f"max_distance={self.max_distance}, bidirectional={self.bidirectional}"
        )