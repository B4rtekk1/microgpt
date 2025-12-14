"""
Docstring for layers.rmsnorm
"""

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    A variant of Layer Normalization where the mean is not subtracted, focusing only on re-scaling based on the 
    root mean square of the activations. This simplifies the computation and is often more stable for deep transformers.

    Attributes:
        dim: The dimension of the input tensor to normalize.
        eps: Small constant to avoid division by zero.
        weight: Learnable scaling parameter.
    
    References:
        - "Root Mean Square Layer Normalization" (Zhang AND Sennrich, 2019)
        - https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Input dimension.
            eps: Epsilon value for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute standard RMS norm."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS Norm to input.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of same shape as input.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight