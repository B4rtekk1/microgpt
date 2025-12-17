import torch
import torch.nn as nn

class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) regularization.
    
    Drops paths (samples) in the batch with a configured probability.
    Used in deep residual networks to improve training convergence and generalization.
    
    Paper: "Deep Networks with Stochastic Depth"
    Link: https://arxiv.org/abs/1603.09382
    """
    def __init__(self, p: float = 0.5) -> None:
        """
        Initialize DropPath.

        Args:
            p (float): Probability of dropping a path. Default: 0.5.
        """
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DropPath during training.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with paths dropped.
        """
        if self.training:
            # x shape: [batch_size, ...]
            keep_prob = 1.0 - self.p
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            # shape: [batch_size, 1, 1, ...] (matches x dim count)
            
            mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
            # mask shape: [batch_size, 1, 1, ...]
            
            return x * mask / keep_prob
        return x
    
    def extra_repr(self) -> str:
        return f"p={self.p}"