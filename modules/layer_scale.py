import torch
import torch.nn as nn

class LayerScale(nn.Module):
    """
    LayerScale for Transformer models.
    
    Learns a per-channel scaling factor for the output of a residual block.
    Improves training stability for deep transformers.
    
    Paper: "Going deeper with Image Transformers"
    Link: https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim: int, init_value: float = 1e-4, inplace: bool = False):
        """
        Initialize LayerScale.

        Args:
            dim (int): Dimension of the input features.
            init_value (float): Initial value for the scaling factor. Default: 1e-4.
            inplace (bool): Whether to perform the scaling in-place. Default: False.
        """
        super().__init__()
        self.dim = dim
        self.init_value = init_value
        self.inplace = inplace

        self.scale = nn.Parameter(torch.ones(dim) * init_value)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerScale.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, dim] or [batch_size, dim, ...].

        Returns:
            torch.Tensor: Scaled output of shape [batch_size, seq_len, dim].
        """
        # x shape: [batch_size, ..., dim]
        # self.scale shape: [dim]
        if self.inplace:
            return x.mul_(self.scale)
        return x * self.scale
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, init_value={self.init_value}"