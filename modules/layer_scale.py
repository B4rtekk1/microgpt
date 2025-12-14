import torch
import torch.nn as nn

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-4, inplace: bool = False):
        super().__init__()
        self.dim = dim
        self.init_value = init_value
        self.inplace = inplace

        self.scale = nn.Parameter(torch.ones(dim) * init_value)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return x.mul_(self.scale)
        return x * self.scale
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, init_value={self.init_value}"