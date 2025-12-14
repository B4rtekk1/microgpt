import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            keep_prob = 1.0 - self.p
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
            return x * mask / keep_prob
        return x
    
    def extra_repr(self) -> str:
        return f"p={self.p}"