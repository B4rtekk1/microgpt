import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ZLoss(nn.Module):
    def __init__(self, coefficient: float = 1e-4):
        super().__init__()
        if coefficient < 0:
            raise ValueError(f"coefficient should be non-negative, got: {coefficient}")
        self.coefficient = coefficient
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.coefficient == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = self.coefficient * torch.mean(log_z ** 2)
        return z_loss
    def extra_repr(self) -> str:
        return f"coefficient={self.coefficient}"

class ZLossWrapper(nn.Module):
    def __init__(self, z_coefficient: float=1e-4, label_smoothing: float = 0.0, ignore_index: int = -100) -> None:
        super().__init__()
        self.z_loss = ZLoss(coefficient=z_coefficient)
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.z_coefficient = z_coefficient
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, return_components: bool = False) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        ce_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )

        z_loss = self.z_loss(logits)
        total_loss = ce_loss + z_loss
        if return_components:
            return total_loss, ce_loss, z_loss
        return total_loss
    
    def extra_repr(self) -> str:
        return (
            f"z_coefficient={self.z_coefficient}, "
            f"label_smoothing={self.label_smoothing}, "
            f"ignore_index={self.ignore_index}"
        )
    
def compute_z_loss(
        logits: torch.Tensor,
        coefficient: float = 1e-4
) -> torch.Tensor:
    if coefficient ==0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    log_z = torch.logsumexp(logits, dim=-1)
    return coefficient * torch.mean(log_z ** 2)