import torch.nn as nn
from contextlib import contextmanager

class GradientAccumulator:
    def __init__(self, accumulation_steps: int = 1) -> None:
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    @contextmanager
    def accumulate(self, model: nn.Module):
        self.current_step += 1
        sync_gradients = self.should_update()
        if hasattr(model, 'no_sync') and not sync_gradients:
            with model.no_sync(): #type: ignore
                yield
        else:
            yield
    
    def should_update(self) -> bool:
        return self.current_step % self.accumulation_steps == 0
    
    def reset(self) -> None:
        self.current_step = 0
    
    @property
    def is_accumulating(self) -> bool:
        return not self.should_update()