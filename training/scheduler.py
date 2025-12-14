import math
import torch.optim as optim

def create_lr_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str = "cosine",
        num_training_steps: int = 1000000,
        num_warmup_steps: int = 2000,
        min_lr_ratio: float = 0.1,
        num_cycles: float = 0.5
):
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        
        if scheduler_type == "cosine":
            cosine_decay = 0.5 * (1 + math.cos(math.pi * num_cycles * 2.0 * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
        
        elif scheduler_type == "linear":
            return max(min_lr_ratio, 1.0 - progress)
        
        elif scheduler_type == "constant":
            return 1.0
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class WarmupCosineScheduler:
    """
    Warmup + Cosine annealing scheduler (manual implementation).
    
    Useful when you need more control over the schedule.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        max_lr: float | None = None
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = max_lr or optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Compute current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
    
    def state_dict(self) -> dict:
        return {"current_step": self.current_step}
    
    def load_state_dict(self, state_dict: dict):
        self.current_step = state_dict["current_step"]