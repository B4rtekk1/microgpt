import torch
from torch.optim import Optimizer
from typing import List, Optional, Tuple, Callable, Iterable
import math


class SophiaOptimized(Optimizer):
    
    def __init__(
        self,
        params: Iterable,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 1.0,
        weight_decay: float = 0.1,
        eps: float = 1e-12,
        maximize: bool = False,
        capturable: bool = False,
        min_numel_batched: int = 4096
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if rho < 0.0:
            raise ValueError(f"Invalid rho: {rho}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            capturable=capturable,
            min_numel_batched=min_numel_batched
        )
        super().__init__(params, defaults)
        
        self.hessian_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # if hasattr(torch, 'compile'):
        #     self._compiled_update = torch.compile(self._update_kernel, mode='max-autotune')
        # else:
        self._compiled_update = self._update_kernel
        
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
            group.setdefault('min_numel_batched', 4096)
    
    def _init_state(self, state, p):
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
    
    @staticmethod
    def _update_kernel(param, grad, exp_avg, hessian, beta1, beta1_comp, 
                       rho, lr, neg_lr, weight_decay, eps):
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)
        
        exp_avg.mul_(beta1).add_(grad, alpha=beta1_comp)
        
        hessian_safe = torch.clamp(hessian, min=eps)
        update = torch.clamp(exp_avg / hessian_safe, min=-rho, max=rho)
        param.add_(update, alpha=neg_lr)
    
    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta2 = group['betas'][1]
            beta2_comp = 1.0 - beta2
            min_numel = group['min_numel_batched']
            
            small_params = []
            large_params = []
            
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.numel() < min_numel:
                    small_params.append(p)
                else:
                    large_params.append(p)
            
            for p in large_params:
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, p)
                
                if torch.cuda.is_available() and p.is_cuda:
                    with torch.amp.autocast('cuda',dtype=torch.bfloat16): #type: ignore
                        grad_sq = p.grad.square()
                    state['hessian'].mul_(beta2).add_(
                        grad_sq.to(state['hessian'].dtype), 
                        alpha=beta2_comp
                    )
                else:
                    grad_sq = p.grad.square()
                    state['hessian'].mul_(beta2).add_(grad_sq, alpha=beta2_comp)
            
            if small_params:
                grads_flat = torch.cat([p.grad.flatten() for p in small_params])
                
                if torch.cuda.is_available() and grads_flat.is_cuda:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16): #type: ignore
                        grad_sq = grads_flat.square()
                    grad_sq = grad_sq.to(grads_flat.dtype)
                else:
                    grad_sq = grads_flat.square()
                
                offset = 0
                for p in small_params:
                    state = self.state[p]
                    if len(state) == 0:
                        self._init_state(state, p)
                    
                    numel = p.numel()
                    state['hessian'].mul_(beta2).add_(
                        grad_sq[offset:offset+numel].view_as(p),
                        alpha=beta2_comp
                    )
                    offset += numel
    
    @torch.no_grad()
    def update_hessian_async(self):
        if self.hessian_stream is None:
            return self.update_hessian()
        
        with torch.cuda.stream(self.hessian_stream):
            self.update_hessian()
    
    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta1_comp = 1 - beta1
            rho = group['rho']
            lr = group['lr']
            neg_lr = -lr
            weight_decay = group['weight_decay']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad if not group['maximize'] else -p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, p)
                
                exp_avg = state['exp_avg']
                hessian = state['hessian']
                state['step'] += 1
                
                self._compiled_update(
                    p, grad, exp_avg, hessian,
                    beta1, beta1_comp, rho, lr, neg_lr, 
                    weight_decay, eps
                )
        
        return loss


class SophiaG(SophiaOptimized):
    pass


class SophiaH(Optimizer):
    
    def __init__(
        self,
        params: Iterable,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 1.0,
        weight_decay: float = 0.1,
        eps: float = 1e-12,
        maximize: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if rho < 0.0:
            raise ValueError(f"Invalid rho: {rho}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize
        )
        super().__init__(params, defaults)
        
        # if hasattr(torch, 'compile'):
        #     self._compiled_update = torch.compile(self._update_kernel, mode='max-autotune')
        # else:
        self._compiled_update = self._update_kernel
    
    @staticmethod
    def _update_kernel(param, grad, exp_avg, hessian, beta1, beta1_comp,
                       rho, lr, neg_lr, weight_decay, eps):
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)
        
        exp_avg.mul_(beta1).add_(grad, alpha=beta1_comp)
        
        hessian_safe = torch.clamp(hessian, min=eps)
        update = torch.clamp(exp_avg / hessian_safe, min=-rho, max=rho)
        param.add_(update, alpha=neg_lr)
    
    @torch.no_grad()
    def update_hessian(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        num_samples: int = 1
    ):
        for group in self.param_groups:
            beta2 = group['betas'][1]
            beta2_comp = 1.0 - beta2
            
            for _ in range(num_samples):
                zs = {}
                for p in group['params']:
                    if p.requires_grad:
                        z = torch.randint_like(p, 0, 2) * 2 - 1
                        zs[p] = z.float()
                
                with torch.enable_grad():
                    loss = loss_fn()
                    grads = torch.autograd.grad(
                        loss,
                        [p for p in group['params'] if p.requires_grad],
                        create_graph=True
                    )
                
                hvp_grads = []
                for grad, (p, z) in zip(grads, zs.items()):
                    hvp = torch.autograd.grad(
                        (grad * z).sum(),
                        p,
                        retain_graph=True
                    )[0]
                    hvp_grads.append(hvp)
                
                for (p, z), hvp in zip(zs.items(), hvp_grads):
                    state = self.state[p]
                    
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['hessian'] = torch.zeros_like(p)
                    
                    hessian_diag = hvp * z
                    
                    state['hessian'].mul_(beta2).add_(
                        hessian_diag.abs(), 
                        alpha=beta2_comp / num_samples
                    )
    
    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1 = group['betas'][0]
            beta1_comp = 1 - beta1
            rho = group['rho']
            lr = group['lr']
            neg_lr = -lr
            weight_decay = group['weight_decay']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad if not group['maximize'] else -p.grad
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                hessian = state['hessian']
                state['step'] += 1
                
                self._compiled_update(
                    p, grad, exp_avg, hessian,
                    beta1, beta1_comp, rho, lr, neg_lr,
                    weight_decay, eps
                )
        
        return loss


def create_sophia_optimizer(
    model: torch.nn.Module,
    lr: float = 2e-4,
    betas: Tuple[float, float] = (0.965, 0.99),
    rho: float = 1.0,
    weight_decay: float = 0.1,
    variant: str = "sophia_g",
    exclude_from_decay: Optional[List[str]] = None
) -> Optimizer:
    
    default_exclude = ['bias', 'layernorm', 'rmsnorm', 'layer_norm', 'ln_', 'embedding']
    if exclude_from_decay:
        default_exclude.extend(exclude_from_decay)
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        name_lower = name.lower()
        if any(pattern in name_lower for pattern in default_exclude):
            no_decay_params.append(param)
        elif param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    if variant.lower() == "sophia_g":
        optimizer_cls = SophiaG
    elif variant.lower() == "sophia_h":
        optimizer_cls = SophiaH
    else:
        optimizer_cls = SophiaOptimized
    
    return optimizer_cls(
        param_groups,
        lr=lr,
        betas=betas,
        rho=rho,
        weight_decay=weight_decay
    )