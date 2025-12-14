import torch.nn as nn

def initialize_weights(
        module: nn.Module,
        n_layers: int = 1,
        initializer_range: float = 0.02,
        residual_scale: bool = False
    ) -> None:
    residual_std = initializer_range / (2 * n_layers) ** 0.5 if residual_scale else initializer_range
    
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)
            
        if hasattr(module, '_is_residual') and module._is_residual:
            nn.init.normal_(module.weight, mean=0.0, std=residual_std)
    
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
    
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def mark_residual_layers(model: nn.Module):
    residual_names = ['out_proj', 'c_proj', 'wo', 'w2', 'output_projection']
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_residual = any(name.endswith(rn) for rn in residual_names)
            module._is_residual = is_residual #type: ignore

def count_params(model: nn.Module, trainable_only : bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def model_size_mb(model: nn.Module, dtype_bytes: int = 2) -> float:
    """
    Estimate model size in MB.
    
    Args:
        model: Model to estimate
        dtype_bytes: Bytes per parameter (4 for fp32, 2 for fp16)
        
    Returns:
        Size in MB
    """
    return count_params(model, trainable_only=False) * dtype_bytes / (1024 * 1024)