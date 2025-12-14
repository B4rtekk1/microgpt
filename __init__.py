__version__ = "0.1.0"
__author__ = "Bartosz Kasyna"

from .layers import (
    RMSNorm,
    SwiGLU,
    RotaryEmbedding,
    FusedQKVLinear,
    RelativePositionBias,
)

from .modules import (
    DropPath,
    LayerScale,
    KVCache,
)

from .training import (
    create_lr_scheduler,
    WarmupCosineScheduler,
    GradientAccumulator,
    initialize_weights,
    mark_residual_layers,
    count_params,
    model_size_mb,
    SophiaG,
    SophiaH,
    create_sophia_optimizer,
    ZLoss,
    ZLossWrapper,
    compute_z_loss,
)

from .model_config import (
    ModelConfig,
)

from .config import (
    GPUArchitecture,
    HardwareInfo,
    DistributedStrategy
)

from .logger import (
    Logger,
    TrainingLogger,
    MetricsLogger,
    get_logger,
)

__all__ = [
    # Version
    "__version__",
    
    # Layers
    "RMSNorm",
    "SwiGLU",
    "RotaryEmbedding",
    "FusedQKVLinear",
    "RelativePositionBias",
    
    # Modules
    "DropPath",
    "LayerScale",
    "KVCache",
    
    # Training
    "create_lr_scheduler",
    "WarmupCosineScheduler",
    "GradientAccumulator",
    "initialize_weights",
    "mark_residual_layers",
    "count_params",
    "model_size_mb",
    
    # Sophia optimizer
    "SophiaG",
    "SophiaH",
    "create_sophia_optimizer",
    
    # Z-loss
    "ZLoss",
    "ZLossWrapper",
    "compute_z_loss",
    
    # Config
    "ModelConfig",
    "HardwareInfo",
    "GPUArchitecture",
    "DistributedStrategy",

    "Logger",
    "TrainingLogger",
    "MetricsLogger",
    "get_logger",
]
