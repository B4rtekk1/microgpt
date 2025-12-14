"""
Docstring for config
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import platform
import os
import json
import torch

class GPUArchitecture(Enum):
    """NVIDIA GPU Architectures with compute capabilities."""
    UNKNOWN = "unknown"
    KEPLER = "kepler"          # SM 3.x (GTX 600/700, Tesla K)
    MAXWELL = "maxwell"        # SM 5.x (GTX 900, Tesla M)
    PASCAL = "pascal"          # SM 6.x (GTX 1000, Tesla P)
    VOLTA = "volta"            # SM 7.0 (Tesla V100)
    TURING = "turing"          # SM 7.5 (RTX 2000, Tesla T4)
    AMPERE = "ampere"          # SM 8.x (RTX 3000, A100)
    ADA_LOVELACE = "ada"       # SM 8.9 (RTX 4000, L40)
    HOPPER = "hopper"          # SM 9.0 (H100)
    BLACKWELL = "blackwell"    # SM 10.0 (B100, B200)

@dataclass
class HardwareInfo:
    """
    Data class for storing hardware information and capabilities.

    Attributes:
        os_name: Name of the operating system (e.g., 'Linux', 'Windows').
        os_version: Version of the operating system.
        python_version: Python version.
        torch_version: PyTorch version.
        cuda_available: Whether CUDA is available.
        cuda_version: CUDA version string.
        cudnn_version: cuDNN version string.
        gpu_count: Number of available GPUs.
        gpu_names: List of GPU names.
        gpu_architectures: List of GPU architectures.
        compute_capabilities: List of compute capability tuples (major, minor).
        gpu_memory: List of total memory per GPU (in MB).
        total_gpu_memory: Total GPU memory across all devices (in MB).
        gpu_details: Detailed information for each GPU.
        supports_bf16: Whether hardware supports bfloat16 precision.
        supports_fp32: Whether hardware supports float32.
        supports_fdsp: Whether hardware supports Fully Sharded Data Parallel (FSDP).
        supports_torch_compile: Whether torch.compile is supported.
        supports_cuda_graphs: Whether CUDA Graphs are supported.
        supports_sdpa: Whether Scaled Dot-Product Attention is supported.
        supports_flash_attention: Whether Flash Attention is supported.
    """
    # System info
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    torch_version: str = ""

    # CUDA info
    cuda_available: bool = False
    cuda_version: str = ""
    cudnn_version: str = ""

    # GPU info
    gpu_count: int = 0
    gpu_names: List[str] | None = None
    gpu_architectures: List[GPUArchitecture] | None = None
    compute_capabilities: tuple | None = None
    gpu_memory: List[float] | None = None # MB
    total_gpu_memory: float = 0.0 # MB
    gpu_details: List[Dict[str, Any]] = field(default_factory=list)

    # Auto-detect hardware info
    supports_bf16: bool = False
    supports_fp32: bool = False
    supports_fdsp: bool = False
    supports_torch_compile: bool = False
    supports_cuda_graphs: bool = False
    supports_sdpa: bool = False # Scaled Dot-Product Attention
    supports_flash_attention: bool = False

    @classmethod
    def detect(cls) -> "HardwareInfo":
        """Auto-detect hardware information and capabilities.

        Returns:
            A HardwareInfo instance populated with system details.
        """
        info = cls()
        info.os_name = platform.system()
        info.os_version = platform.version()
        info.python_version = platform.python_version()
        info.torch_version = torch.__version__

        info.cuda_available = torch.cuda.is_available()
        if info.cuda_available:
            info.cuda_version = torch.version.cuda or ""
            if torch.backends.cudnn.is_available():
                info.cudnn_version = str(torch.backends.cudnn.version()) or ""

            info.gpu_count = torch.cuda.device_count()
            info.gpu_names = []
            info.gpu_architectures = []
            info.compute_capabilities = [] # type: ignore
            info.gpu_memory = []
            info.gpu_details = []

            for i in range(info.gpu_count):
                info.gpu_names.append(torch.cuda.get_device_name(i))
                props = torch.cuda.get_device_properties(i) 
                info.compute_capabilities.append((props.major, props.minor))# type: ignore
                info.gpu_memory.append(props.total_memory / (1024 ** 2)) # Convert to MB
                info.total_gpu_memory += props.total_memory / (1024 ** 2) # Convert to MB
                info.gpu_architectures = cls._detect_architecture(props.major, props.minor) # type: ignore
                gpu_detail = {
                    "index": i,
                    "name": props.name,
                    "total_memory_MB": props.total_memory / (1024 ** 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_gb": props.total_memory / (1024 ** 3),
                    "multiprocessors": props.multi_processor_count,
                }
                info.gpu_details.append(gpu_detail)
        info._detect_capabilities()
        return info
    
    @staticmethod
    def _detect_architecture(major: int, minor: int) -> GPUArchitecture:
        """Map compute capability to GPU architecture.

        Args:
            major: Major compute capability version.
            minor: Minor compute capability version.

        Returns:
            The corresponding GPUArchitecture enum member.
        """
        if major == 3:
            return GPUArchitecture.KEPLER
        elif major == 5:
            return GPUArchitecture.MAXWELL
        elif major == 6:
            return GPUArchitecture.PASCAL
        elif major == 7:
            if minor == 0:
                return GPUArchitecture.VOLTA
            elif minor == 5:
                return GPUArchitecture.TURING
        elif major == 8:
            if minor in [0, 6]:
                return GPUArchitecture.AMPERE
            elif minor == 9:
                return GPUArchitecture.ADA_LOVELACE
        elif major == 9:
            return GPUArchitecture.HOPPER
        elif major >= 10:
            return GPUArchitecture.BLACKWELL
        return GPUArchitecture.UNKNOWN

    def _detect_capabilities(self):
        """Detect supported features based on hardware and software versions.

        Populates flags like supports_flash_attention, supports_bf16, etc.
        """
        major, minor = self.compute_capabilities[0] if self.compute_capabilities else (0, 0)
        self.supports_flash_attention = (
            self.cuda_available and
            major >= 8 and
            self.os_name == "Linux" # Flash Attention requires Linux
        )

        self.supports_bf16 = self.cuda_available and major >= 8
        self.supports_fp32 = self.cuda_available and major >= 8
        self.supports_fdsp = self.cuda_available and self.gpu_count > 0 and major >= 8
        self.supports_torch_compile = torch.__version__ >= "2.0.0"
        self.supports_cuda_graphs = self.cuda_available and major >= 7
        self.supports_sdpa = self.cuda_available and torch.__version__ >= "2.0.0"

    def get_recommended_dtype(self) -> torch.dtype:
        """Get the recommended floating point data type.

        Returns:
            torch.bfloat16 if supported, else torch.float16 if CUDA is available,
            otherwise torch.float32.
        """
        if self.supports_bf16:
            return torch.bfloat16
        elif self.cuda_available:
            return torch.float16
        else:
            return torch.float32
    
    def get_attention_implementation(self) -> str:
        """Get the recommended attention implementation.

        Returns:
            'flash_attention_2' if supported, 'sdpa' if available, else 'eager'.
        """
        if self.supports_flash_attention:
            return "flash_attention_2"
        elif self.supports_sdpa:
            return "sdpa"
        else:
            return "eager"
    
    def summary(self) -> str:
        """Return a JSON summary of the hardware info.

        Returns:
            A pretty-printed JSON string containing the hardware details.
        """
        return json.dumps(self.__dict__, indent=4)

class DistributedStrategy(Enum):
    """
    Strategies for distributed training.
    """
    NONE = "none"
    DDP = "ddp"
    FDSP = "fdsp"
    DEEPSPEED = "deepspeed"