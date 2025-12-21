"""Logging utilities and helpers for training and model inspection.

This module provides a small logging framework built on top of Python's
`logging` module with colored console output, file logging, metric
buffering, and specialized loggers for model architecture inspection and
training workflows (including optional integrations for W&B and
TensorBoard).

Key classes:
- `Logger` / `TrainingLogger` / `MetricsLogger`: high-level loggers used in
    training scripts.
- `ArchitectureLogger`: helpers to summarize and visualize model
    architectures (PyTorch-aware).
- `MetricBuffer`: lightweight storage for computing rolling means of
    metrics.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import contextmanager
import ctypes
import os

class Colors:
    """ANSI color codes for terminal output."""

    Reset = "\033[0m"

    #------Regular Colors------#
    Black = "\033[030m"        # Black
    Red = "\033[031m"          # Red
    Green = "\033[032m"        # Green
    Yellow = "\033[033m"       # Yellow
    Blue = "\033[034m"         # Blue
    Magenta = "\033[035m"      # Magenta
    Cyan = "\033[036m"         # Cyan
    White = "\033[037m"        # White

    #------Bold Colors------#
    BoldBlack = "\033[1;30m"   # Black
    BoldRed = "\033[1;31m"     # Red
    BoldGreen = "\033[1;32m"   # Green
    BoldYellow = "\033[1;33m"  # Yellow
    BoldBlue = "\033[1;34m"    # Blue
    BoldMagenta = "\033[1;35m" # Magenta
    BoldCyan = "\033[1;36m"    # Cyan
    BoldWhite = "\033[1;37m"   # White

    #------Background Colors------#
    BGBlack = "\033[40m"       # Black
    BGRed = "\033[41m"         # Red
    BGGreen = "\033[42m"       # Green
    BGYellow = "\033[43m"      # Yellow
    BGBlue = "\033[44m"        # Blue

    #-------Dim Colors-------#
    DimBlack = "\033[2;30m"    # Black

    @staticmethod
    def colored(text: str, color_code: str) -> str:
        """Wrap text with ANSI color codes.

        Args:
            text: The text to colorize.
            color_code: The ANSI color code to apply.

        Returns:
            The text wrapped in the color code and reset code.
        """
        return f"{color_code}{text}{Colors.Reset}"
    
    @staticmethod
    def is_terminal() -> bool:
        """Check if the output is a terminal.

        Returns:
            True if stdout is a terminal, False otherwise.
        """
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter to add colors based on log level.
    """

    LEVEL_COLOR_MAP = {
        logging.DEBUG: Colors.DimBlack,
        logging.INFO: Colors.Green,
        logging.WARNING: Colors.Yellow,
        logging.ERROR: Colors.Red,
        logging.CRITICAL: Colors.BoldRed,
        logging.FATAL: Colors.BoldRed,
        logging.NOTSET: Colors.White,
    }

    def __init__(self, fmt: Optional[str] = None, use_colors: bool = True):
        """Initialize the formatter.

        Args:
            fmt: Optional format string for the formatter. If None, the
                default format returned by `_default_log_format` is used.
            use_colors: Whether to enable ANSI color codes in the output.
        """
        super().__init__(fmt or self._default_log_format())
        self.use_colors = use_colors and Colors.is_terminal()

    def _default_log_format(self) -> str:
        """Return the formatter string for default logging output.

        The default format includes the timestamp, logger name, level
        and message fields.

        Returns:
            The format string.
        """
        return "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Format time and colorize if enabled."""
        s = super().formatTime(record, datefmt)
        if self.use_colors:
            color = self.LEVEL_COLOR_MAP.get(record.levelno, Colors.White)
            return Colors.colored(s, color)
        return s

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record and apply colors when enabled.

        Temporarily modifies the record's `levelname` to
        include ANSI color codes based on the log level, then restores
        the original values before returning the formatted string.

        Args:
            record: The log record to format.

        Returns:
            The formatted log message string.
        """
        # Save original values
        original_levelname = record.levelname
        
        if self.use_colors:
            color = self.LEVEL_COLOR_MAP.get(record.levelno, Colors.White)
            record.levelname = Colors.colored(str(record.levelname), color)

        # Call super which will call formatTime (now colorized) and formatMessage
        try:
            result = super().format(record)
        finally:
            # Restore original values to avoid affecting other handlers
            record.levelname = original_levelname
            
        return result


class Logger(ABC):
    """
    Abstract base class for loggers.
    """

    def __init__(
            self,
            name: Optional[str] = "Logger",
            level: int = logging.INFO, 
            rank: int = 0, 
            log_to_console: bool = True, 
            log_to_file: bool = True,
            log_dir: Optional[Union[str, Path]] = None, 
            ):
        """Create a new Logger instance.

        Args:
            name: Name associated with this logger (used for file names and
                logging records).
            level: Logging level threshold for the logger.
            rank: Process rank (used to enable logging only on rank 0 for
                multi-process setups).
            log_to_console: If True, attach a console handler.
            log_to_file: If True and `log_dir` is provided, attach a file
                handler to store logs on disk.
            log_dir: Optional directory path where logs will be saved.
        """
        
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.level = level
        self.rank = rank
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file and (self.log_dir is not None)

        # Determine if this process should log
        self._should_log = self.rank == 0

        # Setup logger
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.level)
        self._logger.propagate = False  # Prevent double logging
        self._logger.handlers = []  # Clear existing handlers

        self._setup_handlers()
        self._start_time = time.time()

    def _setup_handlers(self) -> None:
        """Setup console and file handlers.

        Initializes the StreamHandler for console output (with coloring)
        and FileHandler for file output (if enabled).
        """
        if self._should_log:
            if self.log_to_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(self.level)
                console_formatter = ColoredFormatter(use_colors=True)
                console_handler.setFormatter(console_formatter)
                self._logger.addHandler(console_handler)

            if self.log_to_file and self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = self.log_dir / f"{self.name}.log"
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setLevel(self.level)
                file_formatter = logging.Formatter(
                    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                self._logger.addHandler(file_handler)
                self.log_file_path = log_file
                
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a debug message.

        Args:
            msg: The message to log.
            *args: Variable length argument list for the logger.
            **kwargs: Arbitrary keyword arguments for the logger.
        """
        if self._should_log:
            self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an info message.

        Args:
            msg: The message to log.
            *args: Variable length argument list for the logger.
            **kwargs: Arbitrary keyword arguments for the logger.
        """
        if self._should_log:
            self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a warning message.

        Args:
            msg: The message to log.
            *args: Variable length argument list for the logger.
            **kwargs: Arbitrary keyword arguments for the logger.
        """
        if self._should_log:
            self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an error message.

        Args:
            msg: The message to log.
            *args: Variable length argument list for the logger.
            **kwargs: Arbitrary keyword arguments for the logger.
        """
        if self._should_log:
            self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a critical message.

        Args:
            msg: The message to log.
            *args: Variable length argument list for the logger.
            **kwargs: Arbitrary keyword arguments for the logger.
        """
        if self._should_log:
            self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log an exception message.
        
        This logs a message with level ERROR and adds exception information 
        to the logging message.

        Args:
            msg: The message to log.
            *args: Variable length argument list for the logger.
            **kwargs: Arbitrary keyword arguments for the logger.
        """
        if self._should_log:
            self._logger.exception(msg, *args, **kwargs)
    
    def separate(self, char: str = "-", length: int = 60) -> None:
        """Log a separator line.

        Args:
            char: The character to use for the separator.
            length: The length of the separator line.
        """
        if self._should_log:
            self._logger.info(char * length)
    
    def header(self, title: str, char: str = "=", length: int = 60) -> None:
        """Log a header.

        Args:
            title: The title text to display in the header.
            char: The character to use for decoration.
            length: The total length of the header line.
        """
        if self._should_log:
            padding = (length - len(title) - 2) // 2
            header_line = f"{char * padding} {title} {char * padding}"
            self.info(header_line)
    
    def elapsed_time(self) -> float:
        """Get elapsed time since logger initialization.

        Returns:
            The elapsed time in seconds.
        """
        return time.time() - self._start_time
    
    def elapsed_time_str(self) -> str:
        """Return formatted elapsed time string.

        Returns:
            Elapsed time formatted as "HH:MM:SS".
        """
        elapsed = self.elapsed_time()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    @contextmanager
    def timer(self, operation_name: str):
        """Context manager to time a block of code.

        Args:
            operation_name: Name of the operation being timed.

        Yields:
            None
        """
        start_time = time.time()
        self.info(f"Starting operation '{operation_name}'...")
        try:
            yield
        finally:
            end_time = time.time()
            elapsed = end_time - start_time
            self.info(f"Operation '{operation_name}' took {elapsed:.2f} seconds.")


@dataclass
class MetricBuffer:
    """Buffer to store and compute metrics."""
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, key: str, value: float) -> None:
        """Update the buffer with a new metric value.

        Args:
            key: The metric name.
            value: The metric value (e.g., loss, accuracy).
        """
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def get_mean(self, key: str) -> Optional[float]:
        """Get the mean of the stored values for a given key.

        Args:
            key: The metric name.

        Returns:
            The mean value of the metric, or None if no values exist.
        """
        if key in self.metrics and self.metrics[key]:
            return sum(self.metrics[key]) / len(self.metrics[key])
        return None

    def reset(self) -> None:
        """Reset all stored metrics.
        
        Clears the internal dictionary of logged values.
        """
        self.metrics.clear()
    
    def get_all_means(self) -> Dict[str, float]:
        """Get means for all metrics.

        Returns:
            A dictionary mapping metric names to their mean values.
        """
        return {name: self.get_mean(name) for name in self.metrics.keys()}#type: ignore

class ArchitectureLogger(Logger):
    """
    Logger specialized for model architecture analysis.
    """
    
    def log_model_summary(self, model: Any, input_data: Any = None, depth: int = 4) -> None:
        """
        Log detailed model summary including layer shapes and parameter counts.
        
        Args:
            model: PyTorch model
            input_data: Optional input tensor (or tuple/list of tensors) to infer shapes.
                        If None, only static analysis is performed.
            depth: Maximum depth of nested layers to display.
        """
        if not self._should_log:
            return

        self.separate("=")
        self.header("Model Architecture Summary")
        self.separate("-")

        try:
            import torch
            import torch.nn as nn
            from collections import OrderedDict
        except ImportError:
            self.warning("PyTorch not installed. Cannot analyze model architecture.")
            return

        layer_stats = OrderedDict()
        hooks = []

        def register_hook(module, name):
            def hook(mod, inp, out):
                class_name = str(mod.__class__).split(".")[-1].split("'")[0]
                module_idx = len(layer_stats)
                
                # Use name if unique, else append index
                m_key = name if name not in layer_stats else f"{name}-{module_idx}"
                
                stats = {
                    "class_name": class_name,
                    "input_shape": tuple(inp[0].shape) if inp else None,
                    "output_shape": tuple(out.shape) if isinstance(out, torch.Tensor) else None,
                    "params": sum(p.numel() for p in mod.parameters(recurse=False)),
                    "trainable": any(p.requires_grad for p in mod.parameters(recurse=False))
                }
                layer_stats[m_key] = stats

            # Only hook leaf layers or specific depth
            if not isinstance(module, (nn.Sequential, nn.ModuleList)) and \
               not (module == model) and \
               (input_data is not None): 
                hooks.append(module.register_forward_hook(hook))

        if input_data is not None:
            # Dynamic analysis
            for name, module in model.named_modules():
                if name == "": continue
                if name.count('.') >= depth: continue
                register_hook(module, name)
            
            # Forward pass
            model_mode = model.training
            model.eval()
            try:
                if isinstance(input_data, (tuple, list)):
                    if all(isinstance(x, torch.Tensor) for x in input_data):
                         model(*input_data)
                    else:
                         model(input_data)
                else:
                    # Provide batch dim if missing
                    # Assuming input_data is a single sample if dims match expected - 1, but safe to just pass
                    model(input_data.unsqueeze(0) if input_data.dim() == len(input_data.shape)-1 else input_data)
                    
            except Exception as e:
                self.warning(f"Failed to run forward pass for shape inference: {e}")
            finally:
                model.train(model_mode)
                for h in hooks: h.remove()
        
        else:
            # Static analysis
            for name, module in model.named_modules():
                if name == "": continue
                if name.count('.') >= depth: continue
                
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                stats = {
                    "class_name": class_name,
                    "input_shape": "-",
                    "output_shape": "-",
                    "params": sum(p.numel() for p in module.parameters(recurse=False)),
                    "trainable": any(p.requires_grad for p in module.parameters(recurse=False))
                }
                layer_stats[name] = stats

        # Print table
        row_format = "{:<40} | {:<20} | {:<20} | {:<12} | {:<6}"
        self.info(row_format.format("Layer (type)", "Input Shape", "Output Shape", "Param #", "Train?"))
        self.separate("-")
        
        # Calculate totals globally to avoid depth-dependent summation issues
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        for name, stats in layer_stats.items():
            display_name = f"{name} ({stats['class_name']})"
            
            inp_shape = str(stats['input_shape']).replace('torch.Size', '') if stats['input_shape'] else "-"
            out_shape = str(stats['output_shape']).replace('torch.Size', '') if stats['output_shape'] else "-"
            
            params = stats['params']
            
            tr_flag = "True" if stats['trainable'] else "False"
            
            if len(display_name) > 38:
                display_name = display_name[:35] + "..."
                
            self.info(row_format.format(display_name, inp_shape, out_shape, f"{params:,}", tr_flag))

        self.separate("-")
        self.info(f"Total Params: {total_params:,}")
        self.info(f"Trainable Params: {total_trainable:,}")
        self.info(f"Non-Trainable Params: {total_params - total_trainable:,}")
        self.separate("=")

    def log_model_graph(self, model: Any, input_data: Any, filename: str = "model_graph") -> None:
        """Generate and save a visual graph of the model using torchviz.

        Args:
            model: The PyTorch model to visualize.
            input_data: Sample input data for the model to trace the graph.
            filename: The name of the file (without extension) to save the graph to.
        """
        if not self._should_log:
            return
        
        try:
            import torchviz
            # Run forward pass to build graph
            out = model(input_data)
            dot = torchviz.make_dot(out, params=dict(model.named_parameters()))
            
            if self.log_dir:
                save_path = self.log_dir / filename
                dot.render(str(save_path), format="png", cleanup=True)
                self.info(f"Model graph saved to {save_path}.png")
            else:
                self.warning("Log directory not set. Cannot save model graph.")
                
        except ImportError:
            self.warning("torchviz not installed. Run `pip install torchviz` to generate graphs.")
        except Exception as e:
            self.warning(f"Failed to generate model graph: {e}")

class TrainingLogger(ArchitectureLogger):
    """
    Logger specialized for training processes.

    Extends BaseLogger with features specifically for training:
    - Metrics tracking (loss, accuracy, etc.)
    - Epoch and step logging
    - Learning rate logging
    - GPU memory tracking
    - Progress bars
    - Checkpointing logs
    """

    def __init__(self, name: Optional[str] = "TrainingLogger", world_size: int = 1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.world_size = world_size
        self.metric_buffer = MetricBuffer()
        self.history: Dict[str, List[float]] = {}

        # Progress tracking
        self._current_epoch = 0
        self._current_step = 0
        self._last_log_time = time.time()
        self._log_interval = 5.0  # seconds
    
    def log_config(self, config: Dict[str, Any], title: str = "Configuration") -> None:
        """Log training configuration/hyperparameters.

        Args:
            config: A dictionary or object containing configuration parameters.
            title: The title for the configuration section in the internal log.
        """
        if not self._should_log:
            return
        if hasattr(config, '__dataclass_fields__'):
            config_dict = {k: getattr(config, k) for k in config.__dataclass_fields__.keys()} #type: ignore
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = vars(config)
        self.separate("=")

        if self.log_dir:
            config_file = self.log_dir / f"{self.name}_config.json"
            
            def make_serializable(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {str(k): make_serializable(v) for k, v in obj.items()}
                elif hasattr(obj, '__dataclass_fields__'):
                    return {k: make_serializable(getattr(obj, k)) for k in obj.__dataclass_fields__.keys()}
                elif hasattr(obj, '__dict__'):
                    return {k: make_serializable(v) for k, v in vars(obj).items()}
                else:
                    return str(obj)

            with open(config_file, 'w', encoding='utf-8') as f:
                serializable = make_serializable(config_dict)
                json.dump(serializable, f, indent=4)
    

    def log_model_info(
            self, 
            model: Any, 
            input_size: Optional[Any] = None, 
            title: str = "Model Information"
        ) -> None:
        """
        Log model architecture and parameter count.
        Delegates to log_model_summary for detailed info if inputs are provided.
        """
        if not self._should_log:
            return
        
        if input_size is not None:
             # Detailed dynamic analysis
             self.log_model_summary(model, input_data=input_size)
        else:
             # Basic print + detailed static analysis
             self.separate("=")
             self.header(f"{title}:")
             self.separate("-")
             
             self.info(str(model))
             self.separate("-")
             
             # Static analysis
             self.log_model_summary(model, input_data=None, depth=3)
             
             self.info(f"World size (number of processes): {self.world_size}")
             self.separate("-")
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log the start of an epoch.

        Args:
            epoch: The current epoch number (1-indexed).
            total_epochs: The total number of epochs.
        """
        if not self._should_log:
            return
        self._current_epoch = epoch
        self.header(f"Starting Epoch {epoch}/{total_epochs}")
        self._current_step = 0
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float], save_checkpoint: bool = True) -> None:
        """Log the end of an epoch.

        Args:
            epoch: The current epoch number.
            metrics: A dictionary of metrics collected during the epoch.
            save_checkpoint: Whether a checkpoint is being saved this epoch.
        """
        if not self._should_log:
            return
        self.header(f"Finished Epoch {epoch}")
        metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        elapsed = self.elapsed_time_str()
        self.info(f"Epoch {epoch} Metrics: {metrics_str} | Elapsed Time: {elapsed}")

        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        if self.log_dir:
            history_file = self.log_dir / f"{self.name}_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=4)
        if save_checkpoint:
            self.info(f"Checkpoint for epoch {epoch} can be saved now.")
        self.separate("=")
    
    def log_step(self, step: int, total_steps: int, metrics: Dict[str, float], log_interval: int = 100) -> None:
        """Log training progress for a step.

        Aggregates per-step metrics into an internal metric buffer and
        emits a formatted progress update periodically based on
        ``log_interval``.

        Args:
            step: Current training step (0-indexed).
            total_steps: Total steps in the current epoch.
            metrics: Per-step metrics (e.g., loss, accuracy).
            log_interval: How many steps between printing a progress update.
        """
        self._current_step = step

        for key, value in metrics.items():
            self.metric_buffer.update(key, value)
        
        if step % log_interval != 0:
            return
        if not self._should_log:
            return
        
        progress = ((step + 1) / total_steps) * 100
        avg_metrics = self.metric_buffer.get_all_means()
        metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in avg_metrics.items()])

        bar_length = 30
        filled_length = int(bar_length * (step + 1) // total_steps)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)

        self.info(f"Epoch {self._current_epoch} | Step {step+1}/{total_steps} [{bar}] {progress:.2f}% | {metrics_str}")

        self.metric_buffer.reset()
    
    def log_validation(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Log validation metrics.

        Args:
            epoch: The current epoch number.
            metrics: A dictionary of validation metrics.
            is_best: Whether this validation run produced the best model so far.
        """
        if not self._should_log:
            return
        self.header(f"Validation Results after Epoch {epoch}")
        metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.info(f"Validation Metrics: {metrics_str}")
        self.separate("-")
    
    def log_learning_rate(self, lr: float) -> None:
        """Log the current learning rate.

        Args:
            lr: The current learning rate value.
        """
        if not self._should_log:
            return
        self.info(f"Current Learning Rate: {lr:.6f}")
    
    def log_gpu_memory(self, gpu_id: int = 0) -> None:
        """Log GPU memory usage.

        Args:
            gpu_id: The ID of the GPU to check (default: 0).
        """
        if not self._should_log:
            return
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
                self.info(f"GPU {gpu_id} Memory - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
            else:
                self.info("CUDA is not available. Cannot log GPU memory.")
        except ImportError:
            self.info("PyTorch is not installed. Cannot log GPU memory.")
    
    def log_checkpoint(self, epoch: int, step: int, filepath: Union[str, Path]) -> None:
        """Log checkpoint saving.

        Args:
            epoch: The current epoch number.
            step: The current step number.
            filepath: The path where the checkpoint was saved.
        """
        if not self._should_log:
            return
        self.info(f"Checkpoint saved at Epoch {epoch}, Step {step}: {filepath}")
    
    def log_early_stopping(self, epoch: int, metric: str, value: float) -> None:
        """Log early stopping event.

        Args:
            epoch: The epoch where early stopping was triggered.
            metric: The metric name that triggered early stopping.
            value: The value of the metric.
        """
        if not self._should_log:
            return
        self.warning(f"Early stopping triggered at Epoch {epoch} based on metric '{metric}' with value {value:.4f}.")
    
    def training_summary(self) -> None:
        """Print training summary at the end.
        
        Displays total time, total epochs, and best metrics recorded.
        """
        if not self._should_log:
            return
        
        self.separate("=")
        self.header("Training Complete")
        
        self.info(f"  Total time: {self.elapsed_time_str()}")
        self.info(f"  Epochs completed: {self._current_epoch + 1}")
        
        # Best metrics from history
        if isinstance(self.history, dict) and self.history:
            self.info("  Best metrics:")
            for key, values in self.history.items():
                if not values:
                    continue
                if 'loss' in key.lower():
                    best = min(values)
                    best_epoch = values.index(best) + 1
                else:
                    best = max(values)
                    best_epoch = values.index(best) + 1
                self.info(f"    {key}: {best:.4f} (epoch {best_epoch})")
        
        self.separate("=")

class MetricsLogger(TrainingLogger):
    """
    Logger with external logging integration (W&B, TensorBoard).
    
    Extends TrainingLogger with support for:
    - Weights & Biases (wandb) integration
    - TensorBoard integration
    - Automatic metric synchronization across GPUs
    
    Example:
        ```python
        logger = MetricsLogger(
            name="GPT2med",
            log_dir="./logs",
            use_wandb=True,
            wandb_project="gpt2-training",
            use_tensorboard=True
        )
        
        logger.log_metrics({"loss": 2.5, "accuracy": 0.85}, step=100)
        ```
    """
    
    def __init__(
        self,
        name: str = "Metrics",
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        use_tensorboard: bool = False,
        **kwargs
    ):
        super().__init__(name=name, log_dir=log_dir, **kwargs)
        
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        self._wandb_run = None
        self._tb_writer = None
        
        # Initialize external loggers
        if self._should_log:
            if use_wandb and wandb_project and wandb_entity:
                self._init_wandb(wandb_project, wandb_entity)
            if use_tensorboard:
                self._init_tensorboard()
    
    def _init_wandb(self, project: str, entity: Optional[str] = None) -> None:
        """Initialize Weights & Biases.

        Args:
            project: The W&B project name.
            entity: The W&B entity (team/user) name.
        """
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=project or self.name,
                entity=entity,
                name=f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.info("W&B initialized successfully")
        except ImportError:
            self.warning("wandb not installed. Run: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            self.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard writer.

        Sets up the SummaryWriter in the designated log directory.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.log_dir / "tensorboard" if self.log_dir else Path("./tensorboard")
            tb_dir.mkdir(parents=True, exist_ok=True)
            
            log_dir_str = str(tb_dir)
            # Fix for Windows paths with non-ASCII characters (e.g. user names with accents)
            try:
                self._tb_writer = SummaryWriter(log_dir=log_dir_str)
                self.info(f"TensorBoard initialized: {tb_dir}")
            except Exception as e:
                self.warning(f"Failed to initialize TensorBoard (likely due to path issues): {e}")
                self.use_tensorboard = False
        except ImportError:
            self.warning("TensorBoard not available. Run: pip install tensorboard")
            self.use_tensorboard = False
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ) -> None:
        """
        Log metrics to all enabled backends.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step number
            prefix: Optional prefix for metric names (e.g., "train/", "val/")
        """
        if not self._should_log:
            return
        
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # Log to W&B
        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.log(metrics, step=step)
        
        # Log to TensorBoard
        if self.use_tensorboard and self._tb_writer:
            for name, value in metrics.items():
                self._tb_writer.add_scalar(name, value, step)
    
    def log_histogram(
        self,
        name: str,
        values: Any,
        step: int
    ) -> None:
        """Log histogram of values (for weight distributions, etc.).

        Args:
            name: The name of the histogram metric.
            values: The values to bin into a histogram (e.g., tensor, array).
            step: The current step number.
        """
        if not self._should_log:
            return
        
        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.log({name: wandb.Histogram(values)}, step=step)
        
        if self.use_tensorboard and self._tb_writer:
            self._tb_writer.add_histogram(name, values, step)
    
    def finish(self) -> None:
        """Cleanup and close external loggers.
        
        Ends W&B runs and closes TensorBoard writers if they are active.
        """
        if self._should_log:
            if self._wandb_run:
                import wandb
                wandb.finish()
            
            if self._tb_writer:
                self._tb_writer.close()
            
            self.training_summary()


# ==============================================================================
# Module-level convenience functions
# ==============================================================================

_default_logger: Optional[TrainingLogger] = None


def get_logger(name: str = "default", **kwargs) -> TrainingLogger:
    """Get or create a logger instance.

    If a logger with the given name already exists, it is returned.
    Otherwise, a new TrainingLogger is created.

    Args:
        name: The name of the logger.
        **kwargs: Arguments to pass to the Logger constructor.

    Returns:
        The logger instance.
    """
    global _default_logger
    if _default_logger is None or _default_logger.name != name:
        _default_logger = TrainingLogger(name=name, **kwargs)
    return _default_logger


def set_default_logger(logger: TrainingLogger) -> None:
    """Set the default logger instance.

    Args:
        logger: The logger to set as default.
    """
    global _default_logger
    _default_logger = logger