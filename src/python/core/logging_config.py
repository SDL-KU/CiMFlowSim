#!/usr/bin/env python3
"""
Comprehensive logging configuration for CiMFlowSim

Provides structured logging with performance tracking, error context,
and configurable output formats for development and production use.
"""

import inspect
import logging
import logging.handlers
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union

import orjson

# Results directory is now a parameter, no fallback constant needed


class StructuredFormatter(logging.Formatter):
    """JSON-based structured logging formatter"""

    def __init__(self, include_extra_fields: bool = True):
        super().__init__()
        self.include_extra_fields = include_extra_fields

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields if enabled
        if self.include_extra_fields:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in (
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                ):
                    extra_fields[key] = value

            if extra_fields:
                log_data["extra"] = extra_fields

        return orjson.dumps(log_data, default=str).decode()


class PerformanceFormatter(logging.Formatter):
    """Performance-focused formatter with timing information"""

    def format(self, record: logging.LogRecord) -> str:
        """Format with performance timing"""
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

        # Extract performance data if available
        duration = getattr(record, "duration", None)
        memory_usage = getattr(record, "memory_mb", None)
        cache_hits = getattr(record, "cache_hits", None)

        performance_info = []
        if duration is not None:
            performance_info.append(f"{duration:.3f}s")
        if memory_usage is not None:
            performance_info.append(f"{memory_usage:.1f}MB")
        if cache_hits is not None:
            performance_info.append(f"{cache_hits}")

        perf_str = f" [{' '.join(performance_info)}]" if performance_info else ""

        return f"{timestamp} | {record.levelname:8} | {record.name:20} | {record.getMessage()}{perf_str}"


class CiMFlowSimLogger:
    """Centralized logging configuration for CiMFlowSim"""

    def __init__(
        self,
        name: str = "CiMFlowSim",
        level: Union[str, int] = logging.INFO,
        log_file: Optional[str] = None,
        structured: bool = False,
        performance_tracking: bool = True,
    ):
        """
        Initialize logger configuration

        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            structured: Use structured JSON logging
            performance_tracking: Enable performance tracking
        """
        self.name = name
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.log_file = log_file
        self.structured = structured
        self.performance_tracking = performance_tracking

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

        # Prevent propagation to parent loggers (avoid duplicate output)
        self.logger.propagate = False

        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup logging handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)

        if self.structured:
            console_formatter = StructuredFormatter()
        else:
            if self.performance_tracking:
                console_formatter = PerformanceFormatter()
            else:
                console_formatter = logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
                )

        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if self.log_file:
            self._setup_file_handler()

    def _setup_file_handler(self) -> None:
        """Setup file logging handler"""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler for log management
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(self.level)

        # Always use structured format for files
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Get configured logger instance"""
        return self.logger


# Performance tracking decorator
def log_performance(logger_name: str = "CiMFlowSim.performance"):
    """Decorator for logging function performance"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name)
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time

                # Log successful execution
                logger.info(
                    f"Function {func.__name__} completed successfully",
                    extra={
                        "function": func.__name__,
                        "duration": duration,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                        "success": True,
                    },
                )

                return result

            except Exception as e:
                duration = time.perf_counter() - start_time

                # Log failed execution
                logger.error(
                    f"Function {func.__name__} failed: {e}",
                    extra={
                        "function": func.__name__,
                        "duration": duration,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "success": False,
                    },
                    exc_info=True,
                )

                raise

        return wrapper

    return decorator


# Context manager for operation logging
class LoggedOperation:
    """Context manager for logging operations with timing"""

    def __init__(self, operation_name: str, logger_name: str = "CiMFlowSim.operations", **context):
        self.operation_name = operation_name
        self.logger = logging.getLogger(logger_name)
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(
            f"Starting operation: {self.operation_name}",
            extra={"operation": self.operation_name, "status": "started", **self.context},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time

        if exc_type is None:
            self.logger.info(
                f"Completed operation: {self.operation_name}",
                extra={
                    "operation": self.operation_name,
                    "status": "completed",
                    "duration": duration,
                    "success": True,
                    **self.context,
                },
            )
        else:
            self.logger.error(
                f"Failed operation: {self.operation_name} - {exc_val}",
                extra={
                    "operation": self.operation_name,
                    "status": "failed",
                    "duration": duration,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                    "success": False,
                    **self.context,
                },
                exc_info=True,
            )


# Pre-configured loggers for different components
def setup_component_loggers(
    base_level: Union[str, int] = logging.INFO,
    results_dir: Path = Path("results"),
) -> Dict[str, logging.Logger]:
    """Setup loggers for different CiMFlowSim components

    Args:
        base_level: Logging level for all component loggers
        results_dir: Directory for log files (default: ./results)

    Returns:
        Dictionary mapping component names to logger instances
    """
    loggers = {}

    # Core calculation components
    components = [
        ("buffer_calculator", "Buffer calculation operations"),
        ("area_calculator", "Area calculation operations"),
        ("energy_calculator", "Energy calculation operations"),
        ("multi_objective_optimizer", "Multi-objective optimization"),
        ("dse_runner", "Design space exploration"),
        ("visualization", "Visualization and plotting"),
        ("validation", "Input validation and verification"),
        ("performance", "Performance monitoring"),
        ("operations", "High-level operations"),
    ]

    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    for component, description in components:
        logger_config = CiMFlowSimLogger(
            name=f"CiMFlowSim.{component}",
            level=base_level,
            log_file=results_dir / f"{component}.log",
            structured=False,  # Use readable format for development
            performance_tracking=True,
        )
        loggers[component] = logger_config.get_logger()

        # Log component initialization at DEBUG level (hidden by default)
        loggers[component].debug(f"Initialized {description} logger")

    return loggers


# Initialize default logger
default_logger = CiMFlowSimLogger(
    name="CiMFlowSim", level=logging.INFO, structured=False, performance_tracking=True
).get_logger()


# Convenience function for getting module-specific loggers
def get_logger(name: Optional[str] = None, level: Union[str, int] = logging.INFO) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (defaults to caller's module name)
        level: Logging level

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    if name is None:
        # Get caller's module name
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_module = frame.f_back.f_globals.get("__name__", "CiMFlowSim")
            name = caller_module
        else:
            name = "CiMFlowSim"

    logger = logging.getLogger(name)
    if not logger.handlers:
        # Configure logger if not already configured
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
