#!/usr/bin/env python3
"""
Custom exception hierarchy for CiMFlowSim

Provides specialized exceptions with better error context and recovery suggestions.
"""

import os
from typing import Any, Dict, List, Optional


class CiMFlowSimError(Exception):
    """Base exception for all CiMFlowSim errors"""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.context = context or {}
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        msg = super().__str__()

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f" (Context: {context_str})"

        if self.suggestions:
            suggestions_str = "; ".join(self.suggestions)
            msg += f" (Suggestions: {suggestions_str})"

        return msg


class ConfigurationError(CiMFlowSimError):
    """Configuration file or parameter errors"""


class ValidationError(CiMFlowSimError):
    """Input validation errors"""


class CalculationError(CiMFlowSimError):
    """Calculation or computation errors"""


class StrategyError(CiMFlowSimError):
    """Strategy-specific errors"""


class HardwareError(CiMFlowSimError):
    """Hardware configuration errors"""


class FileOperationError(CiMFlowSimError):
    """File I/O operation errors"""


class OptimizationError(CiMFlowSimError):
    """Multi-objective optimization errors"""


# Context builders for common error scenarios


def build_file_error_context(file_path: str, operation: str) -> Dict[str, Any]:
    """Build context for file operation errors"""
    return {
        "file_path": file_path,
        "operation": operation,
        "file_exists": os.path.exists(file_path),
        "directory_exists": os.path.exists(os.path.dirname(file_path)),
    }


def build_calculation_context(calculation_type: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Build context for calculation errors"""
    return {
        "calculation_type": calculation_type,
        "input_keys": list(inputs.keys()),
        "input_types": {k: type(v).__name__ for k, v in inputs.items()},
    }
