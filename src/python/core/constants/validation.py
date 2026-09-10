#!/usr/bin/env python3
"""Validation constants for CiMFlowSim.

Why this module exists:
- Defines required fields for configuration validation
- Provides error message templates
"""

from typing import Final, List

# =============================================================================
# Required Configuration Fields
# =============================================================================

REQUIRED_CNN_FIELDS: Final[List[str]] = [
    "H",
    "W",
    "C",
    "R",
    "S",
    "M",
    "stride",
    "input_bitwidth",
    "output_bitwidth",
]
"""
Why: Minimum required fields for CNN layer configuration.
Missing any of these prevents accurate simulation.
Used by: generate.py, systemc_runner.py
"""

# =============================================================================
# Error Message Templates
# =============================================================================

MISSING_REQUIRED_FIELD: Final[str] = "Missing required field '{}'"
"""
Why: Used by calculator modules for missing configuration field errors.
Provides consistent error messages across all validators.
Used by: area_calculator.py
"""

ERROR_MSG_PIPELINE_SIM_NOT_FOUND: Final[str] = "pipeline_sim binary not found: {path}"
"""
Why: Used when SystemC simulator binary is missing.
Used by: systemc_runner.py
"""

ERROR_MSG_TIMEOUT: Final[str] = "Simulation timeout after {} seconds"
"""
Why: Used when simulation exceeds timeout limit.
Used by: systemc_runner.py
"""

ERROR_MSG_UNKNOWN: Final[str] = "Unknown error during simulation"
"""
Why: Catch-all error message for unexpected failures.
Used by: systemc_runner.py
"""

ERROR_MSG_INVALID_LOG_LEVEL: Final[str] = "Invalid log level: {}"
"""
Why: Used when invalid logging level is specified.
Used by: systemc_runner.py
"""

ERROR_MSG_INVALID_LOG_POLICY: Final[str] = "Invalid log policy: {}"
"""
Why: Used when invalid logging policy is specified.
Used by: systemc_runner.py
"""
