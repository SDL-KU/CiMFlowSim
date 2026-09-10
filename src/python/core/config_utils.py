"""
Configuration Utilities for CiMFlowSim

Common utility functions for loading and processing configuration files.
"""

from typing import Any, Dict, Tuple


def extract_hardware_constraints(
    hardware_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract hardware architecture and port configuration from hardware config.

    Supports both unified and legacy hardware configuration formats:
    - Unified format: hardware_config["hardware"]["ibuf"]["architecture"]
    - Legacy format: hardware_config["buffer_architecture"]

    Args:
        hardware_config: Hardware configuration dictionary

    Returns:
        Tuple of (architecture_config, port_config) dictionaries

    Raises:
        KeyError: If required configuration fields are missing

    Example:
        >>> hw_config = load_hardware_config("baseline.json")
        >>> hw_arch, hw_ports = extract_hardware_constraints(hw_config)
        >>> print(hw_arch["num_banks"])
        4
    """
    if "hardware" in hardware_config:
        # Unified format (current standard)
        try:
            hw_arch = hardware_config["hardware"]["ibuf"]["architecture"]
            hw_ports = hardware_config["hardware"]["ibuf"]["ports"]
        except KeyError as e:
            raise KeyError(
                f"Invalid unified hardware config format. Missing key: {e}. "
                "Expected: hardware_config['hardware']['ibuf']['architecture'] and ['ports']"
            ) from e
    else:
        # Legacy format (for backward compatibility)
        hw_arch = hardware_config.get("buffer_architecture", {})
        hw_ports = hardware_config.get("port_configuration", {})

        if not hw_arch:
            raise KeyError(
                "Hardware config missing 'buffer_architecture'. "
                "Use unified format: hardware_config['hardware']['ibuf']['architecture']"
            )

    return hw_arch, hw_ports
