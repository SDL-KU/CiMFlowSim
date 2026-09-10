#!/usr/bin/env python3
"""Area calculator for CNN hardware systems.

Terminology:
    IBUF (Input Buffer): On-chip SRAM buffer storing input activations
        loaded from external DRAM before CIM computation.
    OBUF (Output Buffer): On-chip SRAM buffer storing output activations
        after CIM computation, before writing back to DRAM.
    CIM (Compute-In-Memory): Processing array where MAC (multiply-accumulate)
        operations are performed directly within memory cells, enabling
        high parallelism and energy efficiency for CNN inference.
    MACRO: A CIM compute unit with fixed input_size × output_size dimensions,
        representing the basic building block of the CIM array.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    MISSING_REQUIRED_FIELD,
)
from .exceptions import ConfigurationError


class AreaCalculator:
    """Calculate physical silicon area for CNN hardware systems"""

    def __init__(self, hw_config: Dict[str, Any]) -> None:
        """
        Initialize area calculator with hardware configuration

        Args:
            hw_config: Hardware config with "hardware" wrapper (JSON format)
        """
        # Extract hardware section
        if "hardware" not in hw_config:
            raise ConfigurationError("Config must have 'hardware' section")
        hw_config = hw_config["hardware"]

        self.hw_config: Dict[str, Any] = hw_config

        # Extract technology parameters
        if "technology" not in hw_config:
            raise ConfigurationError(MISSING_REQUIRED_FIELD.format("technology"))
        tech_params = hw_config["technology"]

        # SRAM bit area (includes all peripheral overhead)
        if "sram_bit_area_um2" not in tech_params:
            raise ConfigurationError(MISSING_REQUIRED_FIELD.format("sram_bit_area_um2"))
        self.sram_bit_area_um2: float = tech_params["sram_bit_area_um2"]

        # Extract hardware components (3 buffers + 1 compute array)
        for key in ["ibuf", "obuf", "external", "cim"]:
            if key not in hw_config:
                raise ConfigurationError(MISSING_REQUIRED_FIELD.format(key))

        self.ibuf_config = hw_config["ibuf"]  # Input buffer
        self.obuf_config = hw_config["obuf"]  # Output buffer
        self.external_config = hw_config["external"]  # External DRAM
        self.cim_config = hw_config["cim"]  # CIM compute array

        # Extract macro configuration
        if "macro_config" not in self.cim_config:
            raise ConfigurationError(MISSING_REQUIRED_FIELD.format("macro_config"))

        macro_config = self.cim_config["macro_config"]
        for key in ["input_size", "output_size", "macro_area_um2"]:
            if key not in macro_config:
                raise ConfigurationError(MISSING_REQUIRED_FIELD.format(key))

        self.macro_input_size = macro_config["input_size"]
        self.macro_output_size = macro_config["output_size"]
        self.macro_area_um2 = macro_config["macro_area_um2"]

    def calculate_sram_buffer_area(
        self, buffer_config: Dict[str, Any], num_lines: int, is_double_buffered: bool = True
    ) -> Dict[str, float]:
        """Calculate SRAM buffer area"""
        if not buffer_config or "architecture" not in buffer_config:
            return {"area_um2": 0.0, "area_mm2": 0.0, "bits": 0}

        arch = buffer_config["architecture"]
        if "bits_per_line" not in arch:
            raise ConfigurationError(MISSING_REQUIRED_FIELD.format("bits_per_line"))
        if "num_banks" not in arch:
            raise ConfigurationError(MISSING_REQUIRED_FIELD.format("num_banks"))

        bits_per_line = arch["bits_per_line"]
        num_banks = arch["num_banks"]
        effective_lines = num_lines * 2 if is_double_buffered else num_lines
        total_bits = effective_lines * bits_per_line
        area_um2 = total_bits * self.sram_bit_area_um2
        area_mm2 = area_um2 / 1_000_000

        return {
            "area_um2": area_um2,
            "area_mm2": area_mm2,
            "bits": total_bits,
            "lines": effective_lines,
            "original_lines": num_lines,
            "double_buffered": is_double_buffered,
            "bits_per_line": bits_per_line,
            "num_banks": num_banks,
        }

    def calculate_required_macros_for_layer(self, cnn_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate required MACRO count for a single CNN layer

        Args:
            cnn_params: CNN layer parameters containing C, R, S, M

        Returns:
            Dictionary with MACRO calculation details
        """
        # Extract CNN parameters - all required
        required_cnn_params = ["C", "R", "S", "M"]
        for param in required_cnn_params:
            if param not in cnn_params:
                raise ConfigurationError(f"Missing required CNN parameter '{param}' in cnn_params")

        C = cnn_params["C"]  # Input channels
        R = cnn_params["R"]  # Kernel height
        S = cnn_params["S"]  # Kernel width
        M = cnn_params["M"]  # Output channels

        # Calculate input size and required MACROs
        input_size = C * R * S
        macros_needed_input = math.ceil(input_size / self.macro_input_size)
        macros_needed_output = math.ceil(M / self.macro_output_size)

        # Total MACROs needed for this layer
        total_macros_needed = macros_needed_input * macros_needed_output

        return {
            "input_size": input_size,
            "output_channels": M,
            "macros_needed_input": macros_needed_input,
            "macros_needed_output": macros_needed_output,
            "total_macros_needed": total_macros_needed,
            "macro_utilization_input": input_size / (macros_needed_input * self.macro_input_size),
            "macro_utilization_output": M / (macros_needed_output * self.macro_output_size),
            "layer_params": {"C": C, "R": R, "S": S, "M": M},
        }

    def calculate_cim_array_area(self, cnn_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate CIM (Compute-in-Memory) array area for single layer

        CIM Array Architecture:
        - Each MAC unit computes one weight × input multiplication
        - Array size = (C × R × S) × M for full parallel convolution
        - C×R×S: Total number of weights per output channel
        - M: Number of output channels (parallel MAC units)
        - Each MAC unit includes: SRAM cell + analog multiplier + accumulator

        Args:
            cnn_params: CNN layer parameters containing C, R, S, M

        Returns:
            Dictionary with CIM area calculations
        """
        # Extract CNN parameters - all required
        required_cnn_params = ["C", "R", "S", "M"]
        for param in required_cnn_params:
            if param not in cnn_params:
                raise ConfigurationError(f"Missing required CNN parameter '{param}' in cnn_params")

        C = cnn_params["C"]  # Input channels
        R = cnn_params["R"]  # Kernel height
        S = cnn_params["S"]  # Kernel width
        M = cnn_params["M"]  # Output channels

        # Calculate MACRO requirements for this layer
        macro_info = self.calculate_required_macros_for_layer(cnn_params)

        # Calculate area based on required MACROs
        num_macros = macro_info["total_macros_needed"]
        area_um2 = num_macros * self.macro_area_um2
        area_mm2 = area_um2 / 1_000_000  # Convert to mm²

        # Calculate equivalent MAC units for comparison
        equivalent_mac_units = num_macros * self.macro_input_size * self.macro_output_size

        return {
            "area_um2": area_um2,
            "area_mm2": area_mm2,
            "num_macros": num_macros,
            "equivalent_mac_units": equivalent_mac_units,
            "macro_config": {
                "input_size": self.macro_input_size,
                "output_size": self.macro_output_size,
                "area_per_macro_um2": self.macro_area_um2,
            },
            "macro_utilization": {
                "input": macro_info["macro_utilization_input"],
                "output": macro_info["macro_utilization_output"],
            },
            "array_size": (
                f"{num_macros} MACROs " f"({self.macro_input_size}×{self.macro_output_size} each)"
            ),
            "layer_params": {"C": C, "R": R, "S": S, "M": M},
        }

    def calculate_cim_array_area_for_network(
        self, all_layers: List[Dict[str, Any]], layer_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Calculate CIM array area based on maximum requirements across selected layers

        Args:
            all_layers: List of CNN layer parameters
            layer_range: Tuple (start, end) for layer selection (inclusive). If None, use all layers.
                        Examples: (0, 1) for layers 1-2, (1, 2) for layers 2-3

        Returns:
            Dictionary with CIM area calculations based on max requirements
        """
        if not all_layers:
            return {"area_um2": 0.0, "area_mm2": 0.0, "num_mac_units": 0}

        # Select layers based on range
        if layer_range is not None:
            start_idx, end_idx = layer_range
            selected_layers = all_layers[start_idx : end_idx + 1]
            range_info = f"layers {start_idx + 1}-{end_idx + 1}"
        else:
            selected_layers = all_layers
            range_info = f"all {len(all_layers)} layers"

        # Calculate MACRO requirements for each layer and find maximum
        max_macros_needed = 0
        layer_info = []

        for i, layer in enumerate(selected_layers):
            # Calculate MACRO requirements for this layer
            macro_info = self.calculate_required_macros_for_layer(layer)
            layer_macros = macro_info["total_macros_needed"]

            # Track maximum MACRO count needed
            max_macros_needed = max(max_macros_needed, layer_macros)

            # Store detailed layer information
            layer_info.append(
                {
                    "layer_idx": start_idx + i + 1 if layer_range else i + 1,
                    "input_size": macro_info["input_size"],
                    "output_channels": macro_info["output_channels"],
                    "macros_needed_input": macro_info["macros_needed_input"],
                    "macros_needed_output": macro_info["macros_needed_output"],
                    "total_macros_needed": layer_macros,
                    "macro_utilization_input": macro_info["macro_utilization_input"],
                    "macro_utilization_output": macro_info["macro_utilization_output"],
                    "layer_params": macro_info["layer_params"],
                }
            )

        # Calculate total area based on maximum MACRO count
        area_um2 = max_macros_needed * self.macro_area_um2
        area_mm2 = area_um2 / 1_000_000

        # Calculate equivalent MAC units for comparison
        equivalent_mac_units = max_macros_needed * self.macro_input_size * self.macro_output_size

        return {
            "area_um2": area_um2,
            "area_mm2": area_mm2,
            "num_macros": max_macros_needed,
            "equivalent_mac_units": equivalent_mac_units,
            "macro_config": {
                "input_size": self.macro_input_size,
                "output_size": self.macro_output_size,
                "area_per_macro_um2": self.macro_area_um2,
            },
            "array_size": (
                f"{max_macros_needed} MACROs "
                f"({self.macro_input_size}×{self.macro_output_size} each)"
            ),
            "layer_details": layer_info,
            "selected_range": range_info,
            "num_selected_layers": len(selected_layers),
        }

    def calculate_total_system_area(
        self, buffer_usage: Dict[str, int], cnn_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate total system area including all on-chip components (excluding external memory)

        Args:
            buffer_usage: Dictionary with buffer line requirements (ibuf_lines, obuf_lines)
            cnn_params: CNN layer parameters for CIM calculation

        Returns:
            Comprehensive area breakdown
        """
        # Validate buffer usage data - keys must exist even if values are 0
        required_buffer_keys = ["ibuf_lines", "obuf_lines"]
        for key in required_buffer_keys:
            if key not in buffer_usage:
                raise ConfigurationError(
                    f"Missing required key '{key}' in buffer usage data. "
                    f"AreaCalculator result must contain all required keys."
                )

        # Calculate individual component areas with double buffering for IBUF/OBUF
        ibuf_area = self.calculate_sram_buffer_area(
            self.ibuf_config, buffer_usage["ibuf_lines"], is_double_buffered=True
        )
        obuf_area = self.calculate_sram_buffer_area(
            self.obuf_config, buffer_usage["obuf_lines"], is_double_buffered=True
        )
        cim_area = self.calculate_cim_array_area(cnn_params)

        # Calculate totals (external memory not included in chip area)
        total_sram_area_um2 = ibuf_area["area_um2"] + obuf_area["area_um2"]
        total_system_area_um2 = total_sram_area_um2 + cim_area["area_um2"]

        total_sram_area_mm2 = total_sram_area_um2 / 1_000_000
        total_system_area_mm2 = total_system_area_um2 / 1_000_000

        return {
            # Individual components
            "ibuf": ibuf_area,
            "obuf": obuf_area,
            "cim": cim_area,
            # Totals
            "total_sram_area_um2": total_sram_area_um2,
            "total_sram_area_mm2": total_sram_area_mm2,
            "total_system_area_um2": total_system_area_um2,
            "total_system_area_mm2": total_system_area_mm2,
            # Area breakdown percentages
            "ibuf_percentage": (
                (ibuf_area["area_um2"] / total_system_area_um2) * 100
                if total_system_area_um2 > 0
                else 0
            ),
            "obuf_percentage": (
                (obuf_area["area_um2"] / total_system_area_um2) * 100
                if total_system_area_um2 > 0
                else 0
            ),
            "cim_percentage": (
                (cim_area["area_um2"] / total_system_area_um2) * 100
                if total_system_area_um2 > 0
                else 0
            ),
            "technology_info": {
                "sram_bit_area_um2": self.sram_bit_area_um2,
                "macro_config": {
                    "input_size": self.macro_input_size,
                    "output_size": self.macro_output_size,
                    "area_per_macro_um2": self.macro_area_um2,
                    "mac_units_per_macro": self.macro_input_size * self.macro_output_size,
                },
            },
        }

    def calculate_area(
        self, required_lines: Dict[str, int], cnn_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate total system area including buffers and CIM array

        Args:
            required_lines: Dictionary with 'ibuf_lines' and 'obuf_lines'
            cnn_params: Optional CNN layer parameters for CIM area calculation.
                       If provided, calculates CIM area for the layer.

        Returns:
            Dictionary with complete area breakdown (buffers + CIM if params provided)
        """
        # Validate required lines data - keys must exist even if values are 0
        required_keys = ["ibuf_lines", "obuf_lines"]
        for key in required_keys:
            if key not in required_lines:
                raise ConfigurationError(
                    f"Missing required key '{key}' in required lines data. "
                    f"AreaCalculator result must contain all required keys."
                )

        ibuf_lines = required_lines["ibuf_lines"]
        obuf_lines = required_lines["obuf_lines"]

        # Apply double buffering for IBUF/OBUF
        ibuf_area = self.calculate_sram_buffer_area(
            self.ibuf_config, ibuf_lines, is_double_buffered=True
        )
        obuf_area = self.calculate_sram_buffer_area(
            self.obuf_config, obuf_lines, is_double_buffered=True
        )

        # Calculate CIM area if CNN params provided
        cim_area_mm2 = 0.0
        if cnn_params is not None:
            cim_area_info = self.calculate_cim_array_area(cnn_params)
            cim_area_mm2 = cim_area_info["area_mm2"]

        # Calculate totals
        buffer_area_um2 = ibuf_area["area_um2"] + obuf_area["area_um2"]
        buffer_area_mm2 = buffer_area_um2 / 1_000_000
        total_area_mm2 = buffer_area_mm2 + cim_area_mm2

        return {
            # Buffer areas (um2)
            "ibuf_area": ibuf_area["area_um2"],
            "obuf_area": obuf_area["area_um2"],
            "buffer_area": buffer_area_um2,
            # Area breakdown (mm2)
            "ibuf_area_mm2": ibuf_area["area_mm2"],
            "obuf_area_mm2": obuf_area["area_mm2"],
            "cim_area_mm2": cim_area_mm2,
            "buffer_area_mm2": buffer_area_mm2,
            "total_area_mm2": total_area_mm2,
            # Buffer bits
            "ibuf_bits": ibuf_area["bits"],
            "obuf_bits": obuf_area["bits"],
            "total_bits": ibuf_area["bits"] + obuf_area["bits"],
        }
