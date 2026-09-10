"""
Energy Calculator for CNN Accelerator Simulation

Converts operation counts (from SystemC simulation) to energy consumption
using configurable energy parameters.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .exceptions import CalculationError, ConfigurationError, build_calculation_context


@dataclass
class EnergyParameters:
    """Energy parameters for different components"""

    # Computation energy (pJ per operation)
    mac_energy_pj: float
    pooling_energy_pj: float

    # Memory access energy (pJ per bit)
    sram_read_energy_pj_per_bit: float
    sram_write_energy_pj_per_bit: float
    dram_read_energy_pj_per_bit: float
    dram_write_energy_pj_per_bit: float

    # Optional computation energy (pJ per operation)
    activation_energy_pj: float = 0.0
    comparison_energy_pj: float = 0.0

    # On-chip communication energy (pJ per byte)
    on_chip_wire_energy_pj_per_byte: float = 0.0

    # Static power (mW)
    static_power_mw: float = 0.0

    @classmethod
    def from_config(cls, config: Dict) -> "EnergyParameters":
        """
        Create EnergyParameters from hardware config JSON

        Args:
            config: Hardware configuration with 'energy' section

        Returns:
            EnergyParameters instance

        Raises:
            ConfigurationError: If required energy fields are missing
        """
        if "energy" not in config:
            raise ConfigurationError(
                "Missing 'energy' section in hardware config",
                suggestions=["Add 'energy' section to hardware config JSON"],
            )

        energy_config = config["energy"]

        # Computation (required: mac_energy, pooling_energy)
        if "computation" not in energy_config:
            raise ConfigurationError(
                "Missing 'computation' section in energy config",
                suggestions=["Add 'computation' section with 'mac_energy' and 'pooling_energy'"],
            )
        computation = energy_config["computation"]
        mac_energy = computation.get("mac_energy")
        if mac_energy is None:
            raise ConfigurationError("Missing 'mac_energy' in energy.computation")
        pooling_energy = computation.get("pooling_energy")
        if pooling_energy is None:
            raise ConfigurationError("Missing 'pooling_energy' in energy.computation")
        activation_energy = computation.get("activation_energy", 0.0)
        comparison_energy = computation.get("comparison_energy", 0.0)

        # Memory (required: all 4 fields)
        if "memory" not in energy_config:
            raise ConfigurationError(
                "Missing 'memory' section in energy config",
                suggestions=["Add 'memory' section with sram/dram read/write energy values"],
            )
        memory = energy_config["memory"]
        required_memory = ["sram_read_energy_per_bit", "sram_write_energy_per_bit",
                          "dram_read_energy_per_bit", "dram_write_energy_per_bit"]
        for field in required_memory:
            if field not in memory:
                raise ConfigurationError(f"Missing '{field}' in energy.memory")

        # Communication (optional, default 0.0)
        communication = energy_config.get("communication", {})
        wire_energy = communication.get("on_chip_wire_energy", 0.0)

        # Static (optional, default 0.0)
        static = energy_config.get("static", {})
        static_power = static.get("static_power_mw", 0.0)

        return cls(
            mac_energy_pj=mac_energy,
            pooling_energy_pj=pooling_energy,
            activation_energy_pj=activation_energy,
            comparison_energy_pj=comparison_energy,
            sram_read_energy_pj_per_bit=memory["sram_read_energy_per_bit"],
            sram_write_energy_pj_per_bit=memory["sram_write_energy_per_bit"],
            dram_read_energy_pj_per_bit=memory["dram_read_energy_per_bit"],
            dram_write_energy_pj_per_bit=memory["dram_write_energy_per_bit"],
            on_chip_wire_energy_pj_per_byte=wire_energy,
            static_power_mw=static_power,
        )

    @classmethod
    def from_config_file(cls, config_path: Path) -> "EnergyParameters":
        """Load energy parameters from JSON file"""
        with open(config_path) as f:
            config = json.load(f)
        return cls.from_config(config)


@dataclass
class EnergyBreakdown:
    """Detailed energy breakdown"""

    # Computation energy (nJ)
    mac_energy_nj: float
    pooling_energy_nj: float
    activation_energy_nj: float
    comparison_energy_nj: float

    # Memory energy (nJ)
    sram_read_energy_nj: float
    sram_write_energy_nj: float
    dram_read_energy_nj: float
    dram_write_energy_nj: float

    # Communication energy (nJ)
    communication_energy_nj: float

    # Static energy (nJ)
    static_energy_nj: float

    @property
    def total_computation_nj(self) -> float:
        """Total computation energy"""
        return (
            self.mac_energy_nj
            + self.pooling_energy_nj
            + self.activation_energy_nj
            + self.comparison_energy_nj
        )

    @property
    def total_memory_nj(self) -> float:
        """Total memory access energy"""
        return (
            self.sram_read_energy_nj
            + self.sram_write_energy_nj
            + self.dram_read_energy_nj
            + self.dram_write_energy_nj
        )

    @property
    def total_dynamic_nj(self) -> float:
        """Total dynamic energy (computation + memory + communication)"""
        return self.total_computation_nj + self.total_memory_nj + self.communication_energy_nj

    @property
    def total_nj(self) -> float:
        """Total energy (dynamic + static)"""
        return self.total_dynamic_nj + self.static_energy_nj

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "computation": {
                "mac_nj": self.mac_energy_nj,
                "pooling_nj": self.pooling_energy_nj,
                "activation_nj": self.activation_energy_nj,
                "comparison_nj": self.comparison_energy_nj,
                "total_nj": self.total_computation_nj,
            },
            "memory": {
                "sram_read_nj": self.sram_read_energy_nj,
                "sram_write_nj": self.sram_write_energy_nj,
                "dram_read_nj": self.dram_read_energy_nj,
                "dram_write_nj": self.dram_write_energy_nj,
                "total_nj": self.total_memory_nj,
            },
            "communication": {"wire_nj": self.communication_energy_nj},
            "static": {"static_nj": self.static_energy_nj},
            "summary": {
                "total_dynamic_nj": self.total_dynamic_nj,
                "total_static_nj": self.static_energy_nj,
                "total_nj": self.total_nj,
            },
        }


class EnergyCalculator:
    """Calculate energy from operation counts"""

    def __init__(self, params: EnergyParameters, hardware_config: Optional[Dict] = None):
        """
        Initialize calculator with energy parameters

        Args:
            params: Energy parameters for computation and memory
            hardware_config: Hardware configuration (for line widths)
        """
        self.params = params
        self.hardware_config = hardware_config or {}

    @classmethod
    def from_config(cls, config: Dict) -> "EnergyCalculator":
        """Create calculator from hardware config"""
        return cls(EnergyParameters.from_config(config), hardware_config=config.get("hardware", {}))

    @classmethod
    def from_config_file(cls, config_path: Path) -> "EnergyCalculator":
        """Create calculator from config file"""
        with open(config_path) as f:
            config = json.load(f)
        return cls(EnergyParameters.from_config(config), hardware_config=config.get("hardware", {}))

    def calculate(self, operation_counts: Dict, timing_ns: Optional[float] = None) -> EnergyBreakdown:
        """
        Calculate energy from operation counts

        Args:
            operation_counts: SystemC simulation output JSON
            timing_ns: Total execution time in nanoseconds (for static energy)

        Returns:
            Detailed energy breakdown

        Raises:
            ValueError: If operation_counts is empty or missing required fields
        """
        if not operation_counts:
            raise CalculationError(
                "operation_counts is empty. "
                "Expected simulation_statistics.json from SystemC simulation.",
                context=build_calculation_context("energy", {"operation_counts": operation_counts}),
                suggestions=[
                    "Check that SystemC simulation completed successfully",
                    "Verify simulation_statistics.json was generated",
                    "Ensure simulation output directory is accessible",
                ],
            )

        # Extract operation counts
        ops = operation_counts.get("operations", {})
        mac_ops = ops.get("mac_ops", 0)
        pooling_ops = ops.get("pooling_ops", 0)
        activation_ops = ops.get("activation_ops", 0)
        comparison_ops = ops.get("comparison_ops", 0)

        # Extract memory line accesses (NEW: line-based for all buffers)
        if "memory_line_accesses" in operation_counts:
            # NEW FORMAT: Line-based access counts from SystemC
            line_accesses = operation_counts["memory_line_accesses"]
            ibuf_read_lines = line_accesses.get("ibuf_read_lines", 0)
            ibuf_write_lines = line_accesses.get("ibuf_write_lines", 0)
            obuf_read_lines = line_accesses.get("obuf_read_lines", 0)
            obuf_write_lines = line_accesses.get("obuf_write_lines", 0)
            external_read_lines = line_accesses.get("external_read_lines", 0)
            external_write_lines = line_accesses.get("external_write_lines", 0)
        else:
            # OLD FORMAT: Fallback to previous method for backward compatibility
            mem = operation_counts.get("memory_accesses", {})
            ibuf_read_lines = mem.get("ibuf_reads", 0)
            ibuf_write_lines = mem.get("ibuf_writes", 0)
            obuf_read_lines = mem.get("obuf_reads", 0)
            obuf_write_lines = mem.get("obuf_writes", 0)
            external_read_lines = mem.get("external_reads", 0)
            external_write_lines = mem.get("external_writes", 0)

        # Get line widths from hardware config
        ibuf_bits_per_line = self.hardware_config.get("ibuf", {}).get("architecture", {}).get("bits_per_line", 256)
        obuf_bits_per_line = self.hardware_config.get("obuf", {}).get("architecture", {}).get("bits_per_line", 128)
        external_bits_per_line = self.hardware_config.get("external", {}).get("architecture", {}).get("bits_per_line", 512)

        # Convert lines to bits
        ibuf_read_bits = ibuf_read_lines * ibuf_bits_per_line
        ibuf_write_bits = ibuf_write_lines * ibuf_bits_per_line
        obuf_read_bits = obuf_read_lines * obuf_bits_per_line
        obuf_write_bits = obuf_write_lines * obuf_bits_per_line
        external_read_bits = external_read_lines * external_bits_per_line
        external_write_bits = external_write_lines * external_bits_per_line

        # Communication energy: all data movement (in bytes)
        total_bytes_moved = ((ibuf_read_bits + ibuf_write_bits +
                            obuf_read_bits + obuf_write_bits +
                            external_read_bits + external_write_bits) // 8)

        # Get timing if available
        if timing_ns is None:
            timing_info = operation_counts.get("timing", {})
            timing_ns = timing_info.get("total_time_ns", 0.0)

        # Calculate computation energy
        mac_energy_nj = (mac_ops * self.params.mac_energy_pj) / 1000.0
        pooling_energy_nj = (pooling_ops * self.params.pooling_energy_pj) / 1000.0
        activation_energy_nj = (activation_ops * self.params.activation_energy_pj) / 1000.0
        comparison_energy_nj = (comparison_ops * self.params.comparison_energy_pj) / 1000.0

        # Calculate memory energy (bit-based, all line-based now)
        # SRAM: IBUF + OBUF (read and write)
        sram_read_bits = ibuf_read_bits + obuf_read_bits
        sram_write_bits = ibuf_write_bits + obuf_write_bits
        sram_read_energy_nj = (sram_read_bits * self.params.sram_read_energy_pj_per_bit) / 1000.0
        sram_write_energy_nj = (sram_write_bits * self.params.sram_write_energy_pj_per_bit) / 1000.0

        # DRAM: External memory (read and write)
        dram_read_energy_nj = (external_read_bits * self.params.dram_read_energy_pj_per_bit) / 1000.0
        dram_write_energy_nj = (external_write_bits * self.params.dram_write_energy_pj_per_bit) / 1000.0

        # Calculate communication energy
        communication_energy_nj = (
            total_bytes_moved * self.params.on_chip_wire_energy_pj_per_byte
        ) / 1000.0

        # Calculate static energy
        # Static power (mW) × time (ns) = energy (pJ), then convert to nJ
        static_energy_nj = (self.params.static_power_mw * timing_ns / 1e6) / 1000.0

        return EnergyBreakdown(
            mac_energy_nj=mac_energy_nj,
            pooling_energy_nj=pooling_energy_nj,
            activation_energy_nj=activation_energy_nj,
            comparison_energy_nj=comparison_energy_nj,
            sram_read_energy_nj=sram_read_energy_nj,
            sram_write_energy_nj=sram_write_energy_nj,
            dram_read_energy_nj=dram_read_energy_nj,
            dram_write_energy_nj=dram_write_energy_nj,
            communication_energy_nj=communication_energy_nj,
            static_energy_nj=static_energy_nj,
        )

    def calculate_from_file(self, operations_json_path: Path) -> EnergyBreakdown:
        """
        Calculate energy from SystemC output JSON file

        Args:
            operations_json_path: Path to energy statistics JSON file

        Returns:
            Detailed energy breakdown
        """
        with open(operations_json_path) as f:
            operation_counts = json.load(f)
        return self.calculate(operation_counts)
