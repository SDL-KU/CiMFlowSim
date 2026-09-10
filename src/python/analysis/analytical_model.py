#!/usr/bin/env python3
"""
Analytical Model for CNN Accelerator Performance Estimation

Provides fast approximate calculations for:
- Latency (based on pipeline bottleneck analysis)
- Energy (operation counts × energy parameters)
- Area (buffer sizes × bit area)

This model can estimate metrics without running SystemC simulation,
useful for rapid design space exploration.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from core.logging_config import get_logger
from core.tiling import CNNLayerParams

logger = get_logger(__name__)

# Type alias for backward compatibility
LayerConfig = CNNLayerParams


@dataclass
class TilingConfig:
    """Tiling configuration"""
    output_tile_p: int
    output_tile_q: int
    input_tile_h: int
    input_tile_w: int
    input_tile_p: int
    input_tile_q: int
    num_output_tiles_p: int
    num_output_tiles_q: int
    num_input_tiles_p: int
    num_input_tiles_q: int
    output_tile_count: int
    input_tile_count: int
    case_type: int = 2

    @classmethod
    def from_dict(cls, d: Dict, layer: Optional["LayerConfig"] = None) -> "TilingConfig":
        """Create TilingConfig from dict, calculating missing fields if needed"""
        output_tile_p = d["output_tile_p"]
        output_tile_q = d["output_tile_q"]
        input_tile_p = d["input_tile_p"]
        input_tile_q = d["input_tile_q"]
        num_output_tiles_p = d["num_output_tiles_p"]
        num_output_tiles_q = d["num_output_tiles_q"]
        output_tile_count = d["output_tile_count"]

        # Determine case type first
        if "case_type" in d:
            case_type = d["case_type"]
        else:
            # Case 2: 1 input covers multiple outputs (super-tiling)
            # Case 1: multiple inputs per 1 output (sub-tiling)
            if input_tile_p >= output_tile_p and input_tile_q >= output_tile_q:
                case_type = 2  # Super-tiling
            else:
                case_type = 1  # Sub-tiling

        # Calculate missing fields
        if "num_input_tiles_p" in d:
            num_input_tiles_p = d["num_input_tiles_p"]
            num_input_tiles_q = d["num_input_tiles_q"]
        else:
            if case_type == 2:
                # Case 2 (super-tiling): 1 input tile covers multiple output tiles
                # tiles_per_input = how many output tiles fit in one input tile
                tiles_per_input_p = max(1, input_tile_p // output_tile_p)
                tiles_per_input_q = max(1, input_tile_q // output_tile_q)
                num_input_tiles_p = max(1, num_output_tiles_p // tiles_per_input_p)
                num_input_tiles_q = max(1, num_output_tiles_q // tiles_per_input_q)
            else:
                # Case 1 (sub-tiling): multiple input tiles per output tile
                inputs_per_output_p = max(1, output_tile_p // input_tile_p)
                inputs_per_output_q = max(1, output_tile_q // input_tile_q)
                num_input_tiles_p = num_output_tiles_p * inputs_per_output_p
                num_input_tiles_q = num_output_tiles_q * inputs_per_output_q

        if "input_tile_count" in d:
            input_tile_count = d["input_tile_count"]
        else:
            input_tile_count = num_input_tiles_p * num_input_tiles_q

        return cls(
            output_tile_p=output_tile_p,
            output_tile_q=output_tile_q,
            input_tile_h=d["input_tile_h"],
            input_tile_w=d["input_tile_w"],
            input_tile_p=input_tile_p,
            input_tile_q=input_tile_q,
            num_output_tiles_p=num_output_tiles_p,
            num_output_tiles_q=num_output_tiles_q,
            num_input_tiles_p=num_input_tiles_p,
            num_input_tiles_q=num_input_tiles_q,
            output_tile_count=output_tile_count,
            input_tile_count=input_tile_count,
            case_type=case_type,
        )


@dataclass
class HardwareConfig:
    """Hardware timing and architecture configuration"""
    # CIM compute time
    compute_time_ns: float

    # Memory timing (base_latency_cycles, clk_frequency_mhz)
    ibuf_base_latency_cycles: int
    ibuf_clk_mhz: float
    obuf_base_latency_cycles: int
    obuf_clk_mhz: float
    external_base_latency_cycles: int
    external_clk_mhz: float

    # Memory architecture (bits_per_line, num_banks)
    ibuf_bits_per_line: int
    ibuf_num_banks: int
    obuf_bits_per_line: int
    obuf_num_banks: int
    external_bits_per_line: int
    external_num_banks: int

    # Technology parameters
    sram_bit_area_um2: float

    # CIM macro configuration
    macro_input_size: int
    macro_output_size: int
    macro_area_um2: float

    # Energy parameters (pJ)
    mac_energy_pj: float
    pooling_energy_pj: float
    sram_read_energy_pj_per_bit: float
    sram_write_energy_pj_per_bit: float
    dram_read_energy_pj_per_bit: float
    dram_write_energy_pj_per_bit: float

    @classmethod
    def from_config(cls, config: Dict) -> "HardwareConfig":
        """Load from hardware config JSON"""
        hw = config.get("hardware", config)
        energy = config.get("energy", {})

        return cls(
            compute_time_ns=hw["cim"]["compute_time_ns"],
            # IBUF timing
            ibuf_base_latency_cycles=hw["ibuf"]["timing"]["base_latency_cycles"],
            ibuf_clk_mhz=hw["ibuf"]["timing"]["clk_frequency_mhz"],
            ibuf_bits_per_line=hw["ibuf"]["architecture"]["bits_per_line"],
            ibuf_num_banks=hw["ibuf"]["architecture"]["num_banks"],
            # OBUF timing
            obuf_base_latency_cycles=hw["obuf"]["timing"]["base_latency_cycles"],
            obuf_clk_mhz=hw["obuf"]["timing"]["clk_frequency_mhz"],
            obuf_bits_per_line=hw["obuf"]["architecture"]["bits_per_line"],
            obuf_num_banks=hw["obuf"]["architecture"]["num_banks"],
            # External timing
            external_base_latency_cycles=hw["external"]["timing"]["base_latency_cycles"],
            external_clk_mhz=hw["external"]["timing"]["clk_frequency_mhz"],
            external_bits_per_line=hw["external"]["architecture"]["bits_per_line"],
            external_num_banks=hw["external"]["architecture"]["num_banks"],
            # Technology
            sram_bit_area_um2=hw["technology"]["sram_bit_area_um2"],
            # CIM macro
            macro_input_size=hw["cim"]["macro_config"]["input_size"],
            macro_output_size=hw["cim"]["macro_config"]["output_size"],
            macro_area_um2=hw["cim"]["macro_config"]["macro_area_um2"],
            # Energy
            mac_energy_pj=energy.get("computation", {}).get("mac_energy", 0.45),
            pooling_energy_pj=energy.get("computation", {}).get("pooling_energy", 0.1),
            sram_read_energy_pj_per_bit=energy.get("memory", {}).get("sram_read_energy_per_bit", 0.9),
            sram_write_energy_pj_per_bit=energy.get("memory", {}).get("sram_write_energy_per_bit", 1.2),
            dram_read_energy_pj_per_bit=energy.get("memory", {}).get("dram_read_energy_per_bit", 48.0),
            dram_write_energy_pj_per_bit=energy.get("memory", {}).get("dram_write_energy_per_bit", 64.0),
        )

    def c_ext(self) -> float:
        """Cycle period for external memory (ns)"""
        return 1000.0 / self.external_clk_mhz

    def c_ibuf(self) -> float:
        """Cycle period for IBUF (ns)"""
        return 1000.0 / self.ibuf_clk_mhz

    def c_obuf(self) -> float:
        """Cycle period for OBUF (ns)"""
        return 1000.0 / self.obuf_clk_mhz


@dataclass
class AnalyticalResult:
    """Result of analytical estimation (simplified)"""
    # Latency (t_exec = calibration × max{t_comm, t_comp})
    latency_ns: float
    t_comm_ns: float  # Memory stage (tiling-dependent)
    t_comp_ns: float  # Compute stage (tiling-independent)
    bottleneck: str  # "ext", "ibuf_w", "obuf_r", "ibuf_r", "imc", "obuf_w"

    # Energy (nJ)
    energy_nj: float

    # Area (mm²)
    area_mm2: float

    # Detailed breakdown (optional, for analysis)
    timing_breakdown: Optional[Dict[str, float]] = None
    energy_breakdown: Optional[Dict[str, float]] = None
    area_breakdown: Optional[Dict[str, float]] = None


class AnalyticalModel:
    """
    Analytical model for CNN accelerator performance estimation.

    Model structure:
        t_exec = max{t_comm, t_comp}

        t_comm (Memory stage, tiling-dependent):
            t_ext_sum = n_in × t_ext_r + n_out × t_ext_w
            t_comm = max{t_ext_sum, n_in × t_ibuf_w, n_out × t_obuf_r}

        t_comp (Compute stage, tiling-independent):
            t_comp = max{P×Q×t_ibuf_r, P×Q×t_mac, P'×Q'×t_obuf_w}
    """

    def __init__(self, hw_config: HardwareConfig):
        self.hw = hw_config

    @classmethod
    def from_config_file(cls, config_path: Path) -> "AnalyticalModel":
        """Create model from hardware config file"""
        with open(config_path) as f:
            config = json.load(f)
        return cls(HardwareConfig.from_config(config))

    @classmethod
    def from_config(cls, config: Dict) -> "AnalyticalModel":
        """Create model from hardware config dict"""
        return cls(HardwareConfig.from_config(config))

    # ========== Memory time calculation ==========

    def _calc_mem_time(
        self,
        total_bits: int,
        bits_per_line: int,
        num_banks: int,
        base_latency_cycles: int,
        cycle_ns: float,
    ) -> float:
        """
        Unified memory access time calculation.

        t_mem = (base_latency + ceil(lines / num_banks)) × cycle_ns

        Args:
            total_bits: Total bits to transfer
            bits_per_line: Memory line width (bits)
            num_banks: Number of parallel banks
            base_latency_cycles: Base latency in cycles
            cycle_ns: Nanoseconds per cycle

        Returns:
            Access time in nanoseconds
        """
        if total_bits <= 0 or bits_per_line <= 0:
            return 0.0
        lines = math.ceil(total_bits / bits_per_line)
        lines_per_bank = math.ceil(lines / num_banks)
        return (base_latency_cycles + lines_per_bank) * cycle_ns

    # ========== Main calculation ==========

    def _calculate_latency(
        self, layer: "LayerConfig", tiling: "TilingConfig"
    ) -> Dict[str, float]:
        """
        Calculate latency: t_exec = max{t_comm, t_comp}
        """
        batch = layer.batch_size
        hw = self.hw

        # === Tile counts ===
        n_in = tiling.input_tile_count
        n_out = tiling.output_tile_count

        # === Bit calculations ===
        input_tile_bits = tiling.input_tile_h * tiling.input_tile_w * layer.C * layer.input_bitwidth
        out_p_pooled = tiling.output_tile_p // layer.pool_height
        out_q_pooled = tiling.output_tile_q // layer.pool_width
        output_tile_bits = out_p_pooled * out_q_pooled * layer.M * layer.output_bitwidth
        ibuf_read_bits = layer.R * layer.S * layer.C * layer.input_bitwidth
        obuf_write_bits = layer.M * layer.output_bitwidth

        # === Per-tile times (Memory Stage) ===
        t_ext_r = self._calc_mem_time(input_tile_bits, hw.external_bits_per_line,
                                       hw.external_num_banks, hw.external_base_latency_cycles, hw.c_ext())
        t_ext_w = self._calc_mem_time(output_tile_bits, hw.external_bits_per_line,
                                       hw.external_num_banks, hw.external_base_latency_cycles, hw.c_ext())
        t_ibuf_w = self._calc_mem_time(input_tile_bits, hw.ibuf_bits_per_line,
                                        hw.ibuf_num_banks, hw.ibuf_base_latency_cycles, hw.c_ibuf())
        t_obuf_r = self._calc_mem_time(output_tile_bits, hw.obuf_bits_per_line,
                                        hw.obuf_num_banks, hw.obuf_base_latency_cycles, hw.c_obuf())

        # === Per-pixel times (Compute Stage) ===
        t_ibuf_r = self._calc_mem_time(ibuf_read_bits, hw.ibuf_bits_per_line,
                                        hw.ibuf_num_banks, hw.ibuf_base_latency_cycles, hw.c_ibuf())
        t_obuf_w = self._calc_mem_time(obuf_write_bits, hw.obuf_bits_per_line,
                                        hw.obuf_num_banks, hw.obuf_base_latency_cycles, hw.c_obuf())
        t_mac = hw.compute_time_ns

        # === t_comm: Memory stage (tiling-dependent) ===
        t_ext_total = batch * (n_in * t_ext_r + n_out * t_ext_w)
        t_ibuf_w_total = batch * n_in * t_ibuf_w
        t_obuf_r_total = batch * n_out * t_obuf_r
        t_comm = max(t_ext_total, t_ibuf_w_total, t_obuf_r_total)

        # === t_comp: Compute stage (tiling-independent) ===
        t_ibuf_r_total = batch * layer.P * layer.Q * t_ibuf_r
        t_imc_total = batch * layer.P * layer.Q * t_mac
        t_obuf_w_total = batch * layer.P_pooled * layer.Q_pooled * t_obuf_w
        t_comp = max(t_ibuf_r_total, t_imc_total, t_obuf_w_total)

        # === Total execution time ===
        t_exec = max(t_comm, t_comp)

        # === Determine bottleneck ===
        if t_comm >= t_comp:
            if t_ext_total >= max(t_ibuf_w_total, t_obuf_r_total):
                bottleneck = "ext"
            elif t_ibuf_w_total >= t_obuf_r_total:
                bottleneck = "ibuf_w"
            else:
                bottleneck = "obuf_r"
        else:
            if t_ibuf_r_total >= max(t_imc_total, t_obuf_w_total):
                bottleneck = "ibuf_r"
            elif t_imc_total >= t_obuf_w_total:
                bottleneck = "imc"
            else:
                bottleneck = "obuf_w"

        return {
            "t_exec": t_exec,
            "t_comm": t_comm,
            "t_comp": t_comp,
            "bottleneck": bottleneck,
            "n_in": n_in,
            "n_out": n_out,
        }

    def estimate(self, layer: LayerConfig, tiling: TilingConfig) -> AnalyticalResult:
        """
        Estimate latency, energy, and area for a given configuration.

        Args:
            layer: CNN layer parameters
            tiling: Tiling configuration

        Returns:
            AnalyticalResult with estimated metrics
        """
        latency_result = self._calculate_latency(layer, tiling)
        energy = self._calculate_energy(layer, tiling)
        area = self._calculate_area(layer, tiling)

        return AnalyticalResult(
            latency_ns=latency_result["t_exec"],
            t_comm_ns=latency_result["t_comm"],
            t_comp_ns=latency_result["t_comp"],
            bottleneck=latency_result["bottleneck"],
            energy_nj=energy["total"],
            area_mm2=area["total"],
            timing_breakdown=latency_result,
            energy_breakdown=energy,
            area_breakdown=area,
        )

    def _calculate_energy(self, layer: LayerConfig, tiling: TilingConfig) -> Dict[str, float]:
        """Calculate energy breakdown in nJ"""
        batch = layer.batch_size
        hw = self.hw

        # === Operation counts ===
        mac_ops = batch * layer.P * layer.Q * layer.M * layer.C * layer.R * layer.S
        pooling_ops = batch * layer.P * layer.Q * layer.M

        # === Bit calculations ===
        input_tile_bits = tiling.input_tile_h * tiling.input_tile_w * layer.C * layer.input_bitwidth
        out_p_pooled = tiling.output_tile_p // layer.pool_height
        out_q_pooled = tiling.output_tile_q // layer.pool_width
        output_tile_bits = out_p_pooled * out_q_pooled * layer.M * layer.output_bitwidth
        ibuf_read_bits_per_px = layer.R * layer.S * layer.C * layer.input_bitwidth
        obuf_write_bits_per_px = layer.M * layer.output_bitwidth

        # === Transfer counts ===
        n_load = tiling.input_tile_count * batch
        n_store = tiling.output_tile_count * batch
        pixels_per_tile = tiling.output_tile_p * tiling.output_tile_q
        pooled_per_tile = out_p_pooled * out_q_pooled
        n_ibuf_read = tiling.output_tile_count * pixels_per_tile * batch
        n_obuf_write = tiling.output_tile_count * pooled_per_tile * batch

        # === Total bits transferred ===
        dram_read_bits = n_load * input_tile_bits
        dram_write_bits = n_store * output_tile_bits
        ibuf_write_bits = n_load * input_tile_bits
        ibuf_read_bits = n_ibuf_read * ibuf_read_bits_per_px
        obuf_write_bits = n_obuf_write * obuf_write_bits_per_px
        obuf_read_bits = n_store * output_tile_bits

        # === Energy (pJ → nJ) ===
        mac_energy = mac_ops * hw.mac_energy_pj / 1000.0
        pooling_energy = pooling_ops * hw.pooling_energy_pj / 1000.0
        sram_read = (ibuf_read_bits + obuf_read_bits) * hw.sram_read_energy_pj_per_bit / 1000.0
        sram_write = (ibuf_write_bits + obuf_write_bits) * hw.sram_write_energy_pj_per_bit / 1000.0
        dram_read = dram_read_bits * hw.dram_read_energy_pj_per_bit / 1000.0
        dram_write = dram_write_bits * hw.dram_write_energy_pj_per_bit / 1000.0

        return {
            "mac": mac_energy,
            "pooling": pooling_energy,
            "sram_read": sram_read,
            "sram_write": sram_write,
            "dram_read": dram_read,
            "dram_write": dram_write,
            "total": mac_energy + pooling_energy + sram_read + sram_write + dram_read + dram_write,
        }

    def _calculate_area(
        self, layer: LayerConfig, tiling: TilingConfig
    ) -> Dict[str, float]:
        """Calculate area breakdown in mm²"""
        # IBUF: input tile × 2 (double buffering)
        ibuf_bits = (
            tiling.input_tile_h * tiling.input_tile_w * layer.C
            * layer.input_bitwidth * 2  # double buffering
        )
        ibuf_area_um2 = ibuf_bits * self.hw.sram_bit_area_um2
        ibuf_area_mm2 = ibuf_area_um2 / 1_000_000

        # OBUF: output tile (pooled) × 2 (double buffering)
        out_p_pooled = tiling.output_tile_p // layer.pool_height
        out_q_pooled = tiling.output_tile_q // layer.pool_width
        obuf_bits = (
            out_p_pooled * out_q_pooled * layer.M
            * layer.output_bitwidth * 2  # double buffering
        )
        obuf_area_um2 = obuf_bits * self.hw.sram_bit_area_um2
        obuf_area_mm2 = obuf_area_um2 / 1_000_000

        # CIM: based on layer weight dimensions
        input_size = layer.C * layer.R * layer.S
        macros_input = math.ceil(input_size / self.hw.macro_input_size)
        macros_output = math.ceil(layer.M / self.hw.macro_output_size)
        num_macros = macros_input * macros_output
        cim_area_um2 = num_macros * self.hw.macro_area_um2
        cim_area_mm2 = cim_area_um2 / 1_000_000

        return {
            "ibuf": ibuf_area_mm2,
            "obuf": obuf_area_mm2,
            "cim": cim_area_mm2,
            "total": ibuf_area_mm2 + obuf_area_mm2 + cim_area_mm2,
        }
