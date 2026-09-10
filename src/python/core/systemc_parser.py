"""
SystemC Output Parser - Parse simulation output from pipeline_sim

Why: Separated from SystemCRunner for:
1. Single Responsibility: Parser only handles output parsing
2. Testability: Can test parsing logic independently
3. Reusability: Parser can be used by other modules

Terminology:
    IBUF (Input Buffer): On-chip SRAM buffer for input activations.
        ibuf_lines: Number of buffer lines (rows) required for inputs.
    OBUF (Output Buffer): On-chip SRAM buffer for output activations.
        obuf_lines: Number of buffer lines (rows) required for outputs.
    CIM (Compute-In-Memory): Processing array for MAC operations.
        cim_computes: Number of CIM compute operations completed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

import orjson

from .logging_config import get_logger

logger = get_logger(__name__)

from .constants import (
    JSON_MARKER_ENERGY_END,
    JSON_MARKER_ENERGY_START,
    PATTERN_IBUF_LINES,
    PATTERN_OBUF_LINES,
    PATTERN_OPS_COMPLETED,
    PATTERN_TOTAL_AREA,
    PATTERN_TOTAL_ENERGY,
    PATTERN_TOTAL_LATENCY,
)

# =============================================================================
# Module-Specific Constants
# =============================================================================

# JSON Markers
JSON_MARKER_BUFFER_START = "=== Buffer Usage JSON ==="
JSON_MARKER_BUFFER_END = "=== End Buffer Usage JSON ==="

# Text Parsing Patterns
PATTERN_SECTION_END = "==="
PATTERN_IBUF_DEPTH = "IBUF buffer depth:"
PATTERN_OBUF_DEPTH = "OBUF buffer depth:"
PATTERN_LINES_KEYWORD = "lines"
PATTERN_TOTAL_KEYWORD = "total"
PATTERN_TOTAL_TIME_NS = "total_time_ns"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ParsedMetrics:
    """Parsed metrics from SystemC output"""

    latency_ns: float = 0
    area_mm2: float = 0
    energy_nj: float = 0
    ibuf_lines: int = 0
    obuf_lines: int = 0
    operations_completed: Dict[str, int] = None
    success: bool = False
    error_message: str = ""

    def __post_init__(self):
        if self.operations_completed is None:
            self.operations_completed = {}


# =============================================================================
# SystemC Output Parser
# =============================================================================


class SystemCOutputParser:
    """Parser for SystemC simulation output"""

    def parse_operations_completed(self, output: str) -> Dict[str, int]:
        """Parse 'Operations completed:' section from SystemC output

        Note: Reads the LAST occurrence of 'Operations completed:' section,
        as log files may contain multiple sections from failed/retried runs.
        """
        ops: Dict[str, int] = {}
        if PATTERN_OPS_COMPLETED not in output:
            return ops

        # Find ALL PATTERN_OPS_COMPLETED sections and parse the LAST one
        lines = output.split("\n")
        current_section_start = -1

        for i, line in enumerate(lines):
            if PATTERN_OPS_COMPLETED in line:
                current_section_start = i

        # If we found at least one section, parse from the last occurrence
        if current_section_start >= 0:
            in_ops_section = False

            for i in range(current_section_start, len(lines)):
                line = lines[i]

                if PATTERN_OPS_COMPLETED in line:
                    in_ops_section = True
                    ops = {}  # Reset to parse this section fresh
                    continue

                if in_ops_section:
                    # Parse lines like "  Loads: 1152/1152"
                    match = re.match(r"\s+(\w[\w\s]+):\s+(\d+)/(\d+)", line)
                    if match:
                        op_name = match.group(1).strip()
                        actual = int(match.group(2))

                        # Normalize operation name
                        key_map = {
                            "Loads": "loads",
                            "IBUF Reads": "ibuf_reads",
                            "CIM Computes": "cim_computes",
                            "OBUF Writes": "obuf_writes",
                            "Stores": "stores",
                        }
                        key = key_map.get(op_name)
                        if key:
                            ops[key] = actual
                    elif line.strip().startswith(PATTERN_SECTION_END):
                        break  # End of section

        return ops

    def parse_json_stats(self, output: str) -> dict:
        """Extract JSON statistics from output including buffer usage"""
        stats = {}

        # Parse Energy Statistics JSON
        if JSON_MARKER_ENERGY_START in output:
            start_idx = output.find(JSON_MARKER_ENERGY_START) + len(JSON_MARKER_ENERGY_START)
            end_idx = output.find(JSON_MARKER_ENERGY_END)
            if end_idx > start_idx:
                json_str = output[start_idx:end_idx].strip()
                try:
                    stats = orjson.loads(json_str)
                except orjson.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse energy statistics JSON: {str(e)}",
                        extra={"json_fragment": json_str[:100]},
                    )

        # Parse Buffer Usage JSON
        if JSON_MARKER_BUFFER_START in output:
            start_idx = output.find(JSON_MARKER_BUFFER_START) + len(JSON_MARKER_BUFFER_START)
            end_idx = output.find(JSON_MARKER_BUFFER_END)
            if end_idx > start_idx:
                json_str = output[start_idx:end_idx].strip()
                try:
                    buffer_stats = orjson.loads(json_str)
                    stats["buffer_usage"] = buffer_stats
                except orjson.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse buffer usage JSON: {str(e)}",
                        extra={"json_fragment": json_str[:100]},
                    )

        return stats

    def extract_json_metrics(self, stats: dict) -> tuple:
        """Extract latency, energy, and buffer usage from JSON stats

        Returns:
            tuple: (latency_ns, energy_nj, ibuf_lines, obuf_lines)
        """
        latency_ns = 0
        energy_nj = 0
        ibuf_lines = 0
        obuf_lines = 0

        if "timing" in stats:
            latency_ns = stats["timing"].get(PATTERN_TOTAL_TIME_NS, 0)

        # Energy is now calculated separately by EnergyCalculator
        # from simulation_statistics.json written by SystemC
        if "energy" in stats and "total_energy_nj" in stats["energy"]:
            energy_nj = stats["energy"]["total_energy_nj"]

        # Extract buffer usage
        if "buffer_usage" in stats:
            ibuf_lines = stats["buffer_usage"].get("ibuf_peak_lines", 0)
            obuf_lines = stats["buffer_usage"].get("obuf_peak_lines", 0)

        return latency_ns, energy_nj, ibuf_lines, obuf_lines

    def parse_text_metrics(self, output: str) -> tuple:
        """Parse metrics from text output

        Returns:
            tuple: (latency_ns, area_mm2, energy_nj, ibuf_lines, obuf_lines)
        """
        latency_ns = 0
        area_mm2 = 0
        energy_nj = 0
        ibuf_lines = 0
        obuf_lines = 0

        for line in output.split("\n"):
            if PATTERN_TOTAL_LATENCY in line:
                latency_ns = float(line.split(":")[1].split("ns")[0].strip())
            elif PATTERN_TOTAL_AREA in line:
                area_mm2 = float(line.split(":")[1].split("mm")[0].strip())
            elif PATTERN_TOTAL_ENERGY in line:
                energy_nj = float(line.split(":")[1].split("nJ")[0].strip())
            elif PATTERN_IBUF_DEPTH in line:
                ibuf_lines = int(line.split(":")[1].split(PATTERN_LINES_KEYWORD)[0].strip())
            elif PATTERN_OBUF_DEPTH in line:
                obuf_lines = int(line.split(":")[1].split(PATTERN_LINES_KEYWORD)[0].strip())
            elif PATTERN_TOTAL_TIME_NS in line:
                try:
                    latency_ns = float(line.split(":")[1].strip().rstrip(","))
                except (ValueError, IndexError):
                    pass  # Ignore parsing errors

        return latency_ns, area_mm2, energy_nj, ibuf_lines, obuf_lines

    def parse_buffer_lines(self, output: str, ibuf_lines: int = 0, obuf_lines: int = 0) -> tuple:
        """Extract buffer lines from tensor registry output

        Args:
            output: Raw simulation output
            ibuf_lines: Current ibuf_lines value (fallback)
            obuf_lines: Current obuf_lines value (fallback)

        Returns:
            tuple: (ibuf_lines, obuf_lines)
        """
        for line in output.split("\n"):
            if (
                PATTERN_IBUF_LINES in line
                and PATTERN_LINES_KEYWORD in line
                and PATTERN_TOTAL_KEYWORD in line
            ):
                try:
                    total_idx = line.find("(")
                    if total_idx > 0:
                        total_part = line[total_idx + 1 :].split(" total")[0]
                        ibuf_lines = int(total_part)
                except (ValueError, IndexError):
                    pass  # Ignore parsing errors
            elif (
                PATTERN_OBUF_LINES in line
                and PATTERN_LINES_KEYWORD in line
                and PATTERN_TOTAL_KEYWORD in line
            ):
                try:
                    total_idx = line.find("(")
                    if total_idx > 0:
                        total_part = line[total_idx + 1 :].split(" total")[0]
                        obuf_lines = int(total_part)
                except (ValueError, IndexError):
                    pass  # Ignore parsing errors

        return ibuf_lines, obuf_lines

    def parse_output(self, output: str) -> ParsedMetrics:
        """Parse SystemC simulation output

        Args:
            output: Combined stdout/stderr from simulation

        Returns:
            ParsedMetrics with all extracted metrics
        """
        try:
            # Parse operations completed
            operations_completed = self.parse_operations_completed(output)

            # Try JSON parsing first
            stats = self.parse_json_stats(output)
            latency_ns, energy_nj, ibuf_lines, obuf_lines = self.extract_json_metrics(stats)
            area_mm2 = 0

            # Fallback to text parsing if JSON incomplete
            if latency_ns == 0:
                latency_ns, area_mm2, energy_nj, fallback_ibuf, fallback_obuf = (
                    self.parse_text_metrics(output)
                )
                # Use fallback buffer values only if JSON didn't provide them
                if ibuf_lines == 0:
                    ibuf_lines = fallback_ibuf
                if obuf_lines == 0:
                    obuf_lines = fallback_obuf

            # Fallback: Extract buffer lines from simulation output
            if ibuf_lines == 0 or obuf_lines == 0:
                ibuf_lines, obuf_lines = self.parse_buffer_lines(output, ibuf_lines, obuf_lines)

            return ParsedMetrics(
                latency_ns=latency_ns,
                area_mm2=area_mm2,
                energy_nj=energy_nj,
                ibuf_lines=ibuf_lines,
                obuf_lines=obuf_lines,
                operations_completed=operations_completed,
                success=True if latency_ns > 0 else False,
            )
        except Exception as e:
            return ParsedMetrics(
                success=False,
                error_message=f"Failed to parse output: {e}",
            )

    def parse_from_json_file(self, stats: dict) -> ParsedMetrics:
        """
        Parse simulation metrics from simulation_statistics.json

        Why: Unified data source - all metrics come from one structured JSON file.
        No stdout parsing needed. This is the primary parsing method going forward.

        Args:
            stats: Dict loaded from simulation_statistics.json

        Returns:
            ParsedMetrics with all metrics extracted from JSON
        """
        try:
            # Extract timing (latency)
            timing = stats.get("timing", {})
            latency_ns = timing.get("total_time_ns", 0)

            # Extract buffer usage
            buffer_usage = stats.get("buffer_usage", {})
            ibuf_lines = buffer_usage.get("ibuf_peak_lines", 0)
            obuf_lines = buffer_usage.get("obuf_peak_lines", 0)

            # Area is calculated by Python from buffer lines, not from JSON
            area_mm2 = 0

            # Energy will be calculated separately by EnergyCalculator
            energy_nj = 0

            # Extract pipeline metrics for validation
            pipeline = stats.get("pipeline", {})
            operations_completed = {
                "loads": pipeline.get("loads", 0),
                "ibuf_reads": pipeline.get("ibuf_reads", 0),
                "cim_computes": pipeline.get("cim_computes", 0),
                "obuf_writes": pipeline.get("obuf_writes", 0),
                "stores": pipeline.get("stores", 0),
            }

            return ParsedMetrics(
                latency_ns=latency_ns,
                area_mm2=area_mm2,
                energy_nj=energy_nj,
                ibuf_lines=ibuf_lines,
                obuf_lines=obuf_lines,
                operations_completed=operations_completed,
                success=True if latency_ns > 0 else False,
            )
        except Exception as e:
            return ParsedMetrics(
                success=False,
                error_message=f"Failed to parse JSON: {e}",
            )
