#!/usr/bin/env python3
"""Workflow file and directory constants for CiMFlowSim.

Why this module exists:
- Centralizes file and directory naming used by systemc_runner.py
- Ensures consistent file structure across simulation workflow
"""

from typing import Final

# ============================================================================
# Directory Names (used by systemc_runner.py)
# ============================================================================

DIR_NAME_STRATEGIES: Final[str] = "strategies"
"""Directory name for strategy JSON files."""

DIR_NAME_SIMULATIONS: Final[str] = "simulations"
"""Directory name for simulation results."""

# ============================================================================
# File Names (used by systemc_runner.py)
# ============================================================================

FILE_NAME_NETWORK_CONFIG: Final[str] = "network_config.json"
"""Standard filename for network configuration snapshot."""

FILE_NAME_HARDWARE_CONFIG: Final[str] = "hardware_config.json"
"""Standard filename for hardware configuration snapshot."""

STRATEGY_FILE_PATTERN: Final[str] = "L{layer_idx}_S{strategy_id}_*.json"
"""
Glob pattern for strategy files with descriptive suffixes.
Format: L<layer_idx>_S<strategy_id>_*.json (e.g., L0_S0_out2x2_in5x5.json)
"""

SIMULATION_DIR_FORMAT: Final[str] = "L{layer_idx}_S{strategy_id}"
"""
Naming pattern for simulation result directories.
Format: L<layer_idx>_S<strategy_id> (e.g., L0_S0)
"""

GANTT_DATA_FILENAME: Final[str] = "gantt_data.bin"
"""Standard filename for Gantt chart data (binary format for fast loading)."""

GANTT_DATA_CSV_FILENAME: Final[str] = "gantt_data.csv"
"""Legacy CSV filename for Gantt chart data (slower but human-readable)."""

GANTT_CHART_PDF_FILENAME: Final[str] = "gantt_chart.pdf"
"""Standard filename for generated Gantt chart PDF."""

TENSOR_REGIONS_FILENAME: Final[str] = "tensor_regions.log"
"""Standard filename for tensor memory region logs."""

EXECUTION_TRACE_FILENAME: Final[str] = "execution_trace.log"
"""Standard filename for execution trace logs."""

SIMULATION_LOG_FILENAME: Final[str] = "simulation_log.txt"
"""Standard filename for simulation console output."""

# ============================================================================
# Workflow Phase Names (used by generate.py, simulate.py)
# ============================================================================

GENERATION_PHASE_NAME: Final[str] = "generation"
"""Standard name for strategy generation phase."""

SIMULATION_PHASE_NAME: Final[str] = "simulation"
"""Standard name for SystemC simulation phase."""

# ============================================================================
# Path Configuration (used by simulate.py)
# ============================================================================

LOCAL_CACHE_BASE: Final[str] = "/tmp/efsim_cache"
"""
Local cache directory for each worker node.
Why: Avoids NFS overhead by caching strategy JSON files locally.
Each worker reads from NFS once, then uses local cache for subsequent access.
"""

WORKER_NFS_BASE: Final[str] = "/mnt/workers"
"""
NFS mount point for worker results.
Why: Each worker writes results to /mnt/workers/{hostname}/.
Head node can access all workers via NFS mounts at same path.
"""
