#!/usr/bin/env python3
"""
Design Space Exploration (DSE) Benchmark for CiMFlowSim

Automated Design Space Exploration: systematically evaluates all hardware
presets across multiple network workloads to identify optimal configurations
for different objectives (latency, energy, EDP, area).

Usage:
    python3 tools/dse_benchmark.py --all
    python3 tools/dse_benchmark.py --all --max-cpus 90
    python3 tools/dse_benchmark.py --presets area_optimized,balanced --networks lenet5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Use common utilities (sets up Python path, signal handlers, etc.)
from common import setup_signal_handlers

# Setup signal handlers for graceful shutdown
setup_signal_handlers()

from core.sweep_database import SweepDatabase
from core.logging_config import get_logger, setup_component_loggers
from core.pattern_analyzer import PatternAnalyzer
from core.strategy_scorer import StrategyScorer
DEFAULT_NUM_SAMPLES = 20_000_000  # Monte Carlo samples for Pareto-front quality estimation

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = get_logger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PresetInfo:
    """Hardware preset metadata"""

    name: str  # "high_performance"
    config_path: Path  # configs/hardware_unified/high_performance.json
    technology_node: Optional[str] = None  # "28nm"
    cim_memory_type: Optional[str] = None  # "SRAM", "ReRAM", etc.
    description: Optional[str] = None
    config_hash: Optional[str] = None


@dataclass
class NetworkInfo:
    """Network configuration metadata"""

    name: str  # "lenet5"
    config_path: Path  # configs/networks/lenet5.json
    num_layers: Optional[int] = None
    total_params: Optional[int] = None
    description: Optional[str] = None
    config_hash: Optional[str] = None


@dataclass
class BenchmarkCombination:
    """Single preset × network combination"""

    preset: PresetInfo
    network: NetworkInfo
    combination_id: Optional[int] = None  # DB primary key

    @property
    def preset_short(self) -> str:
        """Full name for preset (for directory naming)"""
        return self.preset.name

    @property
    def network_short(self) -> str:
        """Full name for network (for directory naming)"""
        return self.network.name

    @property
    def workspace_name(self) -> str:
        """Workspace directory name: {preset}_{network}"""
        return f"{self.preset_short}_{self.network_short}"


@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration"""

    presets: List[Path]  # List of preset config paths (or "all")
    networks: List[Path]  # List of network config paths (or "all")
    max_cpus: Optional[int] = None  # Limit Ray cluster CPU usage (None=use all available)
    generate_gantt: bool = False  # Generate Gantt chart PDFs for timing diagrams
    generate_memory_layout: bool = False  # Generate memory layout visualizations
    tiling_diagrams: bool = False  # Generate tiling strategy diagrams
    hardware_path: Optional[Path] = None  # Custom hardware config path (for sweep)
    output_dir: Optional[Path] = None  # Custom output directory (for sweep)


@dataclass
class BenchmarkResult:
    """Results from a single combination execution"""

    preset_name: str
    network_name: str
    workspace_path: Path
    status: str  # "completed", "failed", "partial"
    num_strategies: Optional[int] = None
    generation_time_sec: Optional[float] = None
    simulation_time_sec: Optional[float] = None
    optimization_time_sec: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Benchmark Orchestrator
# ============================================================================


class BenchmarkOrchestrator:
    """
    Orchestrates benchmark execution across multiple presets and networks

    Key Responsibilities:
    1. Preset/Network discovery from configs/
    2. Workspace creation and management
    3. Parallel execution of combinations
    4. Result aggregation to central database
    5. Progress tracking and error handling
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark orchestrator

        Args:
            config: Benchmark execution configuration
        """
        self.config = config

        # Create benchmark workspace (custom or timestamped)
        if config.output_dir:
            # Use custom output directory (for sweep)
            self.benchmark_workspace = config.output_dir
        else:
            # Default: timestamped workspace
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.benchmark_workspace = Path("workspaces") / f"benchmark_{timestamp}"
        self.benchmark_workspace.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db = SweepDatabase(self.benchmark_workspace / "results.db")

        # Initialize benchmark_config.json
        self._init_benchmark_config()

        logger.info(f"Created benchmark workspace: {self.benchmark_workspace}")

    def _get_hostname(self) -> str:
        """Get system hostname safely"""
        import socket

        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

    def _init_benchmark_config(self):
        """Initialize benchmark_config.json with metadata"""
        config_data = {
            "benchmark_id": self.benchmark_workspace.name,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "presets": [p.name for p in self.config.presets] if self.config.presets else [],
            "networks": [n.name for n in self.config.networks] if self.config.networks else [],
            "execution": {
                "max_cpus": self.config.max_cpus,
            },
            "progress": {
                "total_combinations": 0,
                "completed": 0,
                "failed": 0,
                "running": 0,
                "pending": 0,
            },
            "results": {
                "database_path": "results.db",
            },
            "metadata": {
                "efsim_version": "2.0",
                "systemc_version": "2.3.3",
                "python_version": sys.version.split()[0],
                "host": self._get_hostname(),
            },
            "timing": {
                "started_at": None,
                "completed_at": None,
                "elapsed_seconds": 0,
            },
        }

        config_path = self.benchmark_workspace / "benchmark_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Initialized benchmark config: {config_path}")

    def discover_presets(self) -> List[PresetInfo]:
        """
        Discover hardware preset configurations from configs/hardware/active/
        or use custom hardware_path if specified.

        Returns:
            List of PresetInfo objects with metadata
        """
        # If custom hardware_path is specified, use it directly
        if self.config.hardware_path:
            config_file = self.config.hardware_path
            if not config_file.exists():
                logger.error(f"Hardware config not found: {config_file}")
                return []

            try:
                with open(config_file) as f:
                    config_data = json.load(f)

                preset_name = config_file.stem
                tech_node = config_data.get("hardware", {}).get("technology", {}).get("node")
                cim_type = config_data.get("hardware", {}).get("cim", {}).get("memory_type")
                description = config_data.get("description")

                preset = PresetInfo(
                    name=preset_name,
                    config_path=config_file,
                    technology_node=tech_node,
                    cim_memory_type=cim_type,
                    description=description,
                )
                logger.info(f"Using custom hardware config: {preset_name}")
                return [preset]

            except Exception as e:
                logger.error(f"Failed to parse hardware config {config_file}: {e}")
                return []

        # Default: discover from configs/hardware/active/
        presets_dir = Path("configs/hardware/active")
        if not presets_dir.exists():
            logger.warning(f"Presets directory not found: {presets_dir}")
            return []

        presets = []
        for config_file in sorted(presets_dir.glob("*.json")):
            try:
                with open(config_file) as f:
                    config_data = json.load(f)

                # Extract metadata
                preset_name = config_file.stem
                tech_node = config_data.get("hardware", {}).get("technology", {}).get("node")
                cim_type = config_data.get("hardware", {}).get("cim", {}).get("memory_type")
                description = config_data.get("description")

                preset = PresetInfo(
                    name=preset_name,
                    config_path=config_file,
                    technology_node=tech_node,
                    cim_memory_type=cim_type,
                    description=description,
                )
                presets.append(preset)
                logger.debug(f"Discovered preset: {preset_name}")

            except Exception as e:
                logger.warning(f"Failed to parse preset {config_file}: {e}")
                continue

        logger.info(f"Discovered {len(presets)} hardware presets")
        return presets

    def discover_networks(self, filter_names: List[str] = None) -> List[NetworkInfo]:
        """
        Discover network configurations from configs/networks/active/

        Args:
            filter_names: Optional list of network names to include (e.g., ['resnet18', 'vgg11'])

        Returns:
            List of NetworkInfo objects with metadata
        """
        networks_dir = Path("configs/networks/active")
        if not networks_dir.exists():
            logger.warning(f"Networks directory not found: {networks_dir}")
            return []

        networks = []
        for config_file in sorted(networks_dir.glob("*.json")):
            # Apply filter if specified
            if filter_names and config_file.stem not in filter_names:
                continue
            try:
                with open(config_file) as f:
                    config_data = json.load(f)

                # Extract metadata
                network_name = config_file.stem
                num_layers = len(config_data.get("layers", []))
                description = config_data.get("description")

                network = NetworkInfo(
                    name=network_name,
                    config_path=config_file,
                    num_layers=num_layers,
                    description=description,
                )
                networks.append(network)
                logger.debug(f"Discovered network: {network_name}")

            except Exception as e:
                logger.warning(f"Failed to parse network {config_file}: {e}")
                continue

        logger.info(f"Discovered {len(networks)} network configurations")
        return networks

    def generate_combinations(
        self, presets: List[PresetInfo], networks: List[NetworkInfo]
    ) -> List[BenchmarkCombination]:
        """
        Generate all preset × network combinations

        Args:
            presets: List of hardware presets
            networks: List of networks

        Returns:
            List of BenchmarkCombination objects
        """
        combinations = [
            BenchmarkCombination(preset=preset, network=network)
            for preset in presets
            for network in networks
        ]

        logger.info(
            f"Generated {len(combinations)} combinations "
            f"({len(presets)} presets × {len(networks)} networks)"
        )
        return combinations

    def run_single_benchmark(self, combo: BenchmarkCombination, progress_bar=None, skip_plots: bool = False, num_samples: int = DEFAULT_NUM_SAMPLES) -> BenchmarkResult:
        """
        Execute a single preset+network combination

        Args:
            combo: Combination to execute
            progress_bar: Optional tqdm progress bar for detailed strategy progress
            skip_plots: If True, skip plot generation (for 2-phase execution)
            num_samples: Number of Monte Carlo samples for network Pareto plots

        Returns:
            BenchmarkResult with execution status and metrics
        """
        start_time = time.time()

        logger.info(f"Running benchmark: {combo.preset.name} × {combo.network.name}")

        # Workspace will be created by generate.py
        combo_workspace = self.benchmark_workspace / combo.workspace_name

        result = BenchmarkResult(
            preset_name=combo.preset.name,
            network_name=combo.network.name,
            workspace_path=combo_workspace,
            status="running",
        )

        # Load config data for database
        with open(combo.preset.config_path) as f:
            preset_config = json.load(f)
        with open(combo.network.config_path) as f:
            network_config = json.load(f)

        # Get or create preset/network IDs
        preset_id = self.db.get_or_create_preset(
            name=combo.preset.name,
            config_data=preset_config,
            technology_node=combo.preset.technology_node,
            cim_memory_type=combo.preset.cim_memory_type,
            description=combo.preset.description,
        )
        network_id = self.db.get_or_create_network(
            name=combo.network.name,
            config_data=network_config,
            num_layers=combo.network.num_layers,
            description=combo.network.description,
        )

        # Create combination record
        combination_id = self.db.upsert_combination(
            preset_id=preset_id,
            network_id=network_id,
            workspace_path=str(combo_workspace.relative_to(self.benchmark_workspace)),
            status="running",
        )

        try:
            # Get workspace name relative to workspaces/
            # combo_workspace is: workspaces/benchmark_.../preset_network
            # We need: benchmark_.../preset_network
            workspace_relative = str(combo_workspace.relative_to(Path("workspaces")))

            # Prepare environment with PYTHONPATH and disable buffering
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path.cwd())
            env['PYTHONUNBUFFERED'] = '1'  # Disable stdout buffering for real-time progress

            # ================================================================
            # Benchmark Execution Steps (dependency order):
            #   Step 1: Generate strategies (creates workspace, strategy JSON files)
            #   Step 2: Simulate (runs SystemC, produces simulation_statistics.json)
            #   Step 3: Aggregate results to central DB (copies results to benchmark DB)
            #   Step 4: Export combination results to JSON/CSV (summary files)
            #   Step 5: Calculate strategy scores (per-layer Pareto ranking)
            #   Step 6: Generate plots (creates pareto.csv, Pareto plots)
            #   Step 7: Pattern analysis (needs pareto.csv from Step 6)
            # ================================================================

            # Step 1: Generate strategies (auto-creates workspace if needed)
            gen_start = time.time()
            gen_cmd = [
                sys.executable,
                "tools/generate.py",
                workspace_relative,
                "--network",
                str(combo.network.config_path),
                "--hardware",
                str(combo.preset.config_path),
            ]
            result_gen = subprocess.run(
                gen_cmd, cwd=Path.cwd(), capture_output=True, text=True, timeout=600, env=env
            )
            if result_gen.returncode != 0:
                logger.error(f"Generate failed for {combo.workspace_name}:")
                logger.error(f"   stdout: {result_gen.stdout}")
                logger.error(f"   stderr: {result_gen.stderr}")
                raise RuntimeError(f"Generate failed: {result_gen.stderr}")
            result.generation_time_sec = time.time() - gen_start

            # Step 2: Simulate (runs SystemC simulator)
            sim_start = time.time()
            sim_cmd = [
                sys.executable,
                "tools/simulate.py",
                workspace_relative,
            ]
            if self.config.max_cpus is not None:
                sim_cmd.extend(["--max-cpus", str(self.config.max_cpus)])
            if self.config.generate_gantt:
                sim_cmd.append("--generate-gantt")
                # Note: --generate-gantt only saves gantt_data.txt (~44MB), not simulation_log.txt (~100MB)
            if self.config.generate_memory_layout:
                sim_cmd.append("--generate-memory-layout")
                # Note: Uses memory_metadata.json (~2KB) which is always generated by SystemC

            # Suppress "Next steps" output in benchmark mode
            sim_cmd.append("--quiet")

            # Run simulation - let output go directly to terminal
            # Don't capture stdout so user can see simulate.py progress in real-time
            result_run = subprocess.run(
                sim_cmd,
                cwd=Path.cwd(),
                capture_output=False,  # Let output go to terminal
                env=env,
            )

            if result_run.returncode != 0:
                raise RuntimeError(f"Simulate failed with return code {result_run.returncode}")

            # Update progress bar to 100% after simulation completes
            if progress_bar:
                progress_bar.update(100 - progress_bar.n)
            result.simulation_time_sec = time.time() - sim_start

            # Step 3: Aggregate results to central DB
            self.aggregate_to_central_db(
                preset_id=preset_id,
                network_id=network_id,
                combination_id=combination_id,
                combo_workspace=combo_workspace,
            )

            # Step 4: Export combination results to JSON/CSV
            self.export_combination_results(
                combo=combo,
                combo_workspace=combo_workspace,
            )

            # Step 5: Calculate strategy scores
            self.calculate_strategy_scores(
                combo=combo,
                combo_workspace=combo_workspace,
            )

            # Steps 6-7: Generate plots and pattern analysis
            # (skip if requested for 2-phase execution)
            if not skip_plots:
                # Step 6: Generate plots (creates pareto.csv)
                self.generate_plots_for_combination(combo, combo_workspace, num_samples)

                # Step 7: Cross-layer pattern analysis (needs pareto.csv from Step 6)
                self.calculate_pattern_analysis(
                    combo=combo,
                    combo_workspace=combo_workspace,
                )

            # Update status to completed
            result.status = "completed"
            self.db.upsert_combination(
                preset_id=preset_id,
                network_id=network_id,
                workspace_path=str(combo_workspace.relative_to(self.benchmark_workspace)),
                status="completed",
            )

            elapsed = time.time() - start_time
            logger.info(f"Benchmark completed: {combo.workspace_name} ({elapsed:.1f}s)")

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            self.db.upsert_combination(
                preset_id=preset_id,
                network_id=network_id,
                workspace_path=str(combo_workspace.relative_to(self.benchmark_workspace)),
                status="failed",
                error_message=str(e),
            )
            logger.error(f"Benchmark failed: {combo.workspace_name}: {e}")

        return result

    def generate_plots_for_combination(
        self,
        combo: BenchmarkCombination,
        combo_workspace: Path,
        num_samples: int = DEFAULT_NUM_SAMPLES,
    ):
        """
        Generate all plots for a combination (separated from simulation for 2-phase execution)

        NOTE: Calls tools/plot.py via subprocess (same pattern as generate/simulate).
              This ensures SINGLE SOURCE OF TRUTH - modify tools/plot.py for plot changes.

        Args:
            combo: Combination metadata
            combo_workspace: Path to combination workspace
            num_samples: Number of Monte Carlo samples for network Pareto plots
        """
        logger.info(f"Generating plots for {combo.workspace_name}")

        # Use subprocess like generate.py and simulate.py
        # plot.py expects workspace path relative to workspaces/ directory
        workspace_relative = str(combo_workspace.relative_to(Path("workspaces")))
        plot_cmd = [
            sys.executable,
            "tools/plot.py",
            workspace_relative,
            "--num-samples", str(num_samples),
            "--progressive",  # Always generate progressive layer plots in benchmark
        ]

        # Add tiling diagrams flag if enabled
        if self.config.tiling_diagrams:
            plot_cmd.append("--tiling-diagrams")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())

        result = subprocess.run(
            plot_cmd,
            cwd=Path.cwd(),
            capture_output=False,
            env=env,
        )

        if result.returncode == 0:
            logger.info(f"Plots completed for {combo.workspace_name}")
        else:
            logger.warning(f"Plot generation failed for {combo.workspace_name}")

    def export_combination_results(
        self,
        combo: BenchmarkCombination,
        combo_workspace: Path,
    ):
        """
        Export combination simulation results to JSON and CSV files

        Args:
            combo: Combination metadata
            combo_workspace: Path to combination workspace
        """
        local_db_path = combo_workspace / "strategies.db"
        if not local_db_path.exists():
            return

        # Prepare results directory
        results_dir = combo_workspace / "results"
        results_dir.mkdir(exist_ok=True)

        # Connect to local DB
        local_conn = sqlite3.connect(str(local_db_path))
        local_conn.row_factory = sqlite3.Row
        local_cursor = local_conn.cursor()

        try:
            # Query all simulation results
            local_cursor.execute(
                """
                SELECT
                    layer_idx, strategy_id,
                    latency_ns, energy_nj, area_mm2,
                    mac_energy_nj, pooling_energy_nj, activation_energy_nj,
                    sram_read_energy_nj, sram_write_energy_nj,
                    dram_read_energy_nj, dram_write_energy_nj,
                    communication_energy_nj, static_energy_nj,
                    ibuf_lines, obuf_lines,
                    input_tile_count, output_tile_count,
                    tiling_config
                FROM strategy_results
                ORDER BY layer_idx, strategy_id
            """
            )

            rows = local_cursor.fetchall()

            # Group results by layer
            results_by_layer = {}
            for row in rows:
                layer_idx = row["layer_idx"]
                if layer_idx not in results_by_layer:
                    results_by_layer[layer_idx] = []

                # Compute EDP
                edp = (
                    row["energy_nj"] * row["latency_ns"]
                    if row["energy_nj"] and row["latency_ns"]
                    else None
                )

                # Compute total compute and memory energy
                compute_energy = None
                memory_energy = None
                if all(
                    row[k] is not None
                    for k in ["mac_energy_nj", "pooling_energy_nj", "activation_energy_nj"]
                ):
                    compute_energy = (
                        row["mac_energy_nj"]
                        + row["pooling_energy_nj"]
                        + row["activation_energy_nj"]
                    )

                if all(
                    row[k] is not None
                    for k in [
                        "sram_read_energy_nj",
                        "sram_write_energy_nj",
                        "dram_read_energy_nj",
                        "dram_write_energy_nj",
                    ]
                ):
                    memory_energy = (
                        row["sram_read_energy_nj"]
                        + row["sram_write_energy_nj"]
                        + row["dram_read_energy_nj"]
                        + row["dram_write_energy_nj"]
                    )

                strategy_result = {
                    "strategy_id": row["strategy_id"],
                    "performance": {
                        "latency_ns": row["latency_ns"],
                        "energy_nj": row["energy_nj"],
                        "area_mm2": row["area_mm2"],
                        "edp": edp,
                    },
                    "energy_breakdown": {
                        "computation": {
                            "mac_nj": row["mac_energy_nj"],
                            "pooling_nj": row["pooling_energy_nj"],
                            "activation_nj": row["activation_energy_nj"],
                            "total_nj": compute_energy,
                        },
                        "memory": {
                            "sram_read_nj": row["sram_read_energy_nj"],
                            "sram_write_nj": row["sram_write_energy_nj"],
                            "dram_read_nj": row["dram_read_energy_nj"],
                            "dram_write_nj": row["dram_write_energy_nj"],
                            "total_nj": memory_energy,
                        },
                        "communication_nj": row["communication_energy_nj"],
                        "static_nj": row["static_energy_nj"],
                    },
                    "buffer_usage": {
                        "ibuf_lines": row["ibuf_lines"],
                        "obuf_lines": row["obuf_lines"],
                    },
                    "tile_counts": {
                        "input_tile_count": row["input_tile_count"],
                        "output_tile_count": row["output_tile_count"],
                    },
                    "tiling_config": (
                        json.loads(row["tiling_config"]) if row["tiling_config"] else None
                    ),
                }

                results_by_layer[layer_idx].append(strategy_result)

            # Create metadata file
            metadata = {
                "preset": combo.preset.name,
                "network": combo.network.name,
                "combination_id": f"{combo.preset.name}_{combo.network.name}",
                "generated_at": datetime.now().isoformat(),
                "configuration": {
                    "preset_config": str(combo.preset.config_path),
                    "network_config": str(combo.network.config_path),
                    "technology_node": combo.preset.technology_node,
                    "cim_memory_type": combo.preset.cim_memory_type,
                    "num_layers": combo.network.num_layers,
                },
                "statistics": {
                    "total_strategies": len(rows),
                    "strategies_per_layer": {
                        layer_idx: len(strategies)
                        for layer_idx, strategies in results_by_layer.items()
                    },
                },
            }

            # Write metadata file
            metadata_file = results_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            # Write per-layer strategy files (JSON + CSV)
            for layer_idx, strategies in results_by_layer.items():
                # Write JSON file
                layer_file = results_dir / f"{layer_idx}.json"
                layer_data = {
                    "layer_idx": layer_idx,
                    "num_strategies": len(strategies),
                    "strategies": strategies,
                }
                with open(layer_file, "w") as f:
                    json.dump(layer_data, f, indent=2)

                # Write CSV file for easy comparison
                csv_file = results_dir / f"{layer_idx}.csv"
                self._write_layer_csv(csv_file, layer_idx, strategies)

            logger.debug(f"Exported results to: {results_dir} ({len(results_by_layer)} layers)")

        except Exception as e:
            logger.warning(f"Failed to export combination results: {e}")

        finally:
            local_conn.close()

    def _write_layer_csv(self, csv_file: Path, layer_idx: int, strategies: list):
        """
        Write layer strategies to CSV file for easy comparison

        Args:
            csv_file: Path to CSV file
            layer_idx: Layer identifier
            strategies: List of strategy results
        """
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header row
            writer.writerow(
                [
                    "strategy_id",
                    "latency_ns",
                    "energy_nj",
                    "edp",
                    "area_mm2",
                    "mac_nj",
                    "pooling_nj",
                    "activation_nj",
                    "compute_total_nj",
                    "sram_read_nj",
                    "sram_write_nj",
                    "dram_read_nj",
                    "dram_write_nj",
                    "memory_total_nj",
                    "communication_nj",
                    "static_nj",
                    "ibuf_lines",
                    "obuf_lines",
                    "input_tile_count",
                    "output_tile_count",
                    "output_tile_p",
                    "output_tile_q",
                    "input_tile_h",
                    "input_tile_w",
                ]
            )

            # Data rows
            for strat in strategies:
                perf = strat["performance"]
                eb = strat["energy_breakdown"]
                comp = eb["computation"]
                mem = eb["memory"]
                buf = strat["buffer_usage"]
                tiles = strat["tile_counts"]
                tiling = strat["tiling_config"]

                # Extract tiling parameters (tiling should be a dict from json.loads on line 574, but double-check)
                if tiling and isinstance(tiling, str):
                    tiling = json.loads(tiling)

                output_tile_p = tiling.get("output_tile_p") if tiling else None
                output_tile_q = tiling.get("output_tile_q") if tiling else None
                input_tile_h = tiling.get("input_tile_h") if tiling else None
                input_tile_w = tiling.get("input_tile_w") if tiling else None

                writer.writerow(
                    [
                        strat["strategy_id"],
                        perf["latency_ns"],
                        perf["energy_nj"],
                        perf["edp"],
                        perf["area_mm2"],
                        comp["mac_nj"],
                        comp["pooling_nj"],
                        comp["activation_nj"],
                        comp["total_nj"],
                        mem["sram_read_nj"],
                        mem["sram_write_nj"],
                        mem["dram_read_nj"],
                        mem["dram_write_nj"],
                        mem["total_nj"],
                        eb["communication_nj"],
                        eb["static_nj"],
                        buf["ibuf_lines"],
                        buf["obuf_lines"],
                        tiles["input_tile_count"],
                        tiles["output_tile_count"],
                        output_tile_p,
                        output_tile_q,
                        input_tile_h,
                        input_tile_w,
                    ]
                )

    def calculate_strategy_scores(
        self,
        combo: BenchmarkCombination,
        combo_workspace: Path,
    ):
        """
        Calculate usefulness scores for all strategies in a combination workspace.

        Scores are saved to the database (strategy_scores table) and CSV file.

        Args:
            combo: Combination metadata
            combo_workspace: Path to combination workspace
        """
        db_path = combo_workspace / "strategies.db"
        if not db_path.exists():
            return

        try:
            scorer = StrategyScorer(db_path)
            scorer.score_all_layers()
            scorer.save_to_db()
            scorer.save_to_csv(combo_workspace / "strategy_scores.csv")
            logger.debug(f"Strategy scores calculated for {combo.workspace_name}")
        except Exception as e:
            logger.warning(f"Failed to calculate strategy scores: {e}")

    def calculate_pattern_analysis(
        self,
        combo: BenchmarkCombination,
        combo_workspace: Path,
    ):
        """
        Perform cross-layer pattern analysis for a combination workspace.

        Extracts ratio-based tiling patterns and scores them across all layers.
        Generates pattern analysis CSV files and visualization plots.

        Output structure:
            plots/patterns/
            ├── network_full/          # Full network analysis
            │   └── pattern_*.pdf
            └── progressive/           # Progressive layer analysis
                ├── layers_2/
                ├── layers_3/
                └── layers_N/

        Args:
            combo: Combination metadata
            combo_workspace: Path to combination workspace
        """
        db_path = combo_workspace / "strategies.db"
        if not db_path.exists():
            return

        try:
            analyzer = PatternAnalyzer(db_path)
            all_layers = analyzer.get_layer_indices()
            num_layers = len(all_layers)

            if num_layers == 0:
                logger.debug(f"No layers found for {combo.workspace_name}")
                return

            # Output structure:
            #   pattern_analysis/
            #   ├── layers_2.csv, layers_3.csv, ..., layers_N.csv
            #   └── detailed/
            #       └── layers_2_detailed.csv, ...
            #
            #   plots/patterns/
            #   ├── network_full/      (= layers_N)
            #   └── progressive/
            #       └── layers_2/, layers_3/, ...

            csv_dir = combo_workspace / "pattern_analysis"
            csv_dir.mkdir(parents=True, exist_ok=True)
            detailed_dir = csv_dir / "detailed"
            detailed_dir.mkdir(parents=True, exist_ok=True)

            # Progressive analysis: 2, 3, ..., N layers
            # Note: layers_N = full network (no separate "network_full" CSV needed)
            for n_layers in range(2, num_layers + 1):
                layer_subset = all_layers[:n_layers]
                patterns = analyzer.analyze_all_layers(layer_indices=layer_subset)

                if patterns:
                    # Load network pareto presence from existing plot data
                    # (pareto.csv is generated by plot generation in Step 6)
                    analyzer.compute_network_pareto_presence(
                        layer_indices=layer_subset,
                        quiet=True,
                    )

                    # Save CSV (score is specific to this layer subset)
                    analyzer.save_to_csv(csv_dir / f"layers_{n_layers}.csv")
                    analyzer.save_detailed_csv(detailed_dir / f"layers_{n_layers}_detailed.csv")

                    # Generate plots
                    if n_layers == num_layers:
                        # Full network goes to network_full/
                        plot_dir = combo_workspace / "plots" / "patterns" / "network_full"
                    else:
                        # Progressive goes to progressive/layers_N/
                        plot_dir = combo_workspace / "plots" / "patterns" / "progressive" / f"layers_{n_layers}"
                    analyzer.generate_plots(plot_dir)

            logger.debug(
                f"Pattern analysis completed for {combo.workspace_name}: "
                f"{len(patterns)} patterns, {num_layers} layers"
            )
        except Exception as e:
            logger.warning(f"Failed to calculate pattern analysis: {e}")

    def aggregate_to_central_db(
        self,
        preset_id: int,
        network_id: int,
        combination_id: int,
        combo_workspace: Path,
    ):
        """
        Aggregate results from combination workspace to central database

        Args:
            preset_id: Preset database ID
            network_id: Network database ID
            combination_id: Combination database ID
            combo_workspace: Path to combination workspace
        """
        # Connect to local strategies.db (silent mode for progress bar)

        local_db_path = combo_workspace / "strategies.db"
        if not local_db_path.exists():
            return

        local_conn = sqlite3.connect(str(local_db_path))
        local_conn.row_factory = sqlite3.Row
        local_cursor = local_conn.cursor()

        # Copy simulation results from strategy_results table
        try:
            local_cursor.execute(
                """
                SELECT layer_idx, strategy_id, latency_ns, energy_nj, area_mm2,
                       ibuf_lines, obuf_lines,
                       mac_energy_nj, pooling_energy_nj, activation_energy_nj,
                       sram_read_energy_nj, sram_write_energy_nj,
                       dram_read_energy_nj, dram_write_energy_nj,
                       communication_energy_nj, static_energy_nj,
                       ibuf_area_mm2, obuf_area_mm2, cim_area_mm2
                FROM strategy_results
            """
            )

            rows = local_cursor.fetchall()

            # Build set of valid layer indices from layer files
            valid_layer_indices = set()
            layers_dir = combo_workspace / "layers"
            if layers_dir.exists():
                for layer_file in layers_dir.glob("L*.json"):
                    # Extract index from filename Lx.json
                    valid_layer_indices.add(int(layer_file.stem[1:]))

            count = 0
            for row in rows:
                # Calculate EDP
                edp = row["energy_nj"] * row["latency_ns"]

                # Skip if layer_idx not in valid layers
                layer_idx = row["layer_idx"]
                if layer_idx not in valid_layer_indices:
                    continue

                # Extract energy breakdown from local DB
                mac_energy = row["mac_energy_nj"]
                pooling_energy = row["pooling_energy_nj"]
                activation_energy = row["activation_energy_nj"]
                sram_read = row["sram_read_energy_nj"]
                sram_write = row["sram_write_energy_nj"]
                dram_read = row["dram_read_energy_nj"]
                dram_write = row["dram_write_energy_nj"]
                communication = row["communication_energy_nj"]
                static = row["static_energy_nj"]

                # Compute total computation and memory energy
                compute_energy = None
                memory_energy = None
                if all(x is not None for x in [mac_energy, pooling_energy, activation_energy]):
                    compute_energy = mac_energy + pooling_energy + activation_energy

                if all(x is not None for x in [sram_read, sram_write, dram_read, dram_write]):
                    memory_energy = sram_read + sram_write + dram_read + dram_write

                try:
                    self.db.insert_simulation_result(
                        preset_id=preset_id,
                        network_id=network_id,
                        combination_id=combination_id,
                        layer_idx=layer_idx,
                        strategy_id=row["strategy_id"],
                        latency_ns=row["latency_ns"],
                        energy_nj=row["energy_nj"],
                        area_mm2=row["area_mm2"],
                        edp=edp,
                        # Energy breakdown from local DB
                        compute_energy_nj=compute_energy,
                        memory_energy_nj=memory_energy,
                        communication_energy_nj=communication,
                        static_energy_nj=static,
                        # Area breakdown from local DB
                        ibuf_area_mm2=row["ibuf_area_mm2"],
                        obuf_area_mm2=row["obuf_area_mm2"],
                        cim_area_mm2=row["cim_area_mm2"],
                    )
                    count += 1
                except Exception as e:
                    logger.debug(f"Failed to insert result: {e}")

            # Commit all inserts at once (batch commit for performance)
            # This avoids 9000+ individual commits which causes ~1min delay
            if count > 0:
                self.db.commit_simulation_results()

        except sqlite3.OperationalError as e:
            logger.debug(f"DB operation error during aggregation: {e}")

        local_conn.close()

        # Note: Optimization results are no longer aggregated here.
        # They are computed on-demand via the analysis tool using SQL queries
        # on the simulation_results table.

    def run_full_sweep(self, num_samples: int = DEFAULT_NUM_SAMPLES, network_filter: List[str] = None):
        """
        Execute full benchmark sweep: all presets × all networks

        Args:
            num_samples: Number of Monte Carlo samples for network Pareto plots
            network_filter: Optional list of network names to include

        This is the main entry point for benchmark execution.
        """
        logger.info("Starting full benchmark sweep")

        # Update status to running
        self._update_status("running")

        # Discovery phase
        presets = self.discover_presets()
        networks = self.discover_networks(filter_names=network_filter)

        if not presets:
            logger.error("No presets found. Aborting benchmark.")
            self._update_status("failed")
            return

        if not networks:
            logger.error("No networks found. Aborting benchmark.")
            self._update_status("failed")
            return

        # Generate combinations
        combinations = self.generate_combinations(presets, networks)

        # Update progress
        self._update_progress(total=len(combinations))

        # Print benchmark plan
        logger.info("=" * 80)
        logger.info(f"BENCHMARK PLAN: {len(combinations)} combinations to execute")
        logger.info("=" * 80)
        for i, combo in enumerate(combinations, 1):
            logger.info(f"  {i}. {combo.workspace_name}")
        logger.info("=" * 80)

        # Execute simulations with immediate plot generation after each
        # (Simulation → Plot → Next Simulation workflow)
        results = []

        logger.info(f"Running {len(combinations)} combinations (Simulate → Plot for each)")
        logger.info(f"   (Steps 1-7: Generate → Simulate → Aggregate → Export → Scores → Plots → Pattern)")

        for combo_idx, combo in enumerate(combinations, 1):
            logger.info("=" * 80)
            logger.info(f"[{combo_idx}/{len(combinations)}] {combo.workspace_name}")
            logger.info("=" * 80)

            try:
                # Steps 1-5: Simulation
                result = self.run_single_benchmark(combo, progress_bar=None, skip_plots=True)
                results.append(result)

                if result.status == "completed":
                    logger.info(f"Simulation completed: {combo.workspace_name}")

                    # Steps 6-7: Immediate plot generation after successful simulation
                    try:
                        logger.info(f"Generating plots for {combo.workspace_name}...")
                        self.generate_plots_for_combination(combo, result.workspace_path, num_samples)
                        self.calculate_pattern_analysis(combo, result.workspace_path)
                        logger.info(f"Plots completed: {combo.workspace_name}")
                    except Exception as plot_e:
                        logger.warning(f"Plot generation failed: {plot_e}")

                self._update_progress(completed=1)
                logger.info(f"[{combo_idx}/{len(combinations)}] Completed: {combo.workspace_name}")

            except Exception as e:
                logger.error(f"[{combo_idx}/{len(combinations)}] Failed: {combo.workspace_name}: {e}")
                results.append(BenchmarkResult(
                    preset_name=combo.preset.name,
                    network_name=combo.network.name,
                    workspace_path=self.benchmark_workspace / combo.workspace_name,
                    status="failed",
                    error_message=str(e),
                ))
                self._update_progress(failed=1)

        logger.info("=" * 80)
        logger.info(f"BENCHMARK COMPLETE: {sum(1 for r in results if r.status == 'completed')}/{len(results)} successful")
        logger.info("=" * 80)

        # Update final status
        if all(r.status == "completed" for r in results):
            self._update_status("completed")
        elif any(r.status == "completed" for r in results):
            self._update_status("partial")
        else:
            self._update_status("failed")

        logger.info(f"Benchmark sweep completed: {len(results)} combinations processed")

        # Generate HV improvement analysis report
        try:
            self.generate_hv_improvement_report()
        except Exception as e:
            logger.warning(f"Failed to generate HV improvement report: {e}")

        # Generate three-way HV comparison CSV
        try:
            self.generate_three_way_hv_report()
        except Exception as e:
            logger.warning(f"Failed to generate three-way HV report: {e}")

    def _update_status(self, status: str):
        """Update benchmark status in benchmark_config.json"""
        config_path = self.benchmark_workspace / "benchmark_config.json"
        with open(config_path, "r") as f:
            config_data = json.load(f)

        config_data["status"] = status
        if status == "running" and not config_data["timing"]["started_at"]:
            config_data["timing"]["started_at"] = datetime.now().isoformat()
        elif status in ("completed", "failed", "partial"):
            config_data["timing"]["completed_at"] = datetime.now().isoformat()

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    def _update_progress(self, total: Optional[int] = None, completed: int = 0, failed: int = 0):
        """Update progress in benchmark_config.json"""
        config_path = self.benchmark_workspace / "benchmark_config.json"
        with open(config_path, "r") as f:
            config_data = json.load(f)

        progress = config_data["progress"]
        if total is not None:
            progress["total_combinations"] = total
            progress["pending"] = total
        progress["completed"] += completed
        progress["failed"] += failed
        progress["pending"] = (
            progress["total_combinations"] - progress["completed"] - progress["failed"]
        )

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    def generate_hv_improvement_report(self):
        """
        Generate Hypervolume improvement analysis report after benchmark completion.

        Analyzes all workspace combinations and computes HV improvements for:
        - Decoupled vs Coupled (search space expansion contribution)
        - Simulation vs Analytical (accuracy contribution)

        Output: HV_IMPROVEMENT_ANALYSIS.md in benchmark workspace
        """
        if not HAS_NUMPY:
            logger.warning("numpy not available, skipping HV analysis")
            return

        logger.info("Generating HV improvement analysis report...")

        def load_pareto_data(csv_path: Path, x_col: str = 'latency_ns', y_col: str = 'buffer_eap'):
            """Load Pareto points from CSV"""
            if not csv_path.exists():
                return None
            import csv as csv_module
            points = []
            with open(csv_path) as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    try:
                        x = float(row[x_col])
                        y = float(row[y_col])
                        points.append((x, y))
                    except (KeyError, ValueError):
                        continue
            return np.array(points) if points else None

        def compute_hypervolume_2d(points, ref_point):
            """Compute 2D hypervolume (both objectives minimized)"""
            if points is None or len(points) == 0:
                return 0.0
            sorted_pts = points[np.argsort(points[:, 0])]
            pareto = []
            min_y = float('inf')
            for pt in sorted_pts:
                if pt[1] < min_y:
                    pareto.append(pt)
                    min_y = pt[1]
            if not pareto:
                return 0.0
            pareto = np.array(pareto)
            hv = 0.0
            prev_x = 0
            for pt in pareto:
                if pt[0] > ref_point[0] or pt[1] > ref_point[1]:
                    continue
                width = pt[0] - prev_x
                height = ref_point[1] - pt[1]
                hv += width * height
                prev_x = pt[0]
            if prev_x < ref_point[0] and len(pareto) > 0:
                hv += (ref_point[0] - prev_x) * (ref_point[1] - pareto[-1, 1])
            return hv

        results = []

        for ws_dir in sorted(self.benchmark_workspace.iterdir()):
            if not ws_dir.is_dir() or ws_dir.name in ['results.db', 'benchmark_config.json']:
                continue

            # Parse hardware and network from workspace name
            parts = ws_dir.name.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in ['lenet5', 'mobilenet', 'resnet18', 'vgg11']:
                hardware, network = parts[0], parts[1]
            elif ws_dir.name.endswith('_4layers'):
                parts = ws_dir.name.rsplit('_', 2)
                hardware = parts[0]
                network = f"{parts[1]}_{parts[2]}"
            else:
                hardware, network = ws_dir.name, "unknown"

            result = {"workspace": ws_dir.name, "hardware": hardware, "network": network}

            # Buffer EAP analysis
            all_eap = ws_dir / "plots/network_full/all/per_objective/pareto_latency_ns_vs_buffer_eap.csv"
            coupled_eap = ws_dir / "plots/network_full/coupled/per_objective/pareto_latency_ns_vs_buffer_eap.csv"
            analytical_eap = ws_dir / "plots/network_full/analytical/per_objective/pareto_latency_ns_vs_buffer_eap.csv"

            all_eap_pts = load_pareto_data(all_eap)
            coupled_eap_pts = load_pareto_data(coupled_eap)
            analytical_eap_pts = load_pareto_data(analytical_eap)

            all_pts_list = [p for p in [all_eap_pts, coupled_eap_pts, analytical_eap_pts] if p is not None]
            if all_pts_list:
                combined = np.vstack(all_pts_list)
                ref_eap = [combined[:, 0].max() * 1.1, combined[:, 1].max() * 1.1]
                hv_all = compute_hypervolume_2d(all_eap_pts, ref_eap)
                hv_coupled = compute_hypervolume_2d(coupled_eap_pts, ref_eap)
                hv_analytical = compute_hypervolume_2d(analytical_eap_pts, ref_eap)

                if hv_coupled > 0:
                    result["eap_decoupled_vs_coupled"] = (hv_all - hv_coupled) / hv_coupled * 100
                if hv_analytical > 0:
                    result["eap_sim_vs_analytical"] = (hv_all - hv_analytical) / hv_analytical * 100

            # Buffer Area analysis
            all_area = ws_dir / "plots/network_full/all/per_objective/pareto_latency_ns_vs_buffer_area_mm2.csv"
            coupled_area = ws_dir / "plots/network_full/coupled/per_objective/pareto_latency_ns_vs_buffer_area_mm2.csv"
            analytical_area = ws_dir / "plots/network_full/analytical/per_objective/pareto_latency_ns_vs_buffer_area_mm2.csv"

            all_area_pts = load_pareto_data(all_area, y_col='buffer_area_mm2')
            coupled_area_pts = load_pareto_data(coupled_area, y_col='buffer_area_mm2')
            analytical_area_pts = load_pareto_data(analytical_area, y_col='buffer_area_mm2')

            all_pts_list = [p for p in [all_area_pts, coupled_area_pts, analytical_area_pts] if p is not None]
            if all_pts_list:
                combined = np.vstack(all_pts_list)
                ref_area = [combined[:, 0].max() * 1.1, combined[:, 1].max() * 1.1]
                hv_all = compute_hypervolume_2d(all_area_pts, ref_area)
                hv_coupled = compute_hypervolume_2d(coupled_area_pts, ref_area)
                hv_analytical = compute_hypervolume_2d(analytical_area_pts, ref_area)

                if hv_coupled > 0:
                    result["area_decoupled_vs_coupled"] = (hv_all - hv_coupled) / hv_coupled * 100
                if hv_analytical > 0:
                    result["area_sim_vs_analytical"] = (hv_all - hv_analytical) / hv_analytical * 100

            if len(result) > 3:
                results.append(result)

        if not results:
            logger.warning("No valid results for HV analysis")
            return

        # Generate markdown report
        output = f"""# Hypervolume Improvement Analysis
## Benchmark: {self.benchmark_workspace.name}

### Summary

| Metric | Comparison | Best Setting | Improvement |
|--------|------------|--------------|-------------|
"""

        # Find best for each category
        def find_best(key):
            valid = [r for r in results if key in r]
            return max(valid, key=lambda x: x[key]) if valid else None

        best_eap_dc = find_best('eap_decoupled_vs_coupled')
        best_area_dc = find_best('area_decoupled_vs_coupled')
        best_eap_sa = find_best('eap_sim_vs_analytical')
        best_area_sa = find_best('area_sim_vs_analytical')

        if best_eap_dc:
            output += f"| Buffer EAP | Decoupled vs Coupled | `{best_eap_dc['workspace']}` | **+{best_eap_dc['eap_decoupled_vs_coupled']:.1f}%** |\n"
        if best_area_dc:
            output += f"| Buffer Area | Decoupled vs Coupled | `{best_area_dc['workspace']}` | **+{best_area_dc['area_decoupled_vs_coupled']:.1f}%** |\n"
        if best_eap_sa:
            output += f"| Buffer EAP | Sim vs Analytical | `{best_eap_sa['workspace']}` | **+{best_eap_sa['eap_sim_vs_analytical']:.1f}%** |\n"
        if best_area_sa:
            output += f"| Buffer Area | Sim vs Analytical | `{best_area_sa['workspace']}` | **+{best_area_sa['area_sim_vs_analytical']:.1f}%** |\n"

        output += """
---

## Decoupled vs Coupled (Contribution: Expanded Search Space)

### Buffer EAP (Latency vs Buffer Energy-Area Product)

| Rank | Hardware | Network | HV Improvement |
|------|----------|---------|----------------|
"""
        sorted_eap_dc = sorted([r for r in results if 'eap_decoupled_vs_coupled' in r],
                               key=lambda x: x['eap_decoupled_vs_coupled'], reverse=True)
        for i, r in enumerate(sorted_eap_dc, 1):
            val = r['eap_decoupled_vs_coupled']
            sign = "+" if val >= 0 else ""
            output += f"| {i} | {r['hardware']} | {r['network']} | {sign}{val:.1f}% |\n"

        output += """
### Buffer Area (Latency vs Buffer Area)

| Rank | Hardware | Network | HV Improvement |
|------|----------|---------|----------------|
"""
        sorted_area_dc = sorted([r for r in results if 'area_decoupled_vs_coupled' in r],
                                key=lambda x: x['area_decoupled_vs_coupled'], reverse=True)
        for i, r in enumerate(sorted_area_dc, 1):
            val = r['area_decoupled_vs_coupled']
            sign = "+" if val >= 0 else ""
            output += f"| {i} | {r['hardware']} | {r['network']} | {sign}{val:.1f}% |\n"

        output += """
---

## Simulation vs Analytical (Contribution: Accurate Modeling)

### Buffer EAP (Latency vs Buffer Energy-Area Product)

| Rank | Hardware | Network | HV Improvement |
|------|----------|---------|----------------|
"""
        sorted_eap_sa = sorted([r for r in results if 'eap_sim_vs_analytical' in r],
                               key=lambda x: x['eap_sim_vs_analytical'], reverse=True)
        for i, r in enumerate(sorted_eap_sa, 1):
            val = r['eap_sim_vs_analytical']
            sign = "+" if val >= 0 else ""
            output += f"| {i} | {r['hardware']} | {r['network']} | {sign}{val:.1f}% |\n"

        output += """
### Buffer Area (Latency vs Buffer Area)

| Rank | Hardware | Network | HV Improvement |
|------|----------|---------|----------------|
"""
        sorted_area_sa = sorted([r for r in results if 'area_sim_vs_analytical' in r],
                                key=lambda x: x['area_sim_vs_analytical'], reverse=True)
        for i, r in enumerate(sorted_area_sa, 1):
            val = r['area_sim_vs_analytical']
            sign = "+" if val >= 0 else ""
            output += f"| {i} | {r['hardware']} | {r['network']} | {sign}{val:.1f}% |\n"

        output += """
---

## Recommended Settings for Paper

### For Maximum Decoupled vs Coupled Improvement:
"""
        if best_eap_dc:
            output += f"- **Buffer EAP**: `{best_eap_dc['workspace']}` (+{best_eap_dc['eap_decoupled_vs_coupled']:.1f}%)\n"
        if best_area_dc:
            output += f"- **Buffer Area**: `{best_area_dc['workspace']}` (+{best_area_dc['area_decoupled_vs_coupled']:.1f}%)\n"

        output += """
### For Maximum Simulation vs Analytical Improvement:
"""
        if best_eap_sa:
            output += f"- **Buffer EAP**: `{best_eap_sa['workspace']}` (+{best_eap_sa['eap_sim_vs_analytical']:.1f}%)\n"
        if best_area_sa:
            output += f"- **Buffer Area**: `{best_area_sa['workspace']}` (+{best_area_sa['area_sim_vs_analytical']:.1f}%)\n"

        # Find settings good for both contributions
        output += """
### Settings Good for Both Contributions:

| Workspace | Metric | Decoupled vs Coupled | Sim vs Analytical | Combined |
|-----------|--------|---------------------|-------------------|----------|
"""
        both_good = []
        for r in results:
            eap_dc = r.get('eap_decoupled_vs_coupled', -999)
            eap_sa = r.get('eap_sim_vs_analytical', -999)
            area_dc = r.get('area_decoupled_vs_coupled', -999)
            area_sa = r.get('area_sim_vs_analytical', -999)

            if eap_dc > 0 and eap_sa > 0:
                both_good.append((r['workspace'], 'Buffer EAP', eap_dc, eap_sa))
            if area_dc > 0 and area_sa > 0:
                both_good.append((r['workspace'], 'Buffer Area', area_dc, area_sa))

        both_good.sort(key=lambda x: x[2] + x[3], reverse=True)
        for ws, metric, dc, sa in both_good[:10]:
            output += f"| {ws} | {metric} | +{dc:.1f}% | +{sa:.1f}% | +{dc+sa:.1f}% |\n"

        # Save report
        report_path = self.benchmark_workspace / "HV_IMPROVEMENT_ANALYSIS.md"
        with open(report_path, "w") as f:
            f.write(output)

        logger.info(f"HV improvement report saved: {report_path}")

    def generate_three_way_hv_report(self):
        """
        Generate three-way HV comparison report (All vs Coupled vs Analytical).

        Creates CSV and JSON files comparing Hypervolume across all workspaces:
        - three_way_hv_comparison.csv: Per-workspace HV for all 7 objectives
        - three_way_hv_summary.csv: Summary statistics
        - three_way_hv_comparison.json: Full data for programmatic access
        """
        logger.info("Generating three-way HV comparison report...")

        try:
            # Import the analyze_benchmark function from tools/plot_three_way_comparison.py
            import sys
            tools_path = Path(__file__).parent
            if str(tools_path) not in sys.path:
                sys.path.insert(0, str(tools_path))

            from plot_three_way_comparison import analyze_benchmark

            # Run analysis on benchmark workspace
            summary = analyze_benchmark(self.benchmark_workspace)

            if summary:
                logger.info(f"Three-way HV report saved to: {self.benchmark_workspace}")
                logger.info(f"  - three_way_hv_comparison.csv")
                logger.info(f"  - three_way_hv_summary.csv")
                logger.info(f"  - three_way_hv_comparison.json")
            else:
                logger.warning("Three-way HV analysis returned no results")

        except ImportError as e:
            logger.warning(f"Could not import plot_three_way_comparison: {e}")
        except Exception as e:
            logger.warning(f"Failed to generate three-way HV report: {e}")
            import traceback
            logger.debug(traceback.format_exc())


# ============================================================================
# CLI Interface
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="CiMFlowSim DSE Benchmark - Automated Design Space Exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all hardware × all networks (full DSE)
  python3 tools/dse_benchmark.py --all

  # Run specific hardware with specific networks
  python3 tools/dse_benchmark.py --hardware area_optimized,balanced --networks lenet5,vgg11

  # Limit CPU usage to 90 cores
  python3 tools/dse_benchmark.py --all --max-cpus 90

  # Generate timing diagrams
  python3 tools/dse_benchmark.py --all --generate-gantt
        """,
    )

    parser.add_argument("--all", action="store_true", help="Run all hardware × all networks")
    parser.add_argument(
        "--hardware",
        type=str,
        help="Comma-separated list of hardware names (e.g., area_optimized,balanced)",
    )
    parser.add_argument(
        "--hardware-path",
        type=Path,
        help="Custom path to hardware config JSON (overrides --hardware, for sweep)",
    )
    parser.add_argument(
        "--networks",
        type=str,
        help="Comma-separated list of network names (e.g., lenet5,vgg11)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for benchmark workspace (for sweep)",
    )
    parser.add_argument(
        "--max-cpus",
        type=int,
        default=None,
        help="Limit Ray cluster CPU usage (e.g., 90 to use 90 CPUs out of 180). If not specified, uses all available CPUs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without running benchmarks",
    )
    parser.add_argument(
        "--generate-gantt",
        action="store_true",
        help="Generate Gantt chart timing diagrams for all strategies (slower, ~85KB per strategy)",
    )
    parser.add_argument(
        "--generate-memory-layout",
        action="store_true",
        help="Generate memory layout visualizations for all strategies (slower, ~45KB per strategy)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of Monte Carlo samples for network Pareto plots (default: {DEFAULT_NUM_SAMPLES:,})",
    )
    parser.add_argument(
        "--tiling-diagrams",
        action="store_true",
        help="Generate tiling strategy diagrams (grid and schematic) for all strategies",
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging
    setup_component_loggers(base_level=logging.INFO)

    logger.info("=" * 80)
    logger.info("CiMFlowSim Benchmark Suite")
    logger.info("=" * 80)

    # Build configuration
    config = BenchmarkConfig(
        presets=[],  # Will be populated by discovery
        networks=[],  # Will be populated by discovery
        max_cpus=args.max_cpus,
        generate_gantt=args.generate_gantt,
        generate_memory_layout=args.generate_memory_layout,
        tiling_diagrams=args.tiling_diagrams,
        hardware_path=args.hardware_path,
        output_dir=args.output_dir,
    )

    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(config)

    # Parse network filter
    network_filter = None
    if args.networks:
        network_filter = [n.strip() for n in args.networks.split(",")]
        logger.info(f"Filtering networks: {network_filter}")

    if args.dry_run:
        logger.info("Dry-run mode: showing execution plan only")
        presets = orchestrator.discover_presets()
        networks = orchestrator.discover_networks(filter_names=network_filter)
        combinations = orchestrator.generate_combinations(presets, networks)

        logger.info("Execution Plan:")
        logger.info(f"  Presets:  {len(presets)}")
        for p in presets:
            logger.info(f"    - {p.name} ({p.technology_node}, {p.cim_memory_type})")
        logger.info(f"  Networks: {len(networks)}")
        for n in networks:
            logger.info(f"    - {n.name} ({n.num_layers} layers)")
        logger.info(f"  Total combinations: {len(combinations)}")
        if args.max_cpus:
            logger.info(f"  Max CPUs: {args.max_cpus}")
        else:
            logger.info(f"  Max CPUs: auto (all available)")
        return

    # Execute benchmark
    try:
        orchestrator.run_full_sweep(num_samples=args.num_samples, network_filter=network_filter)
        logger.info(f"Benchmark completed: {orchestrator.benchmark_workspace}")

        # Generate benchmark-level HV-by-layers comparison (across hardware)
        try:
            from plot_hv_by_layers import plot_benchmark_hv_by_layers
            logger.info("Generating benchmark-level HV-by-layers plots...")
            result = plot_benchmark_hv_by_layers(orchestrator.benchmark_workspace)
            if result.get("success"):
                logger.info(f"  Saved to: {result.get('output_dir')}")
            else:
                logger.warning(f"  Failed: {result.get('error', 'unknown')}")
        except Exception as e:
            logger.warning(f"  HV-by-layers benchmark plots failed: {e}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
