#!/usr/bin/env python3
"""
Step 2: Strategy Simulation
Runs SystemC simulation on pre-generated strategies with automatic validation.
Results are saved to a reusable database for Phase 2 optimization.

Usage:
    ./efsim simulate <workspace>
    ./efsim simulate my_experiment --verbose
"""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
import subprocess
import sys
import time
import warnings

# Suppress Ray's pkg_resources deprecation warning (Ray internal issue)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import ray
import ray.exceptions
from datetime import datetime
from pathlib import Path
import socket

import orjson

# Use common utilities (sets up Python path, signal handlers, etc.)
from common import setup_signal_handlers, setup_ray_cleanup, get_task_registry

# Get the shared task registry for tracking Ray tasks
_task_registry = get_task_registry()

# Setup signal handlers with task registry
setup_signal_handlers(_task_registry)
setup_ray_cleanup()

from core.constants import (  # Core metadata; Workflow phase names; Log policies
    DEFAULT_TIMEOUT_SECONDS,
    LOG_POLICY_ALL,
    LOG_POLICY_FAILED,
    LOG_POLICY_NONE,
    PROGRESS_BAR_LENGTH,
    RAY_HEAD_ADDRESS,
    SIMULATION_PHASE_NAME,
    TOOL_VERSION,
    WORKER_NFS_BASE,
)
from core.area_calculator import AreaCalculator
from core.logging_config import get_logger
from core.strategy_database import StrategyDatabase
from core.tiling import (
    CNNLayerParams,
    TilingConfig,
)
from core.systemc_runner import SystemCRunner
from core.simulation_validator import SimulationValidator
from core.workspace_manager import WorkspaceManager

logger = get_logger(__name__)

# Log policy default
DEFAULT_LOG_POLICY = LOG_POLICY_FAILED

# Error types (explicit classification)
ERROR_TYPE_SIMULATION = "SIMULATION_FAILURE"
ERROR_TYPE_EXCEPTION = "EXCEPTION"
ERROR_TYPE_VALIDATION = "VALIDATION_FAILURE"
ERROR_TYPE_UNKNOWN = "UNKNOWN"

# Convert string constant to Path object (from core.constants.workflow)
WORKER_NFS_PATH = Path(WORKER_NFS_BASE)


def _init_ray(num_cpus: int = None) -> None:
    """
    Initialize Ray cluster connection.

    Tries to connect to existing cluster first, falls back to local Ray.

    Args:
        num_cpus: Number of CPUs to use (only for local Ray, ignored for cluster)
    """
    if ray.is_initialized():
        return

    import os
    # Set Ray scheduler to consider all nodes equally (no top-k preference)
    # This prevents head node from getting disproportionate task allocation
    os.environ.setdefault("RAY_scheduler_top_k_fraction", "1.0")

    project_root = str(Path(__file__).parent.parent)
    python_path = str(Path(__file__).parent.parent / "src" / "python")
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{python_path}:{project_root}",
        },
    }

    try:
        # Try to connect to existing cluster
        ray.init(
            address=RAY_HEAD_ADDRESS,
            ignore_reinit_error=True,
            logging_level="WARNING",
            runtime_env=runtime_env,
        )
    except Exception:
        # Fallback to local Ray
        ray.init(
            ignore_reinit_error=True,
            logging_level="WARNING",
            num_cpus=num_cpus,
            runtime_env=runtime_env,
        )


# Batch processing disabled - use individual tasks with SPREAD scheduling
# This ensures even distribution across nodes

# Use SPREAD scheduling to distribute tasks evenly across nodes
# This prevents head node from processing most tasks


# =============================================================================
# Local Caching Functions (NFS contention optimization)
# =============================================================================


@ray.remote(num_cpus=1, scheduling_strategy="SPREAD")
def _setup_local_cache_on_node(
    workspace_name: str, archive_path: str, layer_idx: int
) -> dict:
    """
    Setup local cache on a single node by extracting archive from NFS.

    Called once per node to extract compressed archive to /mnt/workers/{hostname}/.
    This path is bind-mounted locally on each worker and NFS-accessible from head.
    Returns node info for tracking.
    """
    import socket
    import subprocess

    hostname = socket.gethostname()
    node_id = ray.get_runtime_context().get_node_id()

    # Use /mnt/workers/{hostname}/{workspace}/ for both input and output
    # This simplifies cleanup (one location) and is local on workers
    local_dir = WORKER_NFS_PATH / hostname / workspace_name
    local_dir.mkdir(parents=True, exist_ok=True)

    files_cached = 0

    # Extract archive using pigz for parallel decompression
    archive = Path(archive_path)
    if archive.exists():
        try:
            # Use pigz -d for parallel decompression, tar for extraction
            pigz_proc = subprocess.Popen(
                ["pigz", "-d", "-c", str(archive)],
                stdout=subprocess.PIPE,
            )
            subprocess.run(
                ["tar", "-xf", "-", "-C", str(local_dir)],
                stdin=pigz_proc.stdout,
                check=True,
            )
            pigz_proc.wait()

            # Count extracted files
            for subdir in ["strategies", "layers"]:
                subdir_path = local_dir / subdir
                if subdir_path.exists():
                    files_cached += len(list(subdir_path.glob("*.json")))
            # Config files
            for config_name in ["hardware_config.json", "network_config.json"]:
                if (local_dir / config_name).exists():
                    files_cached += 1

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to Python tarfile
            import tarfile
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=local_dir)
                files_cached = len(tar.getnames())

    # Create output subdirectories
    (local_dir / "ray_results").mkdir(parents=True, exist_ok=True)
    (local_dir / "simulations").mkdir(parents=True, exist_ok=True)

    return {
        "node_id": node_id,
        "hostname": hostname,
        "local_dir": str(local_dir),
        "files_cached": files_cached,
    }


def _broadcast_input_files(workspace_path: Path, layer_idx: int) -> str:
    """
    Broadcast input files to all nodes' local cache using compressed archive.

    Creates a compressed archive on HEAD, then each node extracts it locally.
    Much faster than copying individual files over NFS.

    Files included:
    - hardware_config.json
    - network_config.json
    - layers/L{layer_idx}.json (layer config)
    - strategies/L{layer_idx}_*.json (for this layer only)

    Returns workspace_name for cache lookup.
    """
    workspace_path = Path(workspace_path).resolve()
    workspace_name = workspace_path.name

    # Get unique nodes in cluster
    nodes = [n for n in ray.nodes() if n["Alive"]]
    num_nodes = len(nodes)

    # Count files to broadcast
    num_strategies = len(list((workspace_path / "strategies").glob(f"L{layer_idx}_S*.json")))
    total_files = 2 + 1 + num_strategies  # configs + layer + strategies

    logger.info(f"   Broadcasting {total_files} files to {num_nodes} nodes...")

    # Create compressed archive on HEAD node
    archive_path = workspace_path / f"_broadcast_L{layer_idx}.tar.gz"

    # Build list of files to archive
    files_to_archive = []
    # Config files
    for config_name in ["hardware_config.json", "network_config.json"]:
        config_path = workspace_path / config_name
        if config_path.exists():
            files_to_archive.append(config_name)
    # Layer config
    layer_config = f"layers/L{layer_idx}.json"
    if (workspace_path / layer_config).exists():
        files_to_archive.append(layer_config)
    # Strategy files for this layer
    strategies_dir = workspace_path / "strategies"
    for strategy_file in strategies_dir.glob(f"L{layer_idx}_S*.json"):
        files_to_archive.append(f"strategies/{strategy_file.name}")

    # Create archive using pigz for parallel compression
    try:
        tar_proc = subprocess.Popen(
            ["tar", "-cf", "-", "-C", str(workspace_path)] + files_to_archive,
            stdout=subprocess.PIPE,
        )
        try:
            with open(archive_path, "wb") as f:
                subprocess.run(
                    ["pigz", "-1", "-p", "4"],
                    stdin=tar_proc.stdout,
                    stdout=f,
                    check=True,
                )
        finally:
            tar_proc.stdout.close()
            tar_proc.wait()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to Python tarfile
        import tarfile
        with tarfile.open(archive_path, "w:gz", compresslevel=1) as tar:
            for file_path in files_to_archive:
                tar.add(workspace_path / file_path, arcname=file_path)

    # Submit extraction task to each node
    setup_tasks = []
    for node in nodes:
        node_id = node["NodeID"]
        task = _setup_local_cache_on_node.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )
        ).remote(workspace_name, str(archive_path), layer_idx)
        setup_tasks.append(task)

    # Wait for all nodes to complete setup
    results = ray.get(setup_tasks)

    # Cleanup archive
    archive_path.unlink(missing_ok=True)

    total_cached = sum(r["files_cached"] for r in results)
    logger.info(f"   Cached {total_cached} files across {len(results)} nodes")

    return workspace_name


@ray.remote(num_cpus=1, scheduling_strategy="SPREAD")
def _archive_and_cleanup_on_node(workspace_name: str, layer_idx: int) -> dict:
    """
    Archive results to tar.gz and cleanup local directory on a single node.

    Creates: /mnt/workers/{hostname}/{workspace_name}_L{layer_idx}.tar.gz
    Head will read this archive and extract to main workspace.
    """
    import shutil
    import socket
    import subprocess

    hostname = socket.gethostname()
    node_id = ray.get_runtime_context().get_node_id()

    local_dir = WORKER_NFS_PATH / hostname / workspace_name
    archive_path = WORKER_NFS_PATH / hostname / f"{workspace_name}_L{layer_idx}.tar.gz"

    files_archived = 0

    # Create tar.gz archive using pigz (parallel gzip) for speed
    # Falls back to gzip if pigz not available
    if local_dir.exists():
        # Count files for reporting
        ray_results_dir = local_dir / "ray_results"
        simulations_dir = local_dir / "simulations"

        if ray_results_dir.exists():
            files_archived += len(list(ray_results_dir.glob("*.json")))
        if simulations_dir.exists():
            for sim_dir in simulations_dir.iterdir():
                if sim_dir.is_dir():
                    files_archived += len(list(sim_dir.glob("*")))

        # Use tar with pigz for parallel compression
        # pigz -1 = fast compression, -p4 = use 4 threads per node
        try:
            tar_proc = subprocess.Popen(
                ["tar", "-cf", "-", "-C", str(local_dir), "."],
                stdout=subprocess.PIPE,
            )
            try:
                with open(archive_path, "wb") as f:
                    subprocess.run(
                        ["pigz", "-1", "-p", "4"],
                        stdin=tar_proc.stdout,
                        stdout=f,
                        check=True,
                    )
            finally:
                tar_proc.stdout.close()
                tar_proc.wait()
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to Python tarfile if pigz fails
            import tarfile
            with tarfile.open(archive_path, "w:gz", compresslevel=1) as tar:
                tar.add(local_dir, arcname=".")

        # Remove the original directory after archiving
        shutil.rmtree(local_dir, ignore_errors=True)

    return {
        "node_id": node_id,
        "hostname": hostname,
        "archive_path": str(archive_path) if archive_path.exists() else None,
        "files_archived": files_archived,
    }


def _archive_results_on_all_nodes(workspace_name: str, layer_idx: int) -> list[dict]:
    """
    Archive results to tar.gz on all nodes after simulation completes.

    Each node creates: /mnt/workers/{hostname}/{workspace_name}_L{layer_idx}.tar.gz
    Returns list of archive info dicts for extraction.
    """
    nodes = [n for n in ray.nodes() if n["Alive"]]

    logger.info(f"   Archiving results on {len(nodes)} nodes...")

    # Submit archive task to each node
    archive_tasks = []
    for node in nodes:
        node_id = node["NodeID"]
        task = _archive_and_cleanup_on_node.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )
        ).remote(workspace_name, layer_idx)
        archive_tasks.append(task)

    # Wait for all nodes to complete archiving
    results = ray.get(archive_tasks)

    total_files = sum(r["files_archived"] for r in results)
    nodes_with_data = sum(1 for r in results if r["archive_path"])
    logger.info(f"   Archived {total_files} files from {nodes_with_data} nodes")

    return results


def _extract_archives_to_workspace(
    workspace_path: Path, archive_results: list[dict]
) -> int:
    """
    Extract tar.gz archives from worker nodes to main workspace.

    Head reads archives from /mnt/workers/{hostname}/{workspace_name}.tar.gz
    and extracts to workspace_path/ray_results/ and workspace_path/simulations/.

    Returns total number of files extracted.
    """
    import os
    import tarfile
    import time

    workspace_path = Path(workspace_path).resolve()
    workspace_name = workspace_path.name

    total_extracted = 0

    for result in archive_results:
        hostname = result.get("hostname")
        archive_path_str = result.get("archive_path")

        if not archive_path_str:
            continue

        # Head accesses via NFS mount at same path
        archive_path = Path(archive_path_str)

        # Force NFS cache invalidation by listing parent directory
        # This ensures HEAD sees files written by workers via NFS
        parent_dir = archive_path.parent
        if parent_dir.exists():
            try:
                os.listdir(parent_dir)  # Forces NFS attribute cache refresh
            except OSError:
                pass

        # Retry with small delay if file not found (NFS propagation delay)
        if not archive_path.exists():
            time.sleep(0.1)  # Brief delay for NFS sync
            if not archive_path.exists():
                logger.warning(f"   Archive not found: {archive_path}")
                continue

        # Extract to main workspace
        try:
            with warnings.catch_warnings():
                # Suppress Python 3.12+ tarfile security warning (CVE-2007-4559)
                # Our archives are self-created with safe relative paths only
                warnings.filterwarnings("ignore", message="The default behavior of tarfile extraction")
                with tarfile.open(archive_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        # Extract ray_results/* to workspace/ray_results/
                        # Extract simulations/* to workspace/simulations/
                        tar.extract(member, path=workspace_path)
                        total_extracted += 1

            # Remove archive after successful extraction
            archive_path.unlink(missing_ok=True)

            # Also cleanup the workspace directory on worker (should be empty)
            worker_workspace_dir = WORKER_NFS_PATH / hostname / workspace_name
            if worker_workspace_dir.exists():
                shutil.rmtree(worker_workspace_dir, ignore_errors=True)

        except Exception as e:
            logger.warning(f"   Failed to extract {archive_path}: {e}")

    return total_extracted


def _simulate_single_strategy_impl(
    workspace_path: Path,
    layer_idx: int,
    strategy_id: int,
    hardware_config: dict,
    systemc_dir: str,
    save_logs: str = LOG_POLICY_FAILED,
    generate_gantt: bool = False,
    generate_memory_layout: bool = False,
    use_local_cache: bool = False,
    workspace_name: str = None,
) -> dict:
    """
    Implementation of single strategy simulation (shared by batch and single).

    Args:
        use_local_cache: If True, read INPUT and write OUTPUT to /mnt/workers/{hostname}/
                         for efficient local I/O (bind-mounted locally, NFS-accessible from head)
        workspace_name: Workspace name for local cache lookup (required if use_local_cache=True)
    """
    import socket

    # Convert paths to absolute paths for Ray worker
    workspace_path = Path(workspace_path).resolve()
    systemc_dir = str(Path(systemc_dir).resolve())

    # Determine effective paths based on caching mode
    if use_local_cache and workspace_name:
        # All I/O in /mnt/workers/{hostname}/{workspace}/ (local on workers, NFS on head)
        hostname = socket.gethostname()
        local_dir = WORKER_NFS_PATH / hostname / workspace_name
        effective_input_path = local_dir
        effective_output_path = local_dir
    else:
        effective_input_path = workspace_path
        effective_output_path = workspace_path

    # Create per-worker simulator instance
    systemc_sim = SystemCRunner(
        systemc_dir=systemc_dir,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        save_logs=save_logs,
        generate_gantt=generate_gantt,
        generate_memory_layout=generate_memory_layout,
    )

    # Load layer metadata for result dict (from local cache if available)
    workspace = WorkspaceManager(effective_input_path)
    layer_config = workspace.load_layer_config(layer_idx)
    layer_name = str(layer_idx)

    try:
        # Run simulation: read from local cache, write to worker NFS
        result = systemc_sim.simulate_from_workspace(
            effective_input_path, layer_idx, strategy_id, hardware_config,
            output_path=effective_output_path
        )

        if not result.success:
            return _create_failure_result(
                strategy_id, layer_idx, layer_name, result, ERROR_TYPE_SIMULATION
            )

        strategy_config = workspace.load_strategy_config(layer_idx, strategy_id)
        tiling_config_dict = strategy_config["tiling_config"]

        # Save results to worker NFS (head can access via /mnt/workers/{hostname}/)
        return _create_success_result(
            strategy_id, layer_idx, layer_name, result, tiling_config_dict,
            effective_output_path
        )

    except Exception as e:
        return _create_exception_result(strategy_id, layer_idx, layer_name, e)


@ray.remote(num_cpus=1, scheduling_strategy="SPREAD")
def simulate_single_strategy(
    workspace_path: Path,
    layer_idx: int,
    strategy_id: int,
    hardware_config: dict,
    systemc_dir: str,
    save_logs: str = LOG_POLICY_FAILED,
    generate_gantt: bool = False,
    generate_memory_layout: bool = False,
    use_local_cache: bool = False,
    workspace_name: str = None,
) -> dict:
    """
    Simulate a single strategy in Ray worker process.

    Args:
        use_local_cache: If True, use local /tmp cache for I/O instead of NFS
        workspace_name: Workspace name for cache lookup (required if use_local_cache=True)
    """
    return _simulate_single_strategy_impl(
        workspace_path, layer_idx, strategy_id, hardware_config,
        systemc_dir, save_logs, generate_gantt, generate_memory_layout,
        use_local_cache, workspace_name
    )


def _create_failure_result(
    strategy_id: int,
    layer_idx: int,
    layer_name: str,
    result,
    error_type: str,
) -> dict:
    """
    Create failure result dict from SystemC simulation failure.

    Why separate function: Eliminates duplicate error dict construction in
    simulate_single_strategy() and sequential simulation paths.

    OPTIMIZATION: Removed stdout/stderr from return to minimize Ray transfer.
    Logs are already saved to NFS if save_logs policy is enabled.

    Args:
        strategy_id: Strategy ID
        layer_idx: Layer index
        layer_name: Layer name
        result: SystemCResult object with failure details
        error_type: Error type classification

    Returns:
        Dict with standardized failure information
    """
    return {
        "success": False,
        "strategy_id": strategy_id,
        "layer_idx": layer_idx,
        "layer_name": layer_name,
        "error": result.error_message,
        "error_type": error_type,
        "error_stage": "simulation",
        "return_code": result.return_code,
    }


def _create_exception_result(
    strategy_id: int, layer_idx: int, layer_name: str, exception: Exception
) -> dict:
    """
    Create exception result dict from Python exception.

    Why separate function: Consistent exception handling across all simulation paths.

    OPTIMIZATION: Removed stdout/stderr to minimize Ray transfer.

    Args:
        strategy_id: Strategy ID
        layer_idx: Layer index
        layer_name: Layer name
        exception: Python exception object

    Returns:
        Dict with exception details
    """
    return {
        "success": False,
        "strategy_id": strategy_id,
        "layer_idx": layer_idx,
        "layer_name": layer_name,
        "error": str(exception),
        "error_type": ERROR_TYPE_EXCEPTION,
        "error_stage": "setup",
        "return_code": -1,
    }


def _create_success_result(
    strategy_id: int,
    layer_idx: int,
    layer_name: str,
    result,
    tiling_config_dict: dict,
    workspace_path: Path = None,
) -> dict:
    """
    Create success result dict from successful SystemC simulation.

    Why separate function: Single source of truth for success dict structure.

    OPTIMIZATION: Save results to NFS and return only file path to minimize
    Ray Object Store usage and network transfer (reduces 829GB → few MB).

    Args:
        strategy_id: Strategy ID
        layer_idx: Layer index
        layer_name: Layer name
        result: SystemCResult object with simulation outputs
        tiling_config_dict: Tiling configuration dict from strategy file
        workspace_path: Path to workspace (if provided, saves to NFS)

    Returns:
        Dict with simulation results and tiling configuration
    """
    # Build result dict
    result_data = {
        "success": True,
        "strategy_id": strategy_id,
        "layer_idx": layer_idx,
        "layer_name": layer_name,
        "simulation_result": {
            "latency_ns": result.latency_ns,
            "area_mm2": result.area_mm2,
            "energy_nj": result.energy_nj,
            "ibuf_lines": result.ibuf_lines,
            "obuf_lines": result.obuf_lines,
        },
        "tiling_config": {
            "output_tile_p": tiling_config_dict["output_tile_p"],
            "output_tile_q": tiling_config_dict["output_tile_q"],
            "input_tile_h": tiling_config_dict["input_tile_h"],
            "input_tile_w": tiling_config_dict["input_tile_w"],
            "input_tile_p": tiling_config_dict["input_tile_p"],
            "input_tile_q": tiling_config_dict["input_tile_q"],
            "num_output_tiles_p": tiling_config_dict["num_output_tiles_p"],
            "num_output_tiles_q": tiling_config_dict["num_output_tiles_q"],
            "output_tile_count": tiling_config_dict["output_tile_count"],
            "input_tile_count": tiling_config_dict["input_tile_count"],
        },
        "return_code": result.return_code,
    }

    # If workspace_path provided, save to worker NFS and return lightweight reference
    if workspace_path:
        import socket

        # Save result to worker NFS (head accesses via /mnt/workers/{hostname}/)
        result_file = workspace_path / "ray_results" / f"L{layer_idx}_S{strategy_id}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        with open(result_file, "wb") as f:
            f.write(orjson.dumps(result_data))

        # Return metadata including hostname for result collection
        # Head will read from /mnt/workers/{hostname}/{workspace}/ray_results/
        node_id = ray.get_runtime_context().get_node_id()
        hostname = socket.gethostname()
        return {
            "success": True,
            "strategy_id": strategy_id,
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "node_id": node_id,
            "hostname": hostname,  # For result collection path
        }

    # Fallback: return full data (for non-Ray execution)
    return result_data


_sim_progress_bar_started = False

def _update_progress_bar(
    completed: int,
    total: int,
    success: int,
    failure: int,
    node_stats: dict = None,
):
    """
    Update and display progress bar with simulation statistics.

    Why separate function: Reusable progress display logic across parallel/sequential modes.
    Progress bar provides user feedback during long-running batch simulations.

    Args:
        completed: Number of completed simulations
        total: Total number of simulations
        success: Number of successful simulations
        failure: Number of failed simulations
        node_stats: Dict mapping node_ip -> task_count (optional, for per-node display)
    """
    import sys
    global _sim_progress_bar_started

    # Prevent division by zero
    if total == 0:
        percent, filled = 0, 0
    else:
        percent = (completed / total) * 100
        filled = int(PROGRESS_BAR_LENGTH * completed / total)
    bar = "▓" * filled + "░" * (PROGRESS_BAR_LENGTH - filled)

    # Build base progress string
    progress_str = f"   [{bar}] {completed}/{total} ({percent:.1f}%) | ok {success} | failed {failure}"

    # Add per-node stats if available (show last octet of IP for brevity)
    if node_stats:
        # Sort by count (descending) and format as "ip_last_octet:count"
        node_parts = []
        for node_ip, count in sorted(node_stats.items(), key=lambda x: -x[1]):
            # Extract last octet of IP for brevity (e.g., "192.0.2.13" -> "13")
            last_octet = node_ip.split(".")[-1] if "." in str(node_ip) else str(node_ip)[:8]
            node_parts.append(f"{last_octet}:{count}")
        progress_str += f" | {' '.join(node_parts)}"

    # Use ANSI escape codes for in-place updates
    # \033[A = move cursor up, \033[2K = clear line
    if _sim_progress_bar_started:
        sys.stderr.write("\033[A\033[2K")

    sys.stderr.write(f"{progress_str}\n")
    sys.stderr.flush()
    _sim_progress_bar_started = True

    # Reset when complete
    if completed == total:
        _sim_progress_bar_started = False


def _save_strategy_result(
    performance_db: StrategyDatabase,
    layer_name: str,
    result: dict,
    area_calc: AreaCalculator | None = None,
    workspace_path: Path | None = None,
) -> None:
    """
    Save successful strategy result to performance database.

    Why separate function: Eliminates code duplication between parallel and sequential
    execution paths. Single source of truth for database insertion logic.

    Args:
        performance_db: Performance database instance
        layer_name: Layer name identifier
        result: Result dict with simulation outputs and tiling config
        area_calc: AreaCalculator instance for computing actual area (optional)
        workspace_path: Path to workspace for updating results.json (optional)

    Raises:
        KeyError: If required keys are missing from result dict
    """
    # Validate required keys upfront for clear error messages
    required_keys = ["layer_idx", "strategy_id", "simulation_result", "tiling_config"]
    missing = [k for k in required_keys if k not in result]
    if missing:
        raise KeyError(f"Result dict missing required keys: {missing}")

    sim_result = result["simulation_result"]
    sim_required = ["latency_ns", "energy_nj", "area_mm2", "ibuf_lines", "obuf_lines"]
    sim_missing = [k for k in sim_required if k not in sim_result]
    if sim_missing:
        raise KeyError(f"simulation_result missing keys: {sim_missing}")

    # Extract layer_idx early (required for DB insertion)
    layer_idx = result["layer_idx"]

    # Calculate actual area from buffer lines (if AreaCalculator available)
    actual_area_mm2 = result["simulation_result"]["area_mm2"]  # Default: SystemC value (0.0)
    ibuf_area_mm2 = 0.0
    obuf_area_mm2 = 0.0
    cim_area_mm2 = 0.0

    if area_calc is not None:
        try:
            # Load CNN parameters from layer config for CIM area calculation
            cnn_params = None
            if workspace_path is not None:
                layer_config_path = workspace_path / "layers" / f"L{layer_idx}.json"
                if layer_config_path.exists():
                    with open(layer_config_path, "r") as f:
                        layer_config = orjson.loads(f.read())
                    cnn_params = layer_config.get("params")

            buffer_usage = {
                "ibuf_lines": result["simulation_result"]["ibuf_lines"],
                "obuf_lines": result["simulation_result"]["obuf_lines"],
            }
            area_breakdown = area_calc.calculate_area(buffer_usage, cnn_params)
            actual_area_mm2 = area_breakdown["total_area_mm2"]
            ibuf_area_mm2 = area_breakdown["ibuf_area_mm2"]
            obuf_area_mm2 = area_breakdown["obuf_area_mm2"]
            cim_area_mm2 = area_breakdown["cim_area_mm2"]

            # Update results.json file with calculated area breakdown
            if workspace_path is not None:
                strategy_id = result["strategy_id"]
                strategy_dir_name = f"L{layer_idx}_S{strategy_id}"
                results_json_path = (
                    workspace_path / "simulations" / strategy_dir_name / "results.json"
                )
                if results_json_path.exists():
                    try:
                        with open(results_json_path, "r") as f:
                            results_data = orjson.loads(f.read())
                        results_data["area_mm2"] = actual_area_mm2
                        results_data["ibuf_area_mm2"] = ibuf_area_mm2
                        results_data["obuf_area_mm2"] = obuf_area_mm2
                        results_data["cim_area_mm2"] = cim_area_mm2
                        with open(results_json_path, "wb") as f:
                            f.write(orjson.dumps(results_data, option=orjson.OPT_INDENT_2))
                    except Exception:
                        # Silently fall back if file update fails
                        pass
        except Exception:
            # Silently fall back to SystemC value (0.0)
            # Error logging handled by caller if needed
            pass

    performance_db.insert_strategy_result(
        layer_idx=layer_idx,
        strategy_id=result["strategy_id"],
        latency_ns=result["simulation_result"]["latency_ns"],
        area_mm2=actual_area_mm2,  # Use calculated area
        energy_nj=result["simulation_result"]["energy_nj"],
        ibuf_lines=result["simulation_result"]["ibuf_lines"],
        obuf_lines=result["simulation_result"]["obuf_lines"],
        input_tile_count=result["tiling_config"]["input_tile_count"],
        output_tile_count=result["tiling_config"]["output_tile_count"],
        ibuf_area_mm2=ibuf_area_mm2,
        obuf_area_mm2=obuf_area_mm2,
        cim_area_mm2=cim_area_mm2,
        tiling_config=orjson.dumps(
            {
                "output_tile_p": result["tiling_config"]["output_tile_p"],
                "output_tile_q": result["tiling_config"]["output_tile_q"],
                "input_tile_h": result["tiling_config"]["input_tile_h"],
                "input_tile_w": result["tiling_config"]["input_tile_w"],
                "input_tile_p": result["tiling_config"]["input_tile_p"],
                "input_tile_q": result["tiling_config"]["input_tile_q"],
                "num_output_tiles_p": result["tiling_config"]["num_output_tiles_p"],
                "num_output_tiles_q": result["tiling_config"]["num_output_tiles_q"],
                "output_tile_count": result["tiling_config"]["output_tile_count"],
            }
        ).decode(),
    )


def _create_simulation_log(
    workspace_path: Path,
    layer_idx: int,
    layer_name: str,
    result: dict,
):
    """
    Create detailed simulation log file for failed strategies.

    Why this exists: Provide comprehensive debugging information when simulations fail.
    Logs include error details, validation results, stdout/stderr for troubleshooting.

    Args:
        workspace_path: Path to workspace directory
        layer_idx: Layer index
        layer_name: Layer name
        result: Result dict with failure details
    """
    simulation_log_path = (
        workspace_path / "simulations" / f"L{layer_idx}_S{result['strategy_id']}_log.txt"
    )

    # Only create log if C++ didn't already create one
    if not simulation_log_path.exists():
        error_message = result.get("error") or "Unknown error"
        error_type = result.get("error_type", ERROR_TYPE_UNKNOWN)
        error_stage = result.get("error_stage", "unknown")

        with open(simulation_log_path, "w") as f:
            f.write("=== SystemC Simulation Log (FAILED) ===\n")
            f.write(f"Strategy ID: {result['strategy_id']}\n")
            f.write(f"Layer: {layer_name} (index {layer_idx})\n")
            f.write(f"Error Type: {error_type}\n")
            f.write(f"Error Stage: {error_stage}\n")
            f.write(f"Return Code: {result.get('return_code', -1)}\n")
            f.write("\n=== ERROR DETAILS ===\n")
            f.write(error_message)

            if result.get("validation_errors"):
                f.write("\n\n=== VALIDATION ERRORS ===\n")
                for i, err in enumerate(result["validation_errors"], 1):
                    f.write(f"{i}. {err}\n")

            f.write("\n\n=== STDOUT ===\n")
            stdout = result.get("stdout")
            f.write(stdout.decode() if isinstance(stdout, bytes) else (stdout or "(empty)"))
            f.write("\n\n=== STDERR ===\n")
            stderr = result.get("stderr")
            f.write(stderr.decode() if isinstance(stderr, bytes) else (stderr or "(empty)"))


def _create_error_summary(
    workspace_path: Path,
    layer_idx: int,
    result: dict,
):
    """
    Create concise error summary file for quick debugging.

    Why separate from full log: Provides quick overview without requiring
    full log file parsing. Useful for batch failure analysis.

    Args:
        workspace_path: Path to workspace directory
        layer_idx: Layer index
        result: Result dict with failure details
    """
    error_summary_path = (
        workspace_path / "simulations" / f"L{layer_idx}_S{result['strategy_id']}_error.txt"
    )

    error_message = result.get("error") or "Unknown error"
    error_type = result.get("error_type", ERROR_TYPE_UNKNOWN)
    error_stage = result.get("error_stage", "unknown")

    with open(error_summary_path, "w") as f:
        f.write(f"Strategy: L{layer_idx}_S{result['strategy_id']}\n")
        f.write("Status: FAILED\n")
        f.write(f"Error Type: {error_type}\n")
        f.write(f"Error Stage: {error_stage}\n")
        f.write("\nError Message:\n")
        f.write(error_message)

        if result.get("validation_errors"):
            f.write("\n\nValidation Errors:\n")
            for i, err in enumerate(result["validation_errors"], 1):
                f.write(f"  {i}. {err}\n")


def _handle_strategy_failure(
    workspace_path: Path,
    layer_idx: int,
    layer_name: str,
    result: dict,
    workspace: WorkspaceManager,
    verbose: bool,
):
    """
    Handle failed strategy simulation by creating logs and saving error details.

    Why separate function: Centralizes all failure handling logic (logging, error files,
    workspace updates) in one place. Eliminates duplication between parallel and
    sequential execution paths.

    Args:
        workspace_path: Path to workspace directory
        layer_idx: Layer index
        layer_name: Layer name
        result: Result dict with failure details
        workspace: WorkspaceManager instance
        verbose: Whether to print verbose error messages
    """
    # Create detailed simulation log
    _create_simulation_log(workspace_path, layer_idx, layer_name, result)

    # Create concise error summary
    _create_error_summary(workspace_path, layer_idx, result)

    # Update workspace with failure (backward compatibility)
    error_message = result.get("error") or "Unknown error"
    workspace.save_failed_simulation(layer_idx, result["strategy_id"], error_message)

    # Print error in verbose mode
    if verbose:
        logger.error(f"     Strategy {result['strategy_id']} failed: {result['error']}")


def _run_parallel_simulations(
    workspace: WorkspaceManager,
    layer_idx: int,
    layer_name: str,
    strategy_ids: list,
    hardware_config: dict,
    performance_db: StrategyDatabase,
    systemc_dir: str,
    max_cpus: int,
    save_logs: str,
    generate_gantt: bool,
    generate_memory_layout: bool,
    verbose: bool,
    area_calc: AreaCalculator | None = None,
    use_local_cache: bool | None = None,  # None = auto-detect from env
) -> tuple[int, int, list, dict]:
    """
    Run simulations in parallel using Ray.

    Why separate function: Parallel execution has different control flow than sequential.
    Separating keeps simulate_layer_strategies() readable and maintainable.

    Args:
        workspace: WorkspaceManager instance
        layer_idx: Layer index
        layer_name: Layer name
        strategy_ids: List of strategy IDs to simulate
        hardware_config: Hardware configuration dict
        performance_db: Performance database instance
        systemc_dir: Path to SystemC directory
        max_cpus: Maximum number of CPUs to use (limits concurrent tasks)
        save_logs: Log saving policy
        generate_gantt: Whether to generate Gantt charts
        generate_memory_layout: Whether to generate memory layout visualizations
        verbose: Whether to print verbose output
        area_calc: AreaCalculator instance for computing actual area (optional)
        use_local_cache: If True, use local /tmp cache for I/O instead of NFS

    Returns:
        Tuple of (success_count, failure_count, failed_strategy_ids, node_stats)
    """
    import os

    workspace_path = workspace.workspace_path
    success_count = 0
    failure_count = 0

    # Initialize Ray if not already initialized
    _init_ray(num_cpus=max_cpus)

    total = len(strategy_ids)
    logger.info(
        f"   Layer L{layer_idx} ({layer_name}): Simulating {total} strategies with {max_cpus} CPUs..."
    )

    # Convert to absolute paths before submitting to Ray
    # Why: Ray workers may have different CWD, so we need absolute paths
    workspace_path_abs = workspace_path.resolve()
    systemc_dir_abs = str(Path(systemc_dir).resolve())

    # Determine caching mode from env var if not explicitly set
    # EFSIM_NO_CACHE=1 disables local caching (uses NFS direct read)
    if use_local_cache is None:
        use_local_cache = os.environ.get("EFSIM_NO_CACHE", "0") != "1"

    # Local caching: broadcast input files to all nodes
    workspace_name = None
    if use_local_cache:
        workspace_name = _broadcast_input_files(workspace_path_abs, layer_idx)
    else:
        logger.info("   Using NFS direct read (no local cache)")

    # Submit tasks to Ray with limited concurrency
    # When max_cpus is specified, only submit num_workers tasks at a time
    _task_registry.clear()

    completed = 0
    success_count = 0
    failure_count = 0
    failed_strategy_ids = []  # Track failed strategies for potential retry
    node_stats = {}  # Track tasks per node hostname for real-time display

    # Build node_id -> IP mapping (fallback if hostname not in result)
    try:
        nodes_info = {node["NodeID"]: node.get("NodeManagerAddress", "unknown")
                      for node in ray.nodes() if node["Alive"]}
    except Exception:
        nodes_info = {}

    # Submit ALL tasks upfront - Ray handles scheduling with SPREAD strategy
    pending = []
    for strategy_id in strategy_ids:
        future = simulate_single_strategy.remote(
            workspace_path_abs,
            layer_idx,
            strategy_id,
            hardware_config,
            systemc_dir_abs,
            save_logs,
            generate_gantt,
            generate_memory_layout,
            use_local_cache,
            workspace_name,
        )
        pending.append(future)
        _task_registry.add(future)

    # Show initial progress bar immediately
    _update_progress_bar(completed, total, success_count, failure_count, node_stats)
    last_update_percent = 0  # Track last 10% milestone for update

    # Process results as they complete
    while pending:
        # Wait for any task to complete with timeout
        done, pending = ray.wait(pending, num_returns=1, timeout=2.0)

        # If no tasks completed within timeout, continue without updating
        # (progress bar only updates at 10% milestones)
        if not done and pending:
            continue

        for ref in done:
            try:
                result = ray.get(ref)
                completed += 1

                # Track which node executed this task (use hostname for reliable counting)
                if "hostname" in result:
                    hostname = result["hostname"]
                    node_stats[hostname] = node_stats.get(hostname, 0) + 1
                elif "node_id" in result:
                    # Fallback to node_id if hostname not available
                    node_id = result["node_id"]
                    node_ip = nodes_info.get(node_id, node_id[:8] if node_id else "unknown")
                    node_stats[node_ip] = node_stats.get(node_ip, 0) + 1

            except ray.exceptions.RayTaskError as e:
                # Remote function raised an exception
                completed += 1
                failure_count += 1
                logger.warning(f" Task failed (remote): {e.cause}")
            except ray.exceptions.RayError as e:
                # Ray system error (object lost, node died, etc.)
                completed += 1
                failure_count += 1
                logger.warning(f" Task failed (Ray): {type(e).__name__}: {e}")

            # Update progress bar every 10% milestone or at completion
            current_percent = (completed * 100) // total if total > 0 else 0
            current_milestone = (current_percent // 10) * 10
            if current_milestone > last_update_percent or completed >= total:
                _update_progress_bar(completed, total, success_count, failure_count, node_stats)
                last_update_percent = current_milestone

    print()  # New line after progress bar

    # Collect results from all nodes to main workspace
    if use_local_cache and workspace_name:
        # Step 1: Archive results on each node (creates tar.gz)
        archive_results = _archive_results_on_all_nodes(workspace_name, layer_idx)

        # Step 2: Extract archives to main workspace
        logger.info(f"   Extracting results to workspace...")
        extracted = _extract_archives_to_workspace(workspace_path, archive_results)
        logger.info(f"   Extracted {extracted} files to workspace")

    # Process results from main workspace (now unified after extraction)
    logger.info(f"   Processing results from {total} simulations...")
    for strategy_id in strategy_ids:
        # All results are now in main workspace after extraction
        result_file = workspace_path / "ray_results" / f"L{layer_idx}_S{strategy_id}.json"

        if result_file.exists():
            with open(result_file, "rb") as f:
                result = orjson.loads(f.read())

            if result.get("success"):
                success_count += 1
                _save_strategy_result(performance_db, layer_name, result, area_calc, workspace_path)
            else:
                failure_count += 1
                failed_strategy_ids.append(result["strategy_id"])
                _handle_strategy_failure(
                    workspace_path, layer_idx, layer_name, result, workspace, verbose
                )
        else:
            # Result file missing - count as failure
            failure_count += 1
            failed_strategy_ids.append(strategy_id)

    # node_stats is already built in real-time during the loop (IP -> count)

    # Clear global task tracker
    _task_registry.clear()

    return success_count, failure_count, failed_strategy_ids, node_stats


def _run_sequential_simulations(
    workspace: WorkspaceManager,
    layer_idx: int,
    layer_name: str,
    layer_config: dict,
    strategy_ids: list,
    hardware_config: dict,
    performance_db: StrategyDatabase,
    systemc_dir: str,
    save_logs: str,
    generate_gantt: bool,
    generate_memory_layout: bool,
    verbose: bool,
    area_calc: AreaCalculator | None = None,
) -> tuple[int, int, list, dict | None]:
    """
    Run simulations sequentially (single-threaded).

    Why separate function: Sequential execution has different setup requirements
    (CNNLayerParams extraction, single SystemCRunner instance). Separating keeps
    simulate_layer_strategies() clean and focused.

    Why still needed: Sequential mode is useful for debugging, resource-constrained
    environments, and reproducibility testing.

    Args:
        workspace: WorkspaceManager instance
        layer_idx: Layer index
        layer_name: Layer name
        layer_config: Layer configuration dict
        strategy_ids: List of strategy IDs to simulate
        hardware_config: Hardware configuration dict
        performance_db: Performance database instance
        systemc_dir: Path to SystemC directory
        save_logs: Log saving policy
        generate_gantt: Whether to generate Gantt charts
        generate_memory_layout: Whether to generate memory layout visualizations
        verbose: Whether to print verbose output
        area_calc: AreaCalculator instance for computing actual area (optional)
    Returns:
        Tuple of (success_count, failure_count)
    """
    success_count = 0
    failure_count = 0

    # Create single SystemCRunner instance for all simulations
    systemc_sim = SystemCRunner(
        systemc_dir=systemc_dir,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        save_logs=save_logs,
        generate_gantt=generate_gantt,
    )

    # Extract CNN parameters from layer config
    params = layer_config.get("params", layer_config)
    cnn_params = CNNLayerParams(
        H=params["H"],
        W=params["W"],
        C=params["C"],
        R=params["R"],
        S=params["S"],
        M=params["M"],
        stride=params.get("stride", 1),
        batch_size=params.get("batch_size", 1),
        input_bitwidth=params.get("input_bitwidth", 8),
        output_bitwidth=params.get("output_bitwidth", 8),
        pool_height=params.get("pool_height", 1),
        pool_width=params.get("pool_width", 1),
    )

    for strategy_id in strategy_ids:
        # Load strategy configuration
        strategy_config = workspace.load_strategy_config(layer_idx, strategy_id)
        tiling_config_dict = strategy_config["tiling_config"]

        # Create TilingConfig object
        tiling_config = TilingConfig(
            output_tile_p=tiling_config_dict["output_tile_p"],
            output_tile_q=tiling_config_dict["output_tile_q"],
            input_tile_h=tiling_config_dict["input_tile_h"],
            input_tile_w=tiling_config_dict["input_tile_w"],
            input_tile_p=tiling_config_dict["input_tile_p"],
            input_tile_q=tiling_config_dict["input_tile_q"],
            num_output_tiles_p=tiling_config_dict["num_output_tiles_p"],
            num_output_tiles_q=tiling_config_dict["num_output_tiles_q"],
            num_input_tiles_p=tiling_config_dict["num_input_tiles_p"],
            num_input_tiles_q=tiling_config_dict["num_input_tiles_q"],
            output_tile_count=tiling_config_dict["output_tile_count"],
            input_tile_count=tiling_config_dict["input_tile_count"],
            strategy_id=strategy_id,
            description=f"Strategy {strategy_id} for layer {layer_name}",
            case_type=tiling_config_dict.get("case_type", 2),
            # Phase 3: Total operation counts (pre-calculated in Python)
            total_loads=tiling_config_dict["total_loads"],
            total_ibuf_reads=tiling_config_dict["total_ibuf_reads"],
            total_cim_computes=tiling_config_dict["total_cim_computes"],
            total_obuf_writes=tiling_config_dict["total_obuf_writes"],
            total_stores=tiling_config_dict["total_stores"],
        )

        # Determine strategy file path (support descriptive naming: L0_S0_out2x2_in5x5.json)
        strategies_dir = workspace.workspace_path / "strategies"
        pattern = f"L{layer_idx}_S{strategy_id}_*.json"
        matches = list(strategies_dir.glob(pattern))

        if not matches:
            failure_count += 1
            if verbose:
                logger.warning(f"      Strategy file not found: {pattern}")
            continue
        strategy_file = matches[0]

        # Run SystemC simulation with 3 separate file paths
        # C++ reads files directly - no merging needed!
        result = systemc_sim.simulate(
            workspace_path=workspace.workspace_path,
            strategy_path=strategy_file,
            network_path=workspace.workspace_path / "network_config.json",
            hardware_path=workspace.workspace_path / "hardware_config.json",
            layer_idx=layer_idx,
            strategy_id=strategy_id,
            log_dir=None,
        )

        if not result.success:
            failure_count += 1
            if verbose:
                logger.warning(f"     Strategy {strategy_id} simulation failed: {result.error_message}")
            continue

        # Validate simulation results
        validation_config = {
            "cnn_layer": {
                "batch_size": cnn_params.batch_size,
                "pool_height": cnn_params.pool_height,
                "pool_width": cnn_params.pool_width,
            },
            "tiling_config": {
                "output_tile_p": tiling_config.output_tile_p,
                "output_tile_q": tiling_config.output_tile_q,
                "input_tile_h": tiling_config.input_tile_h,
                "input_tile_w": tiling_config.input_tile_w,
                "output_tile_count": tiling_config.output_tile_count,
                "input_tile_count": tiling_config.input_tile_count,
                "case_type": tiling_config_dict.get("case_type", 2),
            },
        }

        validator = SimulationValidator(validation_config, result)
        validation_result = validator.validate_all(level="basic")

        if not validation_result.is_valid():
            failure_count += 1
            if verbose:
                logger.error(f"      Strategy {strategy_id} validation failed:")
                for error in validation_result.errors[:3]:
                    logger.error(f"        - {error}")
            continue

        # Save simulation results to database
        sequential_result = _create_success_result(
            strategy_id, layer_idx, layer_name, result, tiling_config_dict
        )
        _save_strategy_result(
            performance_db, layer_name, sequential_result, area_calc, workspace.workspace_path
        )
        success_count += 1

        # Memory layout PDF is now generated inside SystemCRunner._read_simulation_logs()

    # Sequential mode doesn't track failed_ids or node stats (for simplicity)
    return success_count, failure_count, [], None


def simulate_layer_strategies(
    workspace: WorkspaceManager,
    layer_idx: int,
    hardware_config: dict,
    performance_db: StrategyDatabase,
    systemc_dir: str,
    verbose: bool = False,
    max_cpus: int | None = None,
    save_logs: str = LOG_POLICY_FAILED,
    generate_gantt: bool = False,
    generate_memory_layout: bool = False,
    resume: bool = False,
    retry_failed: bool = False,
    area_calc: AreaCalculator | None = None,
) -> tuple[int, int]:
    """
    Simulate all strategies for a single layer.

    Why this orchestration: Coordinates parallel/sequential execution, resume logic,
    and result collection. Delegates actual simulation work to helper functions.

    Args:
        workspace: WorkspaceManager instance
        layer_idx: Layer index (0-based)
        hardware_config: Hardware configuration dict
        performance_db: Performance database instance
        systemc_dir: Path to SystemC directory
        verbose: Whether to print verbose output
        max_cpus: Maximum number of CPUs to use (None = use all, 1 = sequential)
        save_logs: Log saving policy (none/failed/all)
        generate_gantt: Whether to generate Gantt charts
        generate_memory_layout: Whether to generate memory layout visualizations
        resume: Whether to skip already completed strategies
        retry_failed: Whether to retry failed strategies after initial simulation
        area_calc: AreaCalculator instance for computing actual area (optional)

    Returns:
        Tuple of (success_count, failure_count)
    """
    # Load layer configuration
    layer_config = workspace.load_layer_config(layer_idx)
    layer_name = str(layer_config["layer_idx"])

    # Get all strategies for this layer
    strategy_ids = workspace.list_strategies(layer_idx)

    if not strategy_ids:
        logger.info(f"    No strategies found for layer L{layer_idx} ({layer_name})")
        return 0, 0

    # Resume mode: filter out already completed strategies
    # Why resume mode: Allows recovering from interrupted simulations without
    # re-running expensive SystemC simulations that already completed successfully.
    if resume:
        completed_strategies = performance_db.get_completed_strategies(layer_name)
        original_count = len(strategy_ids)
        strategy_ids = [sid for sid in strategy_ids if sid not in completed_strategies]
        skipped_count = original_count - len(strategy_ids)

        if skipped_count > 0:
            logger.info(f"   Resume mode: Skipping {skipped_count} already completed strategies")

        if not strategy_ids:
            logger.info(
                f"   All {original_count} strategies already completed for layer {layer_name}"
            )
            return original_count, 0

    # Run simulations (parallel or sequential)
    if max_cpus > 1:
        success_count, failure_count, failed_ids, node_stats = _run_parallel_simulations(
            workspace,
            layer_idx,
            layer_name,
            strategy_ids,
            hardware_config,
            performance_db,
            systemc_dir,
            max_cpus,
            save_logs,
            generate_gantt,
            generate_memory_layout,
            verbose,
            area_calc,
        )
    else:
        success_count, failure_count, failed_ids, node_stats = _run_sequential_simulations(
            workspace,
            layer_idx,
            layer_name,
            layer_config,
            strategy_ids,
            hardware_config,
            performance_db,
            systemc_dir,
            save_logs,
            generate_gantt,
            generate_memory_layout,
            verbose,
            area_calc,
        )

    # Retry failed strategies if requested
    if retry_failed and failed_ids:
        if verbose:
            logger.info(
                f"   Retrying {len(failed_ids)} failed strategies for layer L{layer_idx} ({layer_name})..."
            )

        # Retry failed strategies (always use same execution mode as initial run)
        if max_cpus > 1:
            retry_success, retry_failure, _, _ = _run_parallel_simulations(
                workspace,
                layer_idx,
                layer_name,
                failed_ids,
                hardware_config,
                performance_db,
                systemc_dir,
                max_cpus,
                save_logs,
                generate_gantt,
                generate_memory_layout,
                verbose,
                area_calc,
            )
        else:
            retry_success, retry_failure, _, _ = _run_sequential_simulations(
                workspace,
                layer_idx,
                layer_name,
                layer_config,
                failed_ids,
                hardware_config,
                performance_db,
                systemc_dir,
                save_logs,
                generate_gantt,
                generate_memory_layout,
                verbose,
                area_calc,
            )

        # Update counts
        success_count += retry_success
        failure_count = failure_count - retry_success  # Some failures became successes

        if verbose:
            logger.info(
                f"   Retry complete: {retry_success} recovered, {retry_failure} still failed"
            )

    # Print summary
    if verbose:
        if failure_count > 0:
            logger.info(
                f"   Layer L{layer_idx} ({layer_name}): {success_count} simulations succeeded, {failure_count} failed"
            )
        else:
            logger.info(
                f"   Layer L{layer_idx} ({layer_name}): {success_count} simulations completed"
            )

    # Print node statistics if available (always, not just in verbose mode)
    if node_stats:
        logger.info(f"   Node statistics:")
        for node_ip, count in sorted(node_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / (success_count + failure_count)) * 100
            logger.info(f"      {node_ip}: {count} tasks ({percentage:.1f}%)")

    return success_count, failure_count


def save_simulation_metadata(
    workspace_path: Path,
    stats: dict,
):
    """
    Save simulation metadata and human-readable summary.

    Why this exists: Provides comprehensive simulation statistics for analysis,
    troubleshooting, and workflow tracking. Metadata enables Phase 2 optimization
    to understand simulation characteristics.

    Args:
        workspace_path: Path to workspace directory
        stats: Statistics dict with simulation results
    """
    metadata = {
        "total_strategies": stats["total_strategies"],
        "total_layers": stats["total_layers"],
        "successful_simulations": stats["successful_simulations"],
        "failed_simulations": stats["failed_simulations"],
        "simulation_time_s": stats["elapsed_time"],
        "timestamp": datetime.now().isoformat(),
        "tool_version": TOOL_VERSION,
        "phase": SIMULATION_PHASE_NAME,
    }

    # Update workflow metadata with simulation phase
    workflow_metadata_path = workspace_path / "workflow_metadata.json"

    # Load existing workflow metadata or create minimal fallback
    if workflow_metadata_path.exists():
        with open(workflow_metadata_path) as f:
            workflow_metadata = orjson.loads(f.read())
    else:
        # Fallback: create minimal metadata from workspace.json
        workspace_json_path = workspace_path / "workspace.json"
        with open(workspace_json_path) as f:
            workspace_info = orjson.loads(f.read())

        workflow_metadata = {
            "workspace_name": workspace_info.get("name", "Unknown"),
            "network_name": workspace_info.get("network_name", "Unknown"),
            "network_config": workspace_info.get("network", ""),
            "hardware_config": workspace_info.get("hardware", ""),
            "phases": {},
        }

    # Add simulation phase
    workflow_metadata["phases"][SIMULATION_PHASE_NAME] = metadata

    with open(workflow_metadata_path, "w") as f:
        f.write(orjson.dumps(workflow_metadata, option=orjson.OPT_INDENT_2).decode())

    # Create human-readable summary file
    _create_simulation_summary(workspace_path, stats)


def _create_simulation_summary(workspace_path: Path, stats: dict):
    """
    Create human-readable simulation summary file.

    Why separate function: Summary file generation is complex enough to deserve
    its own function. Provides clear, actionable information to users.

    Args:
        workspace_path: Path to workspace directory
        stats: Statistics dict with simulation results
    """
    summary_path = workspace_path / "SIMULATION_SUMMARY.txt"
    success_rate = (
        stats["successful_simulations"] / stats["total_strategies"] * 100
        if stats["total_strategies"] > 0
        else 0
    )
    throughput = (
        stats["successful_simulations"] / stats["elapsed_time"] if stats["elapsed_time"] > 0 else 0
    )

    status_icon = "" if stats["failed_simulations"] == 0 else ""
    status_text = "SUCCESS" if stats["failed_simulations"] == 0 else "COMPLETED WITH ERRORS"

    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"{status_icon} SIMULATION SUMMARY - {status_text}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Workspace:       {workspace_path.name}\n")
        f.write(f"Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tool Version:    {TOOL_VERSION}\n\n")

        f.write("RESULTS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total Layers:              {stats['total_layers']}\n")
        f.write(f"  Total Strategies:          {stats['total_strategies']}\n")
        f.write(f"  Successful Simulations: {stats['successful_simulations']}\n")
        if stats["failed_simulations"] > 0:
            f.write(f"  Failed Simulations:     {stats['failed_simulations']}\n")
        f.write(f"  Success Rate:              {success_rate:.1f}%\n\n")

        f.write("PERFORMANCE:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Elapsed Time:              {stats['elapsed_time']:.2f}s\n")
        f.write(f"  Throughput:                {throughput:.2f} simulations/sec\n\n")

        f.write("OUTPUT FILES:\n")
        f.write("-" * 70 + "\n")
        f.write("  Database:                  strategies.db\n")
        f.write("  CSV Export:                strategies.csv\n")
        f.write("  Simulation Logs:           simulations/\n")
        f.write("  Metadata:                  simulation_metadata.json\n\n")

        if stats["failed_simulations"] == 0:
            f.write("NEXT STEPS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  1. Optimize:               ./optimize.py {workspace_path} --min-latency\n")
            f.write(
                f"  2. Visualize:              ./visualize_characterization.py {workspace_path}\n"
            )
        else:
            f.write("TROUBLESHOOTING:\n")
            f.write("-" * 70 + "\n")
            f.write("  Check simulation logs in:  simulations/\n")
            f.write(f"  Re-run with verbose:       ./efsim simulate {workspace_path.name} -v\n")

        f.write("\n" + "=" * 70 + "\n")


def main():  # noqa: C901
    """
    Main entry point for simulation tool.

    Why this orchestration: Handles CLI parsing, workspace setup, database initialization,
    and coordination of layer-wise simulation. Provides user-friendly progress reporting
    and next-step guidance.
    """
    parser = argparse.ArgumentParser(
        description="Step 2: Simulate pre-generated strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./efsim simulate my_experiment                       # Use all available CPUs
  ./efsim simulate my_experiment --verbose
  ./efsim simulate my_experiment --max-cpus 90         # Use only 90 CPUs (out of 180)
  ./efsim simulate my_experiment --max-cpus 1          # Sequential execution (1 CPU)
  ./efsim simulate my_experiment --save-logs none      # Delete all logs (fastest)
  ./efsim simulate my_experiment --save-logs all       # Keep all logs (debug)
  ./efsim simulate my_experiment --generate-gantt      # Generate Gantt PDFs
  ./efsim simulate my_experiment --resume              # Resume interrupted simulation

Note: Workspace must be initialized and strategies generated first.
      All strategies are automatically validated using SystemC simulation.
      Invalid results are skipped and reported in verbose mode.
      Default: Uses all available CPU cores for parallel execution.
      Default: --save-logs failed (keep only failed simulation logs).
      Resume: Skips already completed strategies (useful for interrupted runs).
        """,
    )
    parser.add_argument("workspace", help="Workspace name (e.g., my_experiment)")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output with validation details"
    )
    # Default to absolute path based on project root (resolved for Ray workers)
    default_systemc_dir = str((Path(__file__).parent.parent / "src" / "systemc").resolve())
    parser.add_argument(
        "--systemc-dir",
        default=default_systemc_dir,
        help=f"Path to SystemC directory (default: {default_systemc_dir})",
    )
    parser.add_argument(
        "--save-logs",
        choices=[LOG_POLICY_NONE, LOG_POLICY_FAILED, LOG_POLICY_ALL],
        default=DEFAULT_LOG_POLICY,
        help=f"Log saving policy: '{LOG_POLICY_NONE}' (no logs), '{LOG_POLICY_FAILED}' (failed only), '{LOG_POLICY_ALL}' (default: {DEFAULT_LOG_POLICY})",
    )
    parser.add_argument(
        "--generate-gantt",
        action="store_true",
        help="Generate Gantt chart PDFs (slower, ~85KB per strategy, default: disabled)",
    )
    parser.add_argument(
        "--generate-memory-layout",
        action="store_true",
        help="Generate memory layout visualization PDFs (slower, ~45KB per strategy, default: disabled)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume simulation by skipping already completed strategies (default: disabled)",
    )
    parser.add_argument(
        "--skip-energy",
        action="store_true",
        help="Skip automatic energy calculation (default: energy calculated)",
    )
    parser.add_argument(
        "--max-cpus",
        type=int,
        default=None,
        help="Limit Ray cluster CPU usage (e.g., 90 to use 90 CPUs out of 180). If not specified, uses all available CPUs.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed strategies after initial simulation completes (default: disabled)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress 'Next steps' output (for benchmark automation)",
    )

    args = parser.parse_args()

    # Determine max CPUs to use
    # If max_cpus is specified, use that value.
    # Otherwise, will be auto-detected from Ray cluster after initialization.
    # Note: We defer CPU detection until after Ray is initialized to get cluster-wide count.

    # Use absolute path for workspace to ensure Ray workers can find it
    project_root = Path(__file__).parent.parent.resolve()
    workspace_path = (project_root / "workspaces" / args.workspace).resolve()

    if not workspace_path.exists():
        logger.error(f"Workspace not found: {args.workspace}")
        logger.error(f"   Run: ./efsim generate {args.workspace}")
        sys.exit(1)

    # Concise header - show workspace name
    # Note: CPU count will be shown later (auto-detected from Ray cluster if not specified)
    if args.max_cpus is not None:
        workers_info = f"{args.max_cpus} CPUs" if args.max_cpus > 1 else "1 CPU"
        logger.info(f"{args.workspace} | {workers_info}")
    else:
        logger.info(f"{args.workspace} | Auto-detecting CPUs from Ray cluster...")
    logger.info("=" * 70)

    # Load workspace
    workspace = WorkspaceManager(workspace_path)

    # Load configurations from workspace
    network_config_path = workspace_path / "network_config.json"
    hardware_config_path = workspace_path / "hardware_config.json"

    with open(network_config_path) as f:
        network_config = orjson.loads(f.read())

    with open(hardware_config_path) as f:
        hardware_config = orjson.loads(f.read())

    # Initialize AreaCalculator for automatic area calculation
    try:
        # AreaCalculator expects full config with "hardware" section
        # It will extract hardware_config["hardware"] internally
        area_calc = AreaCalculator(hardware_config)
        logger.debug("AreaCalculator initialized")  # Debug level - hidden by default
    except Exception as e:
        logger.error(f"AreaCalculator initialization failed: {e}")
        logger.error("   All area values will be 0.0")
        import traceback

        logger.debug(traceback.format_exc())
        area_calc = None

    # Initialize database
    db_path = workspace.get_database_path()
    performance_db = StrategyDatabase(str(db_path))

    # Insert layer information into database
    # Why this step: Database needs layer metadata for query optimization and reporting
    for layer_idx, layer in enumerate(network_config["layers"]):
        params = layer.get("params", layer)

        performance_db.insert_layer(
            layer_idx=layer_idx,
            layer_type=layer.get("type", "conv2d"),
            input_shape=(params["H"], params["W"], params["C"]),
            output_shape=(params.get("P", params["H"]), params.get("Q", params["W"]), params["M"]),
            kernel_size=(params["R"], params["S"]),
            stride=params.get("stride", 1),
            network_name=network_config.get("network_name", "unknown"),
        )

    # Clean up previous simulation results (skip if resuming)
    # Why cleanup: Prevents stale results from previous runs causing confusion
    if not args.resume:
        simulations_dir = workspace_path / "simulations"
        if simulations_dir.exists():
            all_items = list(simulations_dir.glob("*"))
            if all_items:
                logger.info(f"Cleaning up simulations folder ({len(all_items)} items)...")
                for item in all_items:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
    else:
        logger.info("Resume mode enabled - preserving existing simulation results")

    # Initialize Ray cluster and detect available CPUs if not specified
    if args.max_cpus is None:
        _init_ray()
        cluster_resources = ray.cluster_resources()
        args.max_cpus = int(cluster_resources.get("CPU", multiprocessing.cpu_count()))
        logger.info(f"Detected {args.max_cpus} CPUs from Ray cluster")

    # Simulate all layers
    logger.info(" Running SystemC simulation with validation...")
    start_time = time.time()

    total_success = 0
    total_failure = 0

    # Get number of layers from network config
    num_layers = len(network_config["layers"])

    # Enable batch mode for DB inserts (avoids per-row commit overhead)
    # Without this: 9000 commits × 8ms = 72 seconds
    # With batch: 1 commit × 8ms = 8ms
    performance_db.begin_batch()

    try:
        for layer_idx in range(num_layers):
            success, failure = simulate_layer_strategies(
                workspace,
                layer_idx,
                hardware_config,
                performance_db,
                args.systemc_dir,
                verbose=args.verbose,
                max_cpus=args.max_cpus,
                save_logs=args.save_logs,
                generate_gantt=args.generate_gantt,
                generate_memory_layout=args.generate_memory_layout,
                resume=args.resume,
                retry_failed=args.retry_failed,
                area_calc=area_calc,
            )
            total_success += success
            total_failure += failure
    finally:
        # Commit all DB inserts at once
        performance_db.end_batch()

    elapsed_time = time.time() - start_time

    # Export to CSV
    csv_path = workspace_path / "strategies.csv"
    performance_db.export_to_csv(str(csv_path))

    # Save metadata
    stats = {
        "total_strategies": total_success + total_failure,
        "successful_simulations": total_success,
        "failed_simulations": total_failure,
        "total_layers": num_layers,
        "elapsed_time": elapsed_time,
    }
    save_simulation_metadata(workspace_path, stats)

    # Print summary
    logger.info("=" * 70)
    logger.info("Simulation Complete!")
    logger.info(f"   Layers evaluated: {stats['total_layers']}")
    logger.info(f"   Successful simulations: {total_success}")
    if total_failure > 0:
        logger.error(f"   Failed simulations: {total_failure}")
    logger.info(f"   Time elapsed: {elapsed_time:.2f}s")
    if total_success > 0 and elapsed_time > 0:
        logger.info(f"   Throughput: {total_success / elapsed_time:.1f} simulations/sec")
    logger.info("Results saved to:")
    logger.info(f"   {args.workspace}/strategies.db")
    logger.info(f"   {args.workspace}/strategies.csv")
    logger.info(f"   {args.workspace}/simulations/")

    # Calculate energy (automatic by default, skip with --skip-energy flag)
    # Why: Provide complete PPA metrics in one simulation run
    # When: After all simulations complete and before final summary
    if not args.skip_energy:
        logger.info("Calculating energy consumption...")

        try:
            from core.energy_calculator import EnergyCalculator

            # Load energy calculator from hardware config
            energy_calc = EnergyCalculator.from_config(hardware_config)

            # Calculate energy for all successful simulations
            energy_results = {}
            simulations_dir = workspace_path / "simulations"

            if simulations_dir.exists():
                for sim_dir in sorted(simulations_dir.iterdir()):
                    if not sim_dir.is_dir():
                        continue

                    strategy_id = sim_dir.name  # e.g., "L0_S0"

                    try:
                        # Load operation counts from simulation_statistics.json
                        # File written by SystemC EnergyTracker (~1KB)
                        energy_stats_file = sim_dir / "simulation_statistics.json"

                        if energy_stats_file.exists():
                            with open(energy_stats_file) as f:
                                op_counts = orjson.loads(f.read())

                            # Calculate energy from operation counts
                            energy_breakdown = energy_calc.calculate(op_counts)
                            energy_results[strategy_id] = energy_breakdown.total_nj

                            # Update database with energy
                            # Parse strategy_id (e.g., "L0_S0" → layer_idx=0, strategy_id=0)
                            parts = strategy_id.split("_")
                            if len(parts) == 2:
                                layer_idx = int(parts[0][1:])  # "L0" → 0
                                strategy_num = int(parts[1][1:])  # "S0" → 0

                                # Update database with full energy breakdown + simulation statistics
                                with performance_db._get_connection() as conn:
                                    cursor = conn.cursor()

                                    # Extract all fields from simulation_statistics.json
                                    ops = op_counts.get("operations", {})
                                    mem_access = op_counts.get("memory_accesses", {})
                                    data_mov = op_counts.get("data_movement", {})
                                    pipeline = op_counts.get("pipeline", {})
                                    buf_usage = op_counts.get("buffer_usage", {})
                                    summary = op_counts.get("summary", {})

                                    cursor.execute(
                                        """
                                        UPDATE strategy_results
                                        SET
                                            -- Energy breakdown
                                            energy_nj = ?,
                                            mac_energy_nj = ?,
                                            pooling_energy_nj = ?,
                                            activation_energy_nj = ?,
                                            sram_read_energy_nj = ?,
                                            sram_write_energy_nj = ?,
                                            dram_read_energy_nj = ?,
                                            dram_write_energy_nj = ?,
                                            communication_energy_nj = ?,
                                            static_energy_nj = ?,

                                            -- Operation counts
                                            mac_ops = ?,
                                            pooling_ops = ?,
                                            activation_ops = ?,
                                            comparison_ops = ?,
                                            total_operations = ?,

                                            -- Memory access counts
                                            external_reads = ?,
                                            external_writes = ?,
                                            ibuf_reads = ?,
                                            ibuf_writes = ?,
                                            obuf_reads = ?,
                                            obuf_writes = ?,
                                            weight_buf_reads = ?,
                                            weight_buf_writes = ?,
                                            cim_reads = ?,
                                            cim_writes = ?,
                                            total_memory_accesses = ?,

                                            -- Data movement
                                            external_to_ibuf_bytes = ?,
                                            obuf_to_external_bytes = ?,

                                            -- Pipeline operations
                                            pipeline_loads = ?,
                                            pipeline_ibuf_reads = ?,
                                            pipeline_cim_computes = ?,
                                            pipeline_obuf_writes = ?,
                                            pipeline_stores = ?,

                                            -- Buffer peak usage
                                            ibuf_peak_lines = ?,
                                            obuf_peak_lines = ?
                                        WHERE layer_idx = ? AND strategy_id = ?
                                        """,
                                        (
                                            # Energy breakdown
                                            energy_breakdown.total_nj,
                                            energy_breakdown.mac_energy_nj,
                                            energy_breakdown.pooling_energy_nj,
                                            energy_breakdown.activation_energy_nj,
                                            energy_breakdown.sram_read_energy_nj,
                                            energy_breakdown.sram_write_energy_nj,
                                            energy_breakdown.dram_read_energy_nj,
                                            energy_breakdown.dram_write_energy_nj,
                                            energy_breakdown.communication_energy_nj,
                                            energy_breakdown.static_energy_nj,

                                            # Operation counts
                                            ops.get("mac_ops", 0),
                                            ops.get("pooling_ops", 0),
                                            ops.get("activation_ops", 0),
                                            ops.get("comparison_ops", 0),
                                            summary.get("total_operations", 0),

                                            # Memory access counts
                                            mem_access.get("external_reads", 0),
                                            mem_access.get("external_writes", 0),
                                            mem_access.get("ibuf_reads", 0),
                                            mem_access.get("ibuf_writes", 0),
                                            mem_access.get("obuf_reads", 0),
                                            mem_access.get("obuf_writes", 0),
                                            mem_access.get("weight_buf_reads", 0),
                                            mem_access.get("weight_buf_writes", 0),
                                            mem_access.get("cim_reads", 0),
                                            mem_access.get("cim_writes", 0),
                                            summary.get("total_memory_accesses", 0),

                                            # Data movement
                                            data_mov.get("external_to_ibuf_bytes", 0),
                                            data_mov.get("obuf_to_external_bytes", 0),

                                            # Pipeline operations
                                            pipeline.get("loads", 0),
                                            pipeline.get("ibuf_reads", 0),
                                            pipeline.get("cim_computes", 0),
                                            pipeline.get("obuf_writes", 0),
                                            pipeline.get("stores", 0),

                                            # Buffer peak usage
                                            buf_usage.get("ibuf_peak_lines", 0),
                                            buf_usage.get("obuf_peak_lines", 0),

                                            # WHERE clause
                                            layer_idx,
                                            strategy_num,
                                        ),
                                    )
                                    conn.commit()

                    except Exception as e:
                        if args.verbose:
                            logger.warning(
                                f"    Energy calculation failed for {strategy_id}: {e}"
                            )
                        continue

                # Print energy summary
                if energy_results:
                    logger.info(f"   Energy calculated for {len(energy_results)} strategies")

                    # Find best/worst for quick insight
                    energies = [(k, v) for k, v in energy_results.items()]
                    if energies:
                        best_strategy, best_energy = min(energies, key=lambda x: x[1])
                        worst_strategy, worst_energy = max(energies, key=lambda x: x[1])

                        logger.info(f"   Best energy:  {best_strategy} ({best_energy:.2f} nJ)")
                        logger.info(f"   Worst energy: {worst_strategy} ({worst_energy:.2f} nJ)")
                        logger.info(f"   Energy ratio: {worst_energy/best_energy:.2f}x")
                else:
                    logger.warning("    No energy statistics found in simulation results")

        except ImportError:
            logger.warning("    Energy calculator not available")
        except Exception as e:
            logger.error(f"   Energy calculation error: {e}")
    else:
        logger.info(" Energy calculation skipped (--skip-energy flag)")

    # Export complete database to CSV
    # Why: Provide full dataset with all metrics including energy breakdown
    # When: After energy calculation completes (or is skipped)
    logger.info("Exporting complete database to CSV...")
    try:
        full_export_path = workspace_path / "full_export.csv"
        export_result = subprocess.run(
            [
                "sqlite3",
                "-header",
                "-csv",
                str(db_path),
                "SELECT * FROM strategy_results;",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        with open(full_export_path, "w") as f:
            f.write(export_result.stdout)

        logger.info(f"   Full database exported: {full_export_path.name}")
        logger.info(f"   {len(export_result.stdout.splitlines())-1} strategies exported")
    except subprocess.CalledProcessError as e:
        logger.warning(f"    CSV export failed: {e}")
    except Exception as e:
        logger.warning(f"    CSV export error: {e}")

    # Show next steps only in interactive mode (not during benchmark)
    if not args.quiet:
        print()
        # Use relative path from workspaces/ for correct command
        # e.g., "benchmark_xxx/preset_network" instead of just "preset_network"
        workspaces_dir = project_root / "workspaces"
        try:
            relative_workspace = workspace_path.relative_to(workspaces_dir)
        except ValueError:
            relative_workspace = workspace_path.name
        logger.info("▶️  Next step:")
        logger.info(f"   ./efsim plot {relative_workspace}  # Generate Pareto plots")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure Ray is shutdown even if main() exits normally
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
