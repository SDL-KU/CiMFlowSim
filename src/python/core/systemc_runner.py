"""
SystemC Runner - Python wrapper for pipeline_sim binary

Why: This module provides a clean Python interface to the C++ SystemC simulator.
We use a subprocess wrapper instead of embedding C++ in Python because:
1. Isolation: C++ crashes don't affect Python process
2. Flexibility: Can run simulations remotely or in containers
3. Simplicity: Standard JSON-based communication protocol
4. Debugging: Easy to run pipeline_sim standalone for debugging

The actual simulation logic is in C++ (src/systemc/pipeline_sim).
This Python class handles configuration, execution, and result parsing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import orjson

from .logging_config import get_logger

logger = get_logger(__name__)

from .exceptions import ConfigurationError
from .constants import (
    DEFAULT_TIMEOUT_SECONDS,
    DIR_NAME_SIMULATIONS,
    DIR_NAME_STRATEGIES,
    ERROR_MSG_INVALID_LOG_LEVEL,
    ERROR_MSG_INVALID_LOG_POLICY,
    ERROR_MSG_PIPELINE_SIM_NOT_FOUND,
    ERROR_MSG_TIMEOUT,
    ERROR_MSG_UNKNOWN,
    EXECUTION_TRACE_FILENAME,
    FILE_NAME_HARDWARE_CONFIG,
    FILE_NAME_NETWORK_CONFIG,
    GANTT_DATA_FILENAME,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_MINIMAL,
    LOG_LEVEL_STANDARD,
    LOG_POLICY_ALL,
    LOG_POLICY_FAILED,
    LOG_POLICY_NONE,
    PIPELINE_SIM_BINARY_NAME,
    SIMULATION_DIR_FORMAT,
    SIMULATION_LOG_FILENAME,
    STRATEGY_FILE_PATTERN,
    SYSTEMC_LIB_PATH,
    TENSOR_REGIONS_FILENAME,
    VALID_LOG_LEVELS,
    VALID_LOG_POLICIES,
)
from .tiling import (
    CNNLayerParams,
    StrategyDescriptor,
    TilingConfig,
)
from .systemc_parser import SystemCOutputParser, ParsedMetrics
from .systemc_visualizer import SimulationVisualizer

# =============================================================================
# Module-Specific Constants (Not in Centralized Package)
# =============================================================================

# Why: These constants are specific to systemc_runner.py implementation details
# and don't need to be shared across modules

# Subprocess Arguments
SUBPROCESS_OUTPUT_FLAG = "-o"

# Log File Headers
LOG_HEADER_SYSTEMC = "=== SystemC Simulation Log ==="
LOG_HEADER_CPP = "\n=== C++ SIMULATION LOG ==="
LOG_HEADER_STDOUT = "\n=== STDOUT ==="
LOG_HEADER_STDERR = "\n=== STDERR ==="
LOG_FIELD_STRATEGY_ID = "Strategy ID: {strategy_id}\n"
LOG_FIELD_RETURN_CODE = "Return Code: {return_code}\n"

# Error messages (module-specific)
ERROR_MSG_MISSING_BATCH_SIZE = (
    "Missing required field 'batch_size' in network config: {path}\n"
    "Please add 'batch_size' at network level in your network config JSON."
)
ERROR_MSG_MISSING_CNN_FIELDS = (
    "Missing required CNN parameter fields for layer {layer_idx}:\n"
    "  Missing fields: {missing_fields}\n"
    "  Layer config file: {config_file}"
)
ERROR_MSG_MISSING_TILING_FIELDS = (
    "Missing required tiling fields for strategy {strategy_id}:\n"
    "  Missing fields: {missing_fields}\n"
    "  Strategy file: {strategy_file}"
)
ERROR_MSG_MISSING_HW_FIELDS = (
    "Missing required hardware configuration fields:\n"
    "  Missing fields: {missing_fields}\n"
    "  Hardware config: {hw_description}"
)
ERROR_MSG_STRATEGY_FILE_NOT_FOUND = (
    "Strategy file not found: L{layer_idx}_S{strategy_id}_*.json in {strategies_dir}"
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SystemCResult:
    """
    Result from SystemC simulation

    Why: Encapsulate all simulation outputs in a single object for type safety
    and easy access. Includes both success metrics and failure diagnostics.
    """

    latency_ns: float
    area_mm2: float
    energy_nj: float
    ibuf_lines: int
    obuf_lines: int
    success: bool
    error_message: Optional[str] = None
    operations_completed: Optional[Dict[str, int]] = None  # For validation
    return_code: int = 0  # Process return code
    stdout: str = ""  # Simulation stdout
    stderr: str = ""  # Simulation stderr


# =============================================================================
# Helper Functions: Validation
# =============================================================================


def _validate_log_policy(save_logs: str) -> None:
    """
    Validate log saving policy

    Why: Fail-fast validation prevents invalid configuration from reaching
    the simulation loop where it would cause confusing errors.

    Args:
        save_logs: Log policy to validate

    Raises:
        ValueError: If policy is not in VALID_LOG_POLICIES
    """
    if save_logs not in VALID_LOG_POLICIES:
        raise ConfigurationError(
            ERROR_MSG_INVALID_LOG_POLICY.format(
                value=save_logs, valid=", ".join(f"'{p}'" for p in VALID_LOG_POLICIES)
            ),
            context={"provided_value": save_logs, "valid_policies": list(VALID_LOG_POLICIES)},
            suggestions=["Use one of: 'all', 'failed', 'none'"],
        )


def _validate_log_level(log_level: str) -> None:
    """
    Validate log detail level

    Why: Ensures only supported log levels are used, preventing undefined
    behavior in log cleanup logic.

    Args:
        log_level: Log level to validate

    Raises:
        ValueError: If level is not in VALID_LOG_LEVELS
    """
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigurationError(
            ERROR_MSG_INVALID_LOG_LEVEL.format(
                value=log_level,
                valid=", ".join(f"'{level}'" for level in VALID_LOG_LEVELS),
            ),
            context={"provided_value": log_level, "valid_levels": list(VALID_LOG_LEVELS)},
            suggestions=["Use one of: 'minimal', 'standard', 'detailed'"],
        )


def _validate_pipeline_sim_exists(pipeline_sim_path: Path) -> None:
    """
    Validate pipeline_sim binary exists

    Why: Check binary existence at initialization time rather than at first
    simulation, providing immediate feedback to users.

    Args:
        pipeline_sim_path: Path to pipeline_sim binary

    Raises:
        FileNotFoundError: If binary doesn't exist
    """
    if not pipeline_sim_path.exists():
        raise FileNotFoundError(ERROR_MSG_PIPELINE_SIM_NOT_FOUND.format(path=pipeline_sim_path))


def _validate_required_fields(config: dict, required_fields: list, context: str) -> list:
    """
    Validate required configuration fields are present

    Why: Centralize field validation logic to ensure consistent error messages
    and reduce code duplication across different config types.

    Args:
        config: Configuration dictionary to validate
        required_fields: List of required field names
        context: Context for error message (e.g., "CNN parameters")

    Returns:
        List of missing field names (empty if all present)
    """
    return [field for field in required_fields if field not in config]


# =============================================================================
# Helper Functions: Environment Setup
# =============================================================================


def _setup_systemc_environment() -> dict:
    """
    Setup environment variables for SystemC execution

    Why: SystemC library path must be in LD_LIBRARY_PATH for dynamic linking.
    We create a new environment dict instead of modifying os.environ to avoid
    side effects on the parent Python process.

    Returns:
        Environment dictionary with SystemC library path configured
    """
    env = os.environ.copy()
    current_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{SYSTEMC_LIB_PATH}:{current_ld_path}"
    return env


# =============================================================================
# Helper Functions: Subprocess Execution
# =============================================================================


def _build_pipeline_sim_command(
    pipeline_sim_path: Path,
    strategy_path: Path,
    network_path: Path,
    hardware_path: Path,
    log_output_dir: Path,
    save_logs: str = LOG_POLICY_NONE,
    generate_gantt: bool = False,
) -> list:
    """
    Build command-line arguments for pipeline_sim subprocess

    Why: 3-file architecture (strategy, network, hardware) separates concerns:
    - Strategy: varies for each candidate (100s-1000s)
    - Network: shared across all strategies for same layer (1 per layer)
    - Hardware: shared across all simulations (1 per workspace)
    This reduces file I/O and JSON parsing overhead.

    Args:
        pipeline_sim_path: Path to pipeline_sim binary
        strategy_path: Path to strategy JSON file
        network_path: Path to network config JSON file
        hardware_path: Path to hardware config JSON file
        log_output_dir: Directory for simulation output files
        save_logs: Log saving policy (none/failed/all)
        generate_gantt: Whether to generate Gantt chart data

    Returns:
        List of command-line arguments for subprocess.run()
    """
    cmd = [
        str(pipeline_sim_path.resolve()),
        str(strategy_path.resolve()),
        str(network_path.resolve()),
        str(hardware_path.resolve()),
        SUBPROCESS_OUTPUT_FLAG,
        str(log_output_dir.resolve()),
    ]

    # Add granular log control flags based on policy
    # By default (save_logs="none"), no flags added = no large log files generated
    if save_logs == LOG_POLICY_ALL:
        # Save all detailed logs for debugging
        cmd.append("--save-simulation-log")
        cmd.append("--save-gantt-data")
        cmd.append("--save-execution-trace")
        cmd.append("--save-dependency-graph")
    elif generate_gantt:
        # Gantt only needs gantt_data.txt (~44MB), not simulation_log.txt (~100MB)
        cmd.append("--save-gantt-data")
    # Note: LOG_POLICY_FAILED is handled by Python cleanup, not C++ flags
    # (C++ doesn't know if simulation will fail until it runs)
    # Note: memory_metadata.json is always generated by SystemC (lightweight ~2KB)

    return cmd


# =============================================================================
# Helper Functions: Directory and File Naming
# =============================================================================


def _build_simulation_dir_name(layer_idx: int, strategy_id: int) -> str:
    """
    Build standard simulation directory name

    Why: Consistent naming enables easy identification and log management.
    Format: L{layer}_S{strategy} (e.g., L0_S42)

    Args:
        layer_idx: Layer index (0-based)
        strategy_id: Strategy identifier

    Returns:
        Directory name string
    """
    return SIMULATION_DIR_FORMAT.format(layer_idx=layer_idx, strategy_id=strategy_id)


def _build_strategy_file_pattern(layer_idx: int, strategy_id: int) -> str:
    """
    Build glob pattern for strategy files

    Why: Strategy files include descriptive suffixes (e.g., _out2x2_in5x5.json)
    for human readability. Glob pattern matches any suffix.

    Args:
        layer_idx: Layer index (0-based)
        strategy_id: Strategy identifier

    Returns:
        Glob pattern string
    """
    return STRATEGY_FILE_PATTERN.format(layer_idx=layer_idx, strategy_id=strategy_id)


# =============================================================================
# Helper Functions: Log Headers
# =============================================================================


def _build_log_header(strategy_id: int, return_code: int) -> str:
    """
    Build standardized log file header

    Why: Consistent headers enable automated log parsing and filtering.
    Includes strategy ID for correlation and return code for quick status check.

    Args:
        strategy_id: Strategy identifier
        return_code: Process return code

    Returns:
        Formatted log header string
    """
    return (
        f"{LOG_HEADER_SYSTEMC}\n"
        f"{LOG_FIELD_STRATEGY_ID.format(strategy_id=strategy_id)}"
        f"{LOG_FIELD_RETURN_CODE.format(return_code=return_code)}"
    )


# =============================================================================
# Helper Functions: Result Creation
# =============================================================================


def _create_error_result(
    error_message: str,
    return_code: int = -1,
    stdout: str = "",
    stderr: str = "",
) -> SystemCResult:
    """
    Create SystemCResult for error cases

    Why: Centralize error result creation to ensure consistent structure.
    All error results have success=False and zero metrics.

    Args:
        error_message: Human-readable error description
        return_code: Process return code (default: -1 for internal errors)
        stdout: Process stdout (for debugging)
        stderr: Process stderr (for debugging)

    Returns:
        SystemCResult with error status
    """
    return SystemCResult(
        latency_ns=0,
        area_mm2=0,
        energy_nj=0,
        ibuf_lines=0,
        obuf_lines=0,
        success=False,
        error_message=error_message,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
    )


def _create_success_result(
    latency_ns: float,
    area_mm2: float,
    energy_nj: float,
    ibuf_lines: int,
    obuf_lines: int,
    operations_completed: Optional[Dict[str, int]],
    return_code: int,
    stdout: str,
    stderr: str,
) -> SystemCResult:
    """
    Create SystemCResult for successful simulation

    Why: Centralize success result creation and apply validation logic.
    Success requires latency_ns > 0 (zero means simulation didn't run).

    Args:
        latency_ns: Total simulation time in nanoseconds
        area_mm2: Total hardware area in mm²
        energy_nj: Total energy consumption in nJ
        ibuf_lines: Input buffer depth in memory lines
        obuf_lines: Output buffer depth in memory lines
        operations_completed: Operation counts for validation
        return_code: Process return code (should be 0)
        stdout: Process stdout
        stderr: Process stderr

    Returns:
        SystemCResult with success status
    """
    return SystemCResult(
        latency_ns=latency_ns,
        area_mm2=area_mm2,
        energy_nj=energy_nj,
        ibuf_lines=ibuf_lines,
        obuf_lines=obuf_lines,
        operations_completed=operations_completed,
        success=True if latency_ns > 0 else False,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
    )


# =============================================================================
# Helper Functions: Parsing - JSON
# =============================================================================


def _extract_json_section(output: str, start_marker: str, end_marker: str) -> Optional[dict]:
    """
    Extract and parse JSON section from simulation output

    Why: SystemC outputs JSON wrapped in markers for easy extraction.
    This avoids complex regex and handles multi-line JSON robustly.

    Args:
        output: Complete simulation output text
        start_marker: Start marker string (e.g., "=== Energy Statistics JSON ===")
        end_marker: End marker string (e.g., "=== End Energy Statistics JSON ===")

    Returns:
        Parsed JSON dictionary, or None if section not found or invalid JSON
    """
    if start_marker not in output:
        return None

    start_idx = output.find(start_marker) + len(start_marker)
    end_idx = output.find(end_marker)

    if end_idx <= start_idx:
        return None

    json_str = output[start_idx:end_idx].strip()
    try:
        return orjson.loads(json_str)
    except orjson.JSONDecodeError:
        return None


# =============================================================================
# Main Runner Class
# =============================================================================


class SystemCRunner:
    """
    Runner for SystemC pipeline_sim binary

    Why: This wrapper class provides a clean Python interface to the C++ simulator.
    We use subprocess execution instead of direct C++ embedding because:
    1. Isolation: C++ crashes don't kill the Python process
    2. Parallel execution: Can run multiple simulations concurrently
    3. Remote execution: Can distribute simulations across machines
    4. Debugging: Easy to run pipeline_sim standalone for testing

    The C++ simulator (src/systemc/pipeline_sim) contains the actual simulation logic.
    This Python class handles:
    - Configuration file management
    - Subprocess execution and timeout handling
    - Output parsing (JSON and text formats)
    - Log retention policies
    - Gantt chart generation
    """

    def __init__(
        self,
        systemc_dir: str = "src/systemc",
        timeout: int | None = DEFAULT_TIMEOUT_SECONDS,
        save_logs: str = LOG_POLICY_FAILED,
        generate_gantt: bool = False,
        generate_memory_layout: bool = False,
        log_level: str = LOG_LEVEL_STANDARD,
    ):
        """
        Initialize SystemC runner

        Why: Validate all configuration at initialization time to fail fast.
        Setup SystemC library path once instead of repeating for each simulation.

        Args:
            systemc_dir: Path to SystemC directory containing pipeline_sim binary
            timeout: Simulation timeout in seconds (default: 30)
            save_logs: Log saving policy (default: 'failed')
                - 'none': Delete all logs after simulation (fastest, 0 bytes)
                - 'failed': Keep logs only for failed simulations (good for debugging)
                - 'all': Keep all logs (slowest, most disk usage)
            generate_gantt: Generate Gantt chart PDFs (default: False)
                - False: Skip PDF generation (faster, saves ~85KB per strategy)
                - True: Generate timeline visualization (useful for analysis)
            generate_memory_layout: Generate memory layout PDFs (default: False)
                - False: Skip PDF generation (faster)
                - True: Generate memory bank layout visualization
            log_level: Log detail level (default: 'standard')
                - 'minimal': gantt_data.txt only (~156KB)
                - 'standard': + simulation_log (~2.5MB)
                - 'debug': + execution_trace + tensor_regions (~3.1MB)

        Raises:
            ValueError: If save_logs or log_level is invalid
            FileNotFoundError: If pipeline_sim binary not found
        """
        # Convert to absolute path to support Ray parallel execution
        self.systemc_dir = Path(systemc_dir).resolve()
        self.pipeline_sim = self.systemc_dir / PIPELINE_SIM_BINARY_NAME
        self.timeout = timeout
        self.save_logs = save_logs
        self.generate_gantt = generate_gantt
        self.generate_memory_layout = generate_memory_layout
        self.log_level = log_level

        # Validate configuration
        _validate_log_policy(save_logs)
        _validate_log_level(log_level)
        _validate_pipeline_sim_exists(self.pipeline_sim)

        # Set up SystemC library path
        self.env = _setup_systemc_environment()

        # Initialize parser and visualizer (delegation pattern)
        self._parser = SystemCOutputParser()
        self._visualizer = SimulationVisualizer()

    def _run_systemc_subprocess(
        self,
        strategy_path: Path,
        network_path: Path,
        hardware_path: Path,
        log_output_dir: Path,
    ) -> subprocess.CompletedProcess:
        """
        Execute SystemC simulation subprocess with 3 separate config files

        Why: 3-file architecture separates frequently-changing (strategy),
        layer-specific (network), and shared (hardware) configurations.
        This reduces file I/O and JSON parsing overhead in DSE runs.

        Args:
            strategy_path: Path to strategy JSON file
            network_path: Path to network config JSON file
            hardware_path: Path to hardware config JSON file
            log_output_dir: Directory for output files

        Returns:
            CompletedProcess with return code, stdout, stderr
        """
        command = _build_pipeline_sim_command(
            self.pipeline_sim,
            strategy_path,
            network_path,
            hardware_path,
            log_output_dir,
            save_logs=self.save_logs,
            generate_gantt=self.generate_gantt,
        )

        return subprocess.run(
            command,
            cwd=str(self.systemc_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=self.env,
        )

    def _read_simulation_logs(self, log_output_dir: Path) -> tuple:
        """
        Read simulation logs and generate PDF if available

        Why: C++ simulator writes detailed logs to files instead of stdout
        to avoid polluting the console. We read these logs for parsing and
        optionally generate Gantt charts for visualization.

        Args:
            log_output_dir: Directory containing simulation output files

        Returns:
            Tuple of (simulation_log_content, pdf_generated)
        """
        simulation_log_file = log_output_dir / SIMULATION_LOG_FILENAME
        simulation_log_content = ""
        if simulation_log_file.exists():
            with open(simulation_log_file, "r") as f:
                simulation_log_content = f.read()

        # Generate Gantt PDF if enabled (now fast with numpy optimization)
        gantt_generated = False
        if self.generate_gantt:
            gantt_generated = self._generate_gantt_pdf(log_output_dir)

        # Generate memory layout PDF if enabled
        memory_layout_generated = False
        if self.generate_memory_layout:
            memory_layout_generated = self._generate_memory_layout_pdf(log_output_dir)

        return simulation_log_content, gantt_generated or memory_layout_generated

    def _generate_gantt_pdf(self, log_output_dir: Path) -> bool:
        """
        Generate Gantt chart PDF from binary gantt data.

        Delegates to SimulationVisualizer for actual generation.

        Args:
            log_output_dir: Directory containing gantt_data.bin

        Returns:
            True if PDF was generated successfully, False otherwise
        """
        return self._visualizer.generate_gantt_pdf(log_output_dir)

    def _generate_memory_layout_pdf(self, log_output_dir: Path) -> bool:
        """
        Generate memory layout PDF from memory_metadata.json.

        Delegates to SimulationVisualizer for actual generation.

        Args:
            log_output_dir: Directory containing memory_metadata.json

        Returns:
            True if PDF was generated successfully, False otherwise
        """
        return self._visualizer.generate_memory_layout_pdf(log_output_dir)

    def _read_simulation_statistics(self, log_output_dir: Path) -> Optional[dict]:
        """
        Read simulation_statistics.json written by SystemC

        Why: Unified data source - read all simulation metrics from one JSON file
        instead of parsing stdout. SystemC writes this file with complete statistics
        including timing, operations, memory accesses, pipeline metrics, and buffer usage.

        Args:
            log_output_dir: Directory containing simulation output files

        Returns:
            Dict with simulation statistics, or None if file doesn't exist or parsing fails
        """
        stats_file = log_output_dir / "simulation_statistics.json"
        if not stats_file.exists():
            return None

        try:
            with open(stats_file, "r") as f:
                return orjson.loads(f.read())
        except (OSError, orjson.JSONDecodeError) as e:
            logger.warning(
                f"Failed to read simulation_statistics.json: {str(e)}",
                extra={"stats_file": str(stats_file), "error_type": type(e).__name__},
            )
            return None

    def _save_simulation_log(
        self,
        log_dir: Path,
        strategy_id: int,
        return_code: int,
        cpp_log: str,
        stdout: str,
        stderr: str,
    ):
        """
        Save combined simulation log to file

        Why: Combine all simulation outputs (C++ log, stdout, stderr) into
        a single file for easier debugging. Include metadata (strategy ID,
        return code) for quick identification.

        Args:
            log_dir: Directory to save log file
            strategy_id: Strategy identifier
            return_code: Process return code
            cpp_log: C++ simulation log content
            stdout: Process stdout
            stderr: Process stderr
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / SIMULATION_LOG_FILENAME

        header = _build_log_header(strategy_id, return_code)

        with open(log_file, "w") as f:
            f.write(header)
            f.write(LOG_HEADER_CPP)
            f.write(cpp_log)
            f.write(LOG_HEADER_STDOUT)
            f.write(stdout)
            f.write(LOG_HEADER_STDERR)
            f.write(stderr)

    def _cleanup_logs(self, log_dir: Path, success: bool) -> None:
        """
        Clean up simulation logs based on save_logs and log_level policies

        Why: DSE runs generate 100s-1000s of simulations. Without cleanup,
        disk usage can reach 100s of GB. Policy-based cleanup balances
        debugging needs with disk space:
        - 'all': Keep everything (debug mode)
        - 'failed': Keep only failures (development)
        - 'none': Delete everything (production)

        Args:
            log_dir: Directory containing simulation logs
            success: Whether simulation was successful
        """
        # Policy: 'all' - keep logs based on log_level
        if self.save_logs == LOG_POLICY_ALL:
            self._apply_log_level_filter(log_dir)
            return

        # Policy: 'failed' - keep failed logs (with log_level filter), remove successful logs
        if self.save_logs == LOG_POLICY_FAILED:
            if success and log_dir.exists():
                # Remove log files but preserve essential small files:
                # - simulation_statistics.json: Energy calculation needs operation counts (~1KB)
                # - memory_metadata.json: Memory layout visualization needs this (~2KB)
                # - gantt files: Only if --generate-gantt was used
                # These are much smaller than full simulation logs (~100MB)
                files_to_preserve = [
                    "simulation_statistics.json",
                    "memory_metadata.json",
                ]

                # If generate_gantt is enabled, preserve gantt files
                if self.generate_gantt:
                    files_to_preserve.extend([
                        GANTT_DATA_FILENAME,
                        "gantt_chart.pdf",
                    ])

                # If generate_memory_layout is enabled, preserve memory layout PDF
                if self.generate_memory_layout:
                    files_to_preserve.append("memory_layout.pdf")

                # Read files to preserve
                preserved_data = []
                for filename in files_to_preserve:
                    filepath = log_dir / filename
                    if filepath.exists():
                        preserved_data.append((filename, filepath.read_bytes()))

                if preserved_data:
                    # Remove entire directory
                    shutil.rmtree(log_dir)

                    # Recreate directory and restore preserved files
                    log_dir.mkdir(parents=True, exist_ok=True)
                    for filename, data in preserved_data:
                        (log_dir / filename).write_bytes(data)
                else:
                    # No files to preserve, just delete
                    shutil.rmtree(log_dir)
            else:
                # Failed: apply log_level filter
                self._apply_log_level_filter(log_dir)
            return

        # Policy: 'none' - remove all logs (including energy statistics)
        if self.save_logs == LOG_POLICY_NONE:
            if log_dir.exists():
                shutil.rmtree(log_dir)
            return

    def _apply_log_level_filter(self, log_dir: Path) -> None:
        """
        Apply log_level filtering to keep only necessary log files

        Why: Log files have different sizes and utility:
        - gantt_data.txt: ~156KB, essential for timeline analysis
        - simulation_log.txt: ~2.4MB, useful for debugging
        - execution_trace.log: ~400KB, low-level debugging
        - tensor_regions.log: ~100KB, memory access patterns

        Trade-off: minimal saves 95% disk space but loses detailed traces.

        Args:
            log_dir: Directory containing simulation logs
        """
        if not log_dir.exists():
            return

        # Debug: keep everything
        if self.log_level == LOG_LEVEL_DEBUG:
            return

        # Standard: remove execution_trace and tensor_regions
        if self.log_level == LOG_LEVEL_STANDARD:
            files_to_remove = [
                log_dir / EXECUTION_TRACE_FILENAME,
                log_dir / TENSOR_REGIONS_FILENAME,
            ]
            for file in files_to_remove:
                if file.exists():
                    file.unlink()
            return

        # Minimal: keep only gantt_data.txt
        if self.log_level == LOG_LEVEL_MINIMAL:
            # Keep gantt_data.txt, remove everything else
            files_to_keep = [GANTT_DATA_FILENAME]
            for file in log_dir.iterdir():
                if file.is_file() and file.name not in files_to_keep:
                    file.unlink()
            return

    def simulate(
        self,
        workspace_path: Path,
        strategy_path: Path,
        network_path: Path,
        hardware_path: Path,
        layer_idx: int = 0,
        strategy_id: int = 0,
        log_dir: Optional[Path] = None,
    ) -> SystemCResult:
        """
        Run SystemC simulation with 3 separate configuration files

        Why: Main entry point for simulation execution. Orchestrates the
        complete workflow: directory creation, subprocess execution,
        log management, output parsing, and cleanup.

        Args:
            workspace_path: Workspace directory path
            strategy_path: Path to strategy JSON file
            network_path: Path to network config JSON file
            hardware_path: Path to hardware config JSON file
            layer_idx: Layer index for unique file naming
            strategy_id: Strategy ID for logging
            log_dir: Optional directory to save simulation logs

        Returns:
            SystemCResult with performance metrics

        Raises:
            subprocess.TimeoutExpired: If simulation exceeds timeout
            Exception: For other unexpected errors
        """
        try:
            # Create dedicated log directory for this simulation
            sim_dir_name = _build_simulation_dir_name(layer_idx, strategy_id)
            log_output_dir = (workspace_path / DIR_NAME_SIMULATIONS / sim_dir_name).resolve()
            log_output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Execute SystemC subprocess with 3 file paths
            result = self._run_systemc_subprocess(
                strategy_path, network_path, hardware_path, log_output_dir
            )

            # Step 2: Read simulation logs
            simulation_log_content, _ = self._read_simulation_logs(log_output_dir)
            combined_stdout = simulation_log_content if simulation_log_content else result.stdout

            # Step 3: Read simulation statistics JSON (new unified data source)
            stats_json = self._read_simulation_statistics(log_output_dir)

            # Step 4: Save logs if requested
            if log_dir is not None:
                self._save_simulation_log(
                    log_dir,
                    strategy_id,
                    result.returncode,
                    simulation_log_content,
                    result.stdout,
                    result.stderr,
                )

            # Step 5: Check result and parse output
            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else ERROR_MSG_UNKNOWN
                sim_result = _create_error_result(
                    error_msg, result.returncode, combined_stdout, result.stderr
                )
                # Clean up logs based on policy (keep failed logs if policy is 'failed' or 'all')
                self._cleanup_logs(log_output_dir, success=False)
                return sim_result

            # Parse from JSON file if available, otherwise fallback to stdout parsing
            if stats_json is not None:
                sim_result = self._parse_from_json(stats_json, result.returncode, combined_stdout, result.stderr)
            else:
                sim_result = self._parse_output(
                    combined_stdout, result.returncode, combined_stdout, result.stderr
                )

            # Clean up logs based on policy (remove successful logs if policy is 'none')
            self._cleanup_logs(log_output_dir, success=sim_result.success)
            return sim_result

        except subprocess.TimeoutExpired as e:
            return _create_error_result(
                ERROR_MSG_TIMEOUT,
                stdout=e.stdout if e.stdout else "",
                stderr=e.stderr if e.stderr else "",
            )
        except (OSError, subprocess.SubprocessError) as e:
            return _create_error_result(str(e))

    def _parse_output(
        self, output: str, return_code: int = 0, stdout: str = "", stderr: str = ""
    ) -> SystemCResult:
        """
        Parse SystemC simulation output

        Delegates to SystemCOutputParser for actual parsing logic.

        Args:
            output: Combined stdout/stderr from simulation
            return_code: Process return code
            stdout: Process stdout (for logging)
            stderr: Process stderr (for logging)

        Returns:
            SystemCResult with all metrics extracted
        """
        parsed = self._parser.parse_output(output)
        return SystemCResult(
            latency_ns=parsed.latency_ns,
            area_mm2=parsed.area_mm2,
            energy_nj=parsed.energy_nj,
            ibuf_lines=parsed.ibuf_lines,
            obuf_lines=parsed.obuf_lines,
            operations_completed=parsed.operations_completed,
            success=parsed.success,
            error_message=parsed.error_message if not parsed.success else None,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        )

    def _parse_from_json(
        self, stats: dict, return_code: int = 0, stdout: str = "", stderr: str = ""
    ) -> SystemCResult:
        """
        Parse simulation metrics from simulation_statistics.json

        Delegates to SystemCOutputParser for actual parsing logic.

        Args:
            stats: Dict loaded from simulation_statistics.json
            return_code: Process return code
            stdout: Process stdout (for logging)
            stderr: Process stderr (for logging)

        Returns:
            SystemCResult with all metrics extracted from JSON
        """
        parsed = self._parser.parse_from_json_file(stats)
        return SystemCResult(
            latency_ns=parsed.latency_ns,
            area_mm2=parsed.area_mm2,
            energy_nj=parsed.energy_nj,
            ibuf_lines=parsed.ibuf_lines,
            obuf_lines=parsed.obuf_lines,
            operations_completed=parsed.operations_completed,
            success=parsed.success,
            error_message=parsed.error_message if not parsed.success else None,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        )

    def simulate_from_workspace(
        self,
        workspace_path: Path,
        layer_idx: int,
        strategy_id: int,
        hardware_config: dict,
        output_path: Optional[Path] = None,
    ) -> SystemCResult:
        """
        High-level method: Simulate a strategy from workspace files

        This method handles the complete workflow:
        1. Load configurations from workspace
        2. Validate all required fields
        3. Run SystemC simulation
        4. Validate results

        Args:
            workspace_path: Path to workspace directory (for reading input files)
            layer_idx: Layer index (0-based)
            strategy_id: Strategy ID
            hardware_config: Hardware configuration dict
            output_path: Optional separate path for output files (for local caching)

        Returns:
            SystemCResult with performance metrics and validation status

        Raises:
            ValueError: If workspace_path doesn't exist or hardware_config is empty
            FileNotFoundError: If required workspace files are missing
        """
        # Validate inputs
        if not workspace_path.exists():
            raise FileNotFoundError(
                f"Workspace directory not found: {workspace_path}\n"
                "Please check the workspace path and ensure it exists."
            )

        if not hardware_config:
            raise ValueError(
                "hardware_config is empty. "
                "Expected hardware configuration dictionary with 'hardware' section."
            )

        # Use output_path if provided, otherwise use workspace_path
        effective_output_path = output_path if output_path else workspace_path

        # Load and validate strategy from workspace
        strategy, cnn_params = self._load_strategy_from_workspace(
            workspace_path, layer_idx, strategy_id, hardware_config
        )

        # Run simulation with 3-file architecture
        # Find the strategy file (descriptive naming: L0_S0_out2x2_in5x5.json)
        strategies_dir = workspace_path / DIR_NAME_STRATEGIES
        pattern = _build_strategy_file_pattern(layer_idx, strategy_id)
        matches = list(strategies_dir.glob(pattern))

        if not matches:
            raise FileNotFoundError(
                f"Strategy file not found: L{layer_idx}_S{strategy_id}_*.json in {strategies_dir}"
            )

        strategy_path = matches[0]  # Use first match
        network_path = workspace_path / FILE_NAME_NETWORK_CONFIG
        hardware_path = workspace_path / FILE_NAME_HARDWARE_CONFIG

        result = self.simulate(
            workspace_path=effective_output_path,  # Output goes here
            strategy_path=strategy_path,
            network_path=network_path,
            hardware_path=hardware_path,
            layer_idx=layer_idx,
            strategy_id=strategy_id,
            log_dir=None,
        )

        # Validate result
        return self._validate_result(result, strategy, cnn_params)

    def _load_strategy_from_workspace(
        self,
        workspace_path: Path,
        layer_idx: int,
        strategy_id: int,
        hardware_config: dict,
    ) -> tuple:
        """
        Load and validate strategy configuration from workspace

        Returns:
            Tuple of (StrategyDescriptor, CNNLayerParams)
        """
        from .config_utils import extract_hardware_constraints
        from .workspace_manager import WorkspaceManager

        workspace = WorkspaceManager(workspace_path)

        # Load network config to get batch_size
        network_config_path = workspace_path / FILE_NAME_NETWORK_CONFIG
        with open(network_config_path) as f:
            network_config = orjson.loads(f.read())

        if "batch_size" not in network_config:
            raise ValueError(
                f"Missing required field 'batch_size' in network config: {network_config_path}\n"
                "Please add 'batch_size' at network level in your network config JSON."
            )
        network_batch_size = network_config["batch_size"]

        # Load layer configuration
        layer_config = workspace.load_layer_config(layer_idx)
        params = layer_config.get("params", layer_config)

        # Validate required CNN fields (including bitwidths)
        required_cnn_fields = [
            "H",
            "W",
            "C",
            "R",
            "S",
            "M",
            "stride",
            "pool_height",
            "pool_width",
            "input_bitwidth",
            "output_bitwidth",
        ]
        missing_fields = [f for f in required_cnn_fields if f not in params]

        # batch_size: try params first, then layer_config, then network level
        batch_size = (
            params.get("batch_size") or layer_config.get("batch_size") or network_batch_size
        )
        if batch_size is None:
            missing_fields.append("batch_size")

        # Bitwidths are now required fields (no fallback)
        input_bitwidth = params["input_bitwidth"]
        output_bitwidth = params["output_bitwidth"]

        if missing_fields:
            raise ValueError(
                f"Missing required CNN parameter fields for layer {layer_idx}:\n"
                f"  Missing fields: {', '.join(missing_fields)}\n"
                f"  Layer config file: {workspace.layers_dir / f'L{layer_idx}.json'}"
            )

        cnn_params = CNNLayerParams(
            H=params["H"],
            W=params["W"],
            C=params["C"],
            R=params["R"],
            S=params["S"],
            M=params["M"],
            stride=params["stride"],
            batch_size=batch_size,
            input_bitwidth=input_bitwidth,
            output_bitwidth=output_bitwidth,
            pool_height=params["pool_height"],
            pool_width=params["pool_width"],
        )

        # Load strategy configuration
        strategy_config = workspace.load_strategy_config(layer_idx, strategy_id)
        tiling_conf = strategy_config["tiling_config"]

        # Validate required tiling fields
        required_tiling_fields = [
            "output_tile_p",
            "output_tile_q",
            "input_tile_h",
            "input_tile_w",
            "input_tile_p",
            "input_tile_q",
            "num_output_tiles_p",
            "num_output_tiles_q",
            "num_input_tiles_p",
            "num_input_tiles_q",
            "input_tile_count",
            "output_tile_count",
            "case_type",
            "total_loads",
            "total_ibuf_reads",
            "total_cim_computes",
            "total_obuf_writes",
            "total_stores",
        ]
        missing_tiling_fields = [f for f in required_tiling_fields if f not in tiling_conf]

        if missing_tiling_fields:
            strategy_file = workspace.strategies_dir / f"L{layer_idx}_S{strategy_id}.json"
            raise ValueError(
                f"Missing required tiling fields for strategy {strategy_id}:\n"
                f"  Missing fields: {', '.join(missing_tiling_fields)}\n"
                f"  Strategy file: {strategy_file}"
            )

        # Create TilingConfig
        tiling_config = TilingConfig(
            strategy_id=strategy_id,
            description=f"Strategy {strategy_id} for layer {layer_idx}",
            output_tile_p=tiling_conf["output_tile_p"],
            output_tile_q=tiling_conf["output_tile_q"],
            input_tile_h=tiling_conf["input_tile_h"],
            input_tile_w=tiling_conf["input_tile_w"],
            input_tile_p=tiling_conf["input_tile_p"],
            input_tile_q=tiling_conf["input_tile_q"],
            num_output_tiles_p=tiling_conf["num_output_tiles_p"],
            num_output_tiles_q=tiling_conf["num_output_tiles_q"],
            num_input_tiles_p=tiling_conf["num_input_tiles_p"],
            num_input_tiles_q=tiling_conf["num_input_tiles_q"],
            input_tile_count=tiling_conf["input_tile_count"],
            output_tile_count=tiling_conf["output_tile_count"],
            case_type=tiling_conf["case_type"],
            total_loads=tiling_conf["total_loads"],
            total_ibuf_reads=tiling_conf["total_ibuf_reads"],
            total_cim_computes=tiling_conf["total_cim_computes"],
            total_obuf_writes=tiling_conf["total_obuf_writes"],
            total_stores=tiling_conf["total_stores"],
        )

        # Extract and validate hardware constraints
        hw_arch, hw_ports = extract_hardware_constraints(hardware_config)
        required_hw_arch_fields = ["num_banks", "bits_per_line"]
        required_hw_port_fields = ["num_rw_ports"]

        missing_hw_fields = []
        missing_hw_fields.extend(
            f"architecture.{f}" for f in required_hw_arch_fields if f not in hw_arch
        )
        missing_hw_fields.extend(f"ports.{f}" for f in required_hw_port_fields if f not in hw_ports)

        if missing_hw_fields:
            raise ValueError(
                f"Missing required hardware configuration fields:\n"
                f"  Missing fields: {', '.join(missing_hw_fields)}\n"
                f"  Hardware config: {hardware_config.get('description', 'unknown')}"
            )

        # Create strategy descriptor
        strategy = StrategyDescriptor(
            strategy_id=strategy_id,
            description=f"Strategy {strategy_id} for layer {layer_idx}",
            tiling_config=tiling_config,
            cnn_params=cnn_params,
        )

        return strategy, cnn_params

    def _validate_result(
        self, result: SystemCResult, strategy: StrategyDescriptor, cnn_params: CNNLayerParams
    ) -> SystemCResult:
        """
        Validate simulation result against expected operation counts

        Args:
            result: Raw simulation result
            strategy: Strategy descriptor with expected counts
            cnn_params: CNN layer parameters

        Returns:
            SystemCResult with validation status updated
        """
        from .simulation_validator import SimulationValidator

        if not result.success:
            return result  # Already failed, no need to validate

        # Create validation config dict (SimulationValidator API)
        validation_config = {
            "cnn_layer": {
                "batch_size": cnn_params.batch_size,
                "pool_height": cnn_params.pool_height,
                "pool_width": cnn_params.pool_width,
            },
            "tiling_config": {
                "output_tile_p": strategy.tiling_config.output_tile_p,
                "output_tile_q": strategy.tiling_config.output_tile_q,
                "input_tile_h": strategy.tiling_config.input_tile_h,
                "input_tile_w": strategy.tiling_config.input_tile_w,
                "output_tile_count": strategy.tiling_config.output_tile_count,
                "input_tile_count": strategy.tiling_config.input_tile_count,
                "case_type": strategy.tiling_config.case_type,
            },
        }

        # Validate operation counts
        validator = SimulationValidator(validation_config, result)
        validation_result = validator.validate_all(level="basic")

        if not validation_result.is_valid():
            # Mark as failed if validation fails
            error_details = (
                "\n".join(validation_result.errors)
                if validation_result.errors
                else "Unknown validation error"
            )
            return SystemCResult(
                latency_ns=result.latency_ns,
                area_mm2=result.area_mm2,
                energy_nj=result.energy_nj,
                ibuf_lines=result.ibuf_lines,
                obuf_lines=result.obuf_lines,
                success=False,
                error_message=f"Validation failed: {error_details}",
                operations_completed=result.operations_completed,
                return_code=result.return_code,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        return result
