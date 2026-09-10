"""
Simulation Validation Framework
Validates that SystemC simulations execute correctly and meet expected operation counts.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ValidationResult:
    """Result of simulation validation"""

    errors: List[str]
    warnings: List[str]

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)"""
        return len(self.errors) == 0

    def summary(self) -> str:
        """Get summary string"""
        if self.is_valid():
            return "PASS"
        return f"FAIL ({len(self.errors)} errors, {len(self.warnings)} warnings)"


class SimulationValidator:
    """Validates SystemC simulation results"""

    def __init__(
        self, strategy_config: dict, simulation_result: Any, gantt_path: Optional[str] = None
    ):
        """
        Initialize validator

        Args:
            strategy_config: Strategy configuration dict with tiling_config and cnn_layer
            simulation_result: SystemCResult object from simulation
            gantt_path: Optional path to gantt_data.txt for dependency validation
        """
        self.config = strategy_config
        self.result = simulation_result
        self.gantt_path = gantt_path
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self, level: str = "basic") -> ValidationResult:
        """
        Perform all validation checks

        Args:
            level: 'basic' (operation counts only) or 'full' (includes gantt analysis)

        Returns:
            ValidationResult with errors and warnings
        """
        # Level 1: Operation completeness (always run)
        self.validate_completeness()

        # Level 2: Dependency validation (only if gantt available and level='full')
        if level == "full" and self.gantt_path:
            if os.path.exists(self.gantt_path):
                self.validate_dependencies()
            else:
                self.warnings.append("Gantt file not found, skipping dependency validation")

        return ValidationResult(self.errors, self.warnings)

    def validate_completeness(self) -> None:
        """Validate that all operations completed successfully"""
        expected = self._calculate_expected_operations()
        actual = self.result.operations_completed or {}

        for op_type in ["loads", "ibuf_reads", "cim_computes", "obuf_writes", "stores"]:
            exp_count = expected.get(op_type, 0)
            act_count = actual.get(op_type, 0)

            if act_count != exp_count:
                self.errors.append(f"{op_type}: expected {exp_count}, got {act_count}")

    def validate_dependencies(self) -> None:
        """Validate pipeline dependencies from gantt data"""
        gantt_ops = self._parse_gantt_data()

        if self._is_case1():
            violations = self._check_case1_dependencies(gantt_ops)
        else:
            violations = self._check_case2_dependencies(gantt_ops)

        if violations:
            self.errors.extend(violations[:5])  # First 5 only
            if len(violations) > 5:
                self.warnings.append(f"... and {len(violations) - 5} more dependency violations")

    def _calculate_expected_operations(self) -> Dict[str, int]:
        """Calculate expected operation counts from configuration"""
        cnn = self.config["cnn_layer"]
        tiling = self.config["tiling_config"]

        batch = cnn["batch_size"]
        tile_p = tiling["output_tile_p"]
        tile_q = tiling["output_tile_q"]
        total_tiles = tiling["output_tile_count"]
        # Pool dimensions are required fields (use 1 for no pooling)
        pool_h = cnn["pool_height"]
        pool_w = cnn["pool_width"]

        # Computes: batch × tiles × (tile_p × tile_q)
        total_output_pixels = batch * total_tiles * tile_p * tile_q
        total_computes = total_output_pixels

        # OBUF_WRITE: computes / pooling_ratio
        total_obuf_writes = total_computes // (pool_h * pool_w)

        # LOADS calculation (both cases use input_tile_count from tiling config)
        # Case 1: input_tile_count = tiles × inputs_per_output
        # Case 2: input_tile_count = tiles / outputs_per_input
        total_loads = tiling.get("input_tile_count", total_tiles) * batch

        # STORE: Always use output_tile_count from tiling config
        # Independent Tiling provides output_tile_count which already accounts for Case 1/2
        # Case 1: output_tile_count = total_tiles (1 store per output tile)
        # Case 2: output_tile_count < total_tiles (1 store per input tile)
        total_stores = tiling["output_tile_count"] * batch

        return {
            "loads": total_loads,
            "ibuf_reads": total_computes,
            "cim_computes": total_computes,
            "obuf_writes": total_obuf_writes,
            "stores": total_stores,
        }

    def _is_case1(self) -> bool:
        """Check if this is Case 1 (sub-tiling)"""
        tiling_config = self.config.get("tiling_config", {})
        case_type = tiling_config.get("case_type", 2)
        return case_type == 1

    def _parse_gantt_data(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Parse gantt_data.txt file (supports new format with key=value pairs)"""
        operations: Dict[str, Dict[int, Dict[str, float]]] = {}

        if self.gantt_path is None:
            return operations

        with open(self.gantt_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and dependency lines
                if not line or line.startswith("#") or "depends_on:" in line:
                    continue

                parts = line.split()
                # Accept 4 or more parts (new format has key=value pairs after timing)
                if len(parts) >= 4:
                    op_type = parts[0]
                    try:
                        op_id = int(parts[1])
                        start = float(parts[2])
                        end = float(parts[3])
                    except ValueError:
                        continue
                    if op_type not in operations:
                        operations[op_type] = {}
                    operations[op_type][op_id] = {"start": start, "end": end}

        return operations

    def _check_case1_dependencies(self, gantt_ops: Dict) -> List[str]:
        """Check Case 1: 1:1 OBUF_WRITE→STORE mapping"""
        violations = []

        obuf_writes = gantt_ops.get("OBUF_WRITE", {})
        stores = gantt_ops.get("STORE", {})

        for store_id, store_op in stores.items():
            # Case 1: 1:1 mapping
            obuf_id = store_id
            if obuf_id in obuf_writes:
                obuf_op = obuf_writes[obuf_id]
                if store_op["start"] < obuf_op["end"]:
                    violations.append(
                        f"STORE {store_id} starts at {store_op['start']:.1f}ns "
                        f"before OBUF_WRITE {obuf_id} ends at {obuf_op['end']:.1f}ns"
                    )

        return violations

    def _check_case2_dependencies(self, gantt_ops: Dict) -> List[str]:
        """Check Case 2: 4:1 OBUF_WRITE→STORE mapping (with 2×2 pooling)"""
        violations = []

        obuf_writes = gantt_ops.get("OBUF_WRITE", {})
        stores = gantt_ops.get("STORE", {})

        # Calculate pooling ratio
        tile_p = self.config["tiling_config"]["output_tile_p"]
        tile_q = self.config["tiling_config"]["output_tile_q"]
        # Pool dimensions are required fields (no fallback)
        pool_h = self.config["cnn_layer"]["pool_height"]
        pool_w = self.config["cnn_layer"]["pool_width"]

        pooled_pixels = (tile_p // pool_h) * (tile_q // pool_w)
        obuf_per_store = (tile_p * tile_q) // pooled_pixels

        for store_id, store_op in stores.items():
            # Case 2: Wait for last OBUF_WRITE in group
            last_obuf_id = (store_id + 1) * obuf_per_store - 1
            if last_obuf_id in obuf_writes:
                obuf_op = obuf_writes[last_obuf_id]
                if store_op["start"] < obuf_op["end"]:
                    violations.append(
                        f"STORE {store_id} starts at {store_op['start']:.1f}ns "
                        f"before OBUF_WRITE {last_obuf_id} ends at {obuf_op['end']:.1f}ns"
                    )

        return violations
