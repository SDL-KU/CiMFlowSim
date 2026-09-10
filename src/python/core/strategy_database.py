"""
Performance Database for CNN Strategy Results

SQLite-based storage for layer-strategy performance metrics.
Supports efficient queries for optimization algorithms.
"""

import csv
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import orjson

from .logging_config import get_logger
from .tiling import LayerParams

logger = get_logger(__name__)


class StrategyDatabase:
    """SQLite database for strategy performance results"""

    def __init__(self, db_path: str = "results/characterization.db"):
        """
        Initialize database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._batch_conn = None  # Persistent connection for batch mode

        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize schema
        self._init_schema()

    def begin_batch(self):
        """Begin batch insert mode - keeps connection open until end_batch()

        Use this when inserting many rows to avoid per-row commit overhead.
        Each commit on NFS takes ~8ms, so 9000 rows = 72 seconds overhead.
        With batch mode: single commit at end = ~8ms total.
        """
        if self._batch_conn is not None:
            return  # Already in batch mode

        self._batch_conn = sqlite3.connect(self.db_path)
        self._batch_conn.row_factory = sqlite3.Row
        self._batch_conn.execute("PRAGMA synchronous = OFF")
        self._batch_conn.execute("PRAGMA journal_mode = MEMORY")
        self._batch_conn.execute("PRAGMA temp_store = MEMORY")

    def end_batch(self):
        """End batch insert mode - commits all pending changes"""
        if self._batch_conn is None:
            return

        try:
            self._batch_conn.commit()
        finally:
            self._batch_conn.close()
            self._batch_conn = None

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections"""
        conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name

        # Performance optimizations for batch operations
        conn.execute("PRAGMA synchronous = OFF")  # Faster writes (OK for benchmarks)
        conn.execute("PRAGMA journal_mode = MEMORY")  # Use memory for journal
        conn.execute("PRAGMA temp_store = MEMORY")  # Use memory for temp tables

        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _migrate_schema(self, cursor: sqlite3.Cursor) -> None:
        """Migrate old schema to add energy breakdown and area breakdown columns"""
        # Check if strategy_results table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_results'"
        )
        if not cursor.fetchone():
            return  # New database, no migration needed

        # Check if energy breakdown columns exist
        cursor.execute("PRAGMA table_info(strategy_results)")
        columns = {row[1] for row in cursor.fetchall()}

        new_columns = [
            "mac_energy_nj",
            "pooling_energy_nj",
            "activation_energy_nj",
            "sram_read_energy_nj",
            "sram_write_energy_nj",
            "dram_read_energy_nj",
            "dram_write_energy_nj",
            "communication_energy_nj",
            "static_energy_nj",
            "ibuf_area_mm2",
            "obuf_area_mm2",
            "cim_area_mm2",
        ]

        # Add missing columns
        for col in new_columns:
            if col not in columns:
                cursor.execute(f"ALTER TABLE strategy_results ADD COLUMN {col} REAL")
                logger.info(f"DB migration: added column '{col}'")

    def _init_schema(self) -> None:
        """Create database schema if not exists"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if migration needed
            self._migrate_schema(cursor)

            # Layers table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS layers (
                    layer_idx INTEGER PRIMARY KEY,
                    layer_type TEXT,
                    input_shape TEXT,
                    output_shape TEXT,
                    kernel_size TEXT,
                    stride INTEGER,
                    network_name TEXT
                )
            """
            )

            # Strategy results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_results (
                    -- Primary Keys
                    layer_idx INTEGER,
                    strategy_id INTEGER,

                    -- ================================================================
                    -- Core Performance Metrics (Top-level results)
                    -- ================================================================
                    latency_ns REAL,           -- Total execution time (nanoseconds)
                    energy_nj REAL,            -- Total energy consumption (nanojoules)
                    area_mm2 REAL,             -- Total hardware area (mm²)

                    -- ================================================================
                    -- Energy Breakdown (Detailed energy analysis)
                    -- ================================================================
                    -- Computation energy
                    mac_energy_nj REAL,        -- Multiply-accumulate energy
                    pooling_energy_nj REAL,    -- Pooling operation energy
                    activation_energy_nj REAL, -- Activation function energy

                    -- Memory access energy
                    sram_read_energy_nj REAL,  -- SRAM (IBUF/OBUF) read energy
                    sram_write_energy_nj REAL, -- SRAM (IBUF/OBUF) write energy
                    dram_read_energy_nj REAL,  -- DRAM (external) read energy
                    dram_write_energy_nj REAL, -- DRAM (external) write energy

                    -- Communication & static
                    communication_energy_nj REAL, -- On-chip interconnect energy
                    static_energy_nj REAL,        -- Static power consumption

                    -- ================================================================
                    -- Area Breakdown (Hardware resource allocation)
                    -- ================================================================
                    ibuf_area_mm2 REAL,        -- Input buffer area
                    obuf_area_mm2 REAL,        -- Output buffer area
                    cim_area_mm2 REAL,         -- Compute-in-memory area

                    -- ================================================================
                    -- Operation Counts (Computation workload)
                    -- ================================================================
                    mac_ops INTEGER,           -- Number of MAC operations
                    pooling_ops INTEGER,       -- Number of pooling operations
                    activation_ops INTEGER,    -- Number of activation operations
                    comparison_ops INTEGER,    -- Number of comparison operations
                    total_operations INTEGER,  -- Sum of all operations

                    -- ================================================================
                    -- Memory Access Counts (Data access patterns)
                    -- ================================================================
                    -- External memory (DRAM)
                    external_reads INTEGER,    -- DRAM read accesses
                    external_writes INTEGER,   -- DRAM write accesses

                    -- Input buffer (SRAM)
                    ibuf_reads INTEGER,        -- IBUF read accesses
                    ibuf_writes INTEGER,       -- IBUF write accesses

                    -- Output buffer (SRAM)
                    obuf_reads INTEGER,        -- OBUF read accesses
                    obuf_writes INTEGER,       -- OBUF write accesses

                    -- Weight buffer (SRAM)
                    weight_buf_reads INTEGER,  -- Weight buffer read accesses
                    weight_buf_writes INTEGER, -- Weight buffer write accesses

                    -- Compute-in-memory
                    cim_reads INTEGER,         -- CIM read accesses
                    cim_writes INTEGER,        -- CIM write accesses

                    -- Summary
                    total_memory_accesses INTEGER, -- Sum of all memory accesses

                    -- ================================================================
                    -- Data Movement (Bytes transferred between memory levels)
                    -- ================================================================
                    external_to_ibuf_bytes INTEGER, -- DRAM → IBUF data transfer
                    obuf_to_external_bytes INTEGER, -- OBUF → DRAM data transfer

                    -- ================================================================
                    -- Pipeline Metrics (Pipeline operation counts)
                    -- ================================================================
                    pipeline_loads INTEGER,         -- Number of load operations
                    pipeline_ibuf_reads INTEGER,    -- Number of IBUF read operations
                    pipeline_cim_computes INTEGER,  -- Number of CIM compute operations
                    pipeline_obuf_writes INTEGER,   -- Number of OBUF write operations
                    pipeline_stores INTEGER,        -- Number of store operations

                    -- ================================================================
                    -- Buffer Metrics (Memory utilization)
                    -- ================================================================
                    -- Buffer allocation (capacity)
                    ibuf_lines INTEGER,        -- IBUF allocated lines
                    obuf_lines INTEGER,        -- OBUF allocated lines

                    -- Buffer peak usage (actual utilization)
                    ibuf_peak_lines INTEGER,   -- IBUF peak usage
                    obuf_peak_lines INTEGER,   -- OBUF peak usage

                    -- ================================================================
                    -- Tiling Configuration (Strategy parameters)
                    -- ================================================================
                    input_tile_count INTEGER,  -- Number of input tiles
                    output_tile_count INTEGER, -- Number of output tiles
                    output_tile_p INTEGER,     -- Output tile height in P dimension
                    output_tile_q INTEGER,     -- Output tile width in Q dimension
                    input_tile_p INTEGER,      -- Input tile height in P dimension (for coupled detection)
                    input_tile_q INTEGER,      -- Input tile width in Q dimension (for coupled detection)
                    tiling_config TEXT,        -- Full tiling config (JSON)

                    -- ================================================================
                    -- Metadata
                    -- ================================================================
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    PRIMARY KEY (layer_idx, strategy_id),
                    FOREIGN KEY (layer_idx) REFERENCES layers(layer_idx)
                )
            """
            )

            # Covering indices for optimization queries (index-only scans)
            # These include all columns needed by optimizer to avoid table lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_latency_covering
                ON strategy_results(layer_idx, latency_ns, area_mm2, energy_nj,
                                    strategy_id, ibuf_lines, obuf_lines)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_area_covering
                ON strategy_results(layer_idx, area_mm2, latency_ns, energy_nj,
                                    strategy_id, ibuf_lines, obuf_lines)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_energy_covering
                ON strategy_results(layer_idx, energy_nj, latency_ns, area_mm2,
                                    strategy_id, ibuf_lines, obuf_lines)
            """
            )

            # Drop old single-column indices if they exist (replaced by covering indices)
            cursor.execute("DROP INDEX IF EXISTS idx_latency")
            cursor.execute("DROP INDEX IF EXISTS idx_area")
            cursor.execute("DROP INDEX IF EXISTS idx_energy")

            # Migration: Add input_tile_count and output_tile_count columns if they don't exist
            cursor.execute("PRAGMA table_info(strategy_results)")
            columns = [row[1] for row in cursor.fetchall()]

            if "input_tile_count" not in columns:
                cursor.execute(
                    """
                    ALTER TABLE strategy_results
                    ADD COLUMN input_tile_count INTEGER
                    """
                )

            if "output_tile_count" not in columns:
                cursor.execute(
                    """
                    ALTER TABLE strategy_results
                    ADD COLUMN output_tile_count INTEGER
                    """
                )

            # Migration: Add derived metrics (eap, buffer_eap, buffer_area_mm2)
            if "eap" not in columns:
                cursor.execute(
                    """
                    ALTER TABLE strategy_results
                    ADD COLUMN eap REAL
                    """
                )

            if "buffer_eap" not in columns:
                cursor.execute(
                    """
                    ALTER TABLE strategy_results
                    ADD COLUMN buffer_eap REAL
                    """
                )

            if "buffer_area_mm2" not in columns:
                cursor.execute(
                    """
                    ALTER TABLE strategy_results
                    ADD COLUMN buffer_area_mm2 REAL
                    """
                )

            # Migration: Add tiling dimension columns for coupled detection
            if "output_tile_p" not in columns:
                cursor.execute(
                    """
                    ALTER TABLE strategy_results
                    ADD COLUMN output_tile_p INTEGER
                    """
                )

            if "output_tile_q" not in columns:
                cursor.execute(
                    """
                    ALTER TABLE strategy_results
                    ADD COLUMN output_tile_q INTEGER
                    """
                )

            if "input_tile_p" not in columns:
                cursor.execute(
                    """
                    ALTER TABLE strategy_results
                    ADD COLUMN input_tile_p INTEGER
                    """
                )

            if "input_tile_q" not in columns:
                cursor.execute(
                    """
                    ALTER TABLE strategy_results
                    ADD COLUMN input_tile_q INTEGER
                    """
                )

    def insert_layer(
        self,
        layer_idx: int,
        layer_type: str,
        input_shape: Tuple[int, int, int],
        output_shape: Tuple[int, int, int],
        kernel_size: Tuple[int, int],
        stride: int,
        network_name: str = "unknown",
    ) -> None:
        """
        Insert layer metadata

        Args:
            layer_idx: Unique layer index (0, 1, 2, ...)
            layer_type: Type (conv2d, fc, etc.)
            input_shape: (H, W, C)
            output_shape: (P, Q, M)
            kernel_size: (R, S)
            stride: Convolution stride
            network_name: Network this layer belongs to
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO layers
                (layer_idx, layer_type, input_shape, output_shape, kernel_size, stride, network_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    layer_idx,
                    layer_type,
                    f"{input_shape[0]}x{input_shape[1]}x{input_shape[2]}",
                    f"{output_shape[0]}x{output_shape[1]}x{output_shape[2]}",
                    f"{kernel_size[0]}x{kernel_size[1]}",
                    stride,
                    network_name,
                ),
            )

    def insert_strategy_result(
        self,
        layer_idx: int,
        strategy_id: int,
        latency_ns: float,
        area_mm2: float,
        energy_nj: float,
        ibuf_lines: int,
        obuf_lines: int,
        tiling_config: Dict,
        input_tile_count: Optional[int] = None,
        output_tile_count: Optional[int] = None,
        energy_breakdown: Optional[Dict[str, float]] = None,
        ibuf_area_mm2: Optional[float] = None,
        obuf_area_mm2: Optional[float] = None,
        cim_area_mm2: Optional[float] = None,
    ) -> None:
        """Insert strategy performance result with energy and area breakdown

        NOTE: If begin_batch() was called, uses persistent connection without commit.
              Otherwise, creates new connection and commits immediately.
        """
        # Use batch connection if available (no per-row commit)
        if self._batch_conn is not None:
            cursor = self._batch_conn.cursor()
            self._insert_strategy_result_impl(
                cursor, layer_idx, strategy_id, latency_ns, area_mm2, energy_nj,
                ibuf_lines, obuf_lines, tiling_config, input_tile_count,
                output_tile_count, energy_breakdown, ibuf_area_mm2, obuf_area_mm2,
                cim_area_mm2
            )
            return

        # Normal mode: create connection and commit
        with self._get_connection() as conn:
            cursor = conn.cursor()
            self._insert_strategy_result_impl(
                cursor, layer_idx, strategy_id, latency_ns, area_mm2, energy_nj,
                ibuf_lines, obuf_lines, tiling_config, input_tile_count,
                output_tile_count, energy_breakdown, ibuf_area_mm2, obuf_area_mm2,
                cim_area_mm2
            )

    def _insert_strategy_result_impl(
        self,
        cursor,
        layer_idx: int,
        strategy_id: int,
        latency_ns: float,
        area_mm2: float,
        energy_nj: float,
        ibuf_lines: int,
        obuf_lines: int,
        tiling_config: Dict,
        input_tile_count: Optional[int],
        output_tile_count: Optional[int],
        energy_breakdown: Optional[Dict[str, float]],
        ibuf_area_mm2: Optional[float],
        obuf_area_mm2: Optional[float],
        cim_area_mm2: Optional[float],
    ) -> None:
        """Internal implementation of insert_strategy_result"""
        # Extract energy breakdown if provided
        if energy_breakdown:
            mac_energy = energy_breakdown.get("mac_energy_nj", 0.0)
            pooling_energy = energy_breakdown.get("pooling_energy_nj", 0.0)
            activation_energy = energy_breakdown.get("activation_energy_nj", 0.0)
            sram_read_energy = energy_breakdown.get("sram_read_energy_nj", 0.0)
            sram_write_energy = energy_breakdown.get("sram_write_energy_nj", 0.0)
            dram_read_energy = energy_breakdown.get("dram_read_energy_nj", 0.0)
            dram_write_energy = energy_breakdown.get("dram_write_energy_nj", 0.0)
            communication_energy = energy_breakdown.get("communication_energy_nj", 0.0)
            static_energy = energy_breakdown.get("static_energy_nj", 0.0)
        else:
            mac_energy = pooling_energy = activation_energy = None
            sram_read_energy = sram_write_energy = dram_read_energy = dram_write_energy = None
            communication_energy = static_energy = None

        # Calculate derived metrics
        buffer_area = (ibuf_area_mm2 or 0.0) + (obuf_area_mm2 or 0.0)
        eap = area_mm2 * energy_nj
        buffer_eap = buffer_area * energy_nj

        # Extract tiling dimensions for coupled detection
        # Handle both dict and JSON string formats
        if isinstance(tiling_config, str):
            tiling_dict = orjson.loads(tiling_config)
        else:
            tiling_dict = tiling_config
        output_tile_p = tiling_dict.get("output_tile_p")
        output_tile_q = tiling_dict.get("output_tile_q")
        input_tile_p = tiling_dict.get("input_tile_p")
        input_tile_q = tiling_dict.get("input_tile_q")

        cursor.execute(
            """
            INSERT OR REPLACE INTO strategy_results
            (layer_idx, strategy_id, latency_ns, area_mm2, energy_nj,
             mac_energy_nj, pooling_energy_nj, activation_energy_nj,
             sram_read_energy_nj, sram_write_energy_nj,
             dram_read_energy_nj, dram_write_energy_nj,
             communication_energy_nj, static_energy_nj,
             ibuf_area_mm2, obuf_area_mm2, cim_area_mm2,
             ibuf_lines, obuf_lines, input_tile_count, output_tile_count,
             output_tile_p, output_tile_q, input_tile_p, input_tile_q,
             tiling_config, buffer_area_mm2, eap, buffer_eap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                layer_idx,
                strategy_id,
                latency_ns,
                area_mm2,
                energy_nj,
                mac_energy,
                pooling_energy,
                activation_energy,
                sram_read_energy,
                sram_write_energy,
                dram_read_energy,
                dram_write_energy,
                communication_energy,
                static_energy,
                ibuf_area_mm2,
                obuf_area_mm2,
                cim_area_mm2,
                ibuf_lines,
                obuf_lines,
                input_tile_count,
                output_tile_count,
                output_tile_p,
                output_tile_q,
                input_tile_p,
                input_tile_q,
                orjson.dumps(tiling_config).decode(),
                buffer_area,
                eap,
                buffer_eap,
            ),
        )

    def _get_layer_params(self, layer_idx: int) -> LayerParams:
        """Load layer parameters from workspace layer file.

        Args:
            layer_idx: Layer index (0, 1, 2, ...)

        Returns:
            LayerParams with P, Q, H, W values

        Raises:
            FileNotFoundError: If layer file not found
            KeyError: If required params missing from layer file
        """
        # Database path is workspaces/<name>/strategies.db
        # Layer files are at workspaces/<name>/layers/L{index}.json
        workspace_path = Path(self.db_path).parent
        layer_file = workspace_path / "layers" / f"L{layer_idx}.json"

        if not layer_file.exists():
            raise FileNotFoundError(
                f"Layer file not found: {layer_file}. "
                f"Make sure layer files exist."
            )

        with open(layer_file) as f:
            config = json.load(f)

        params = config["params"]
        return LayerParams(
            P=int(params["P"]),
            Q=int(params["Q"]),
            H=int(params["H"]),
            W=int(params["W"]),
        )

    def get_strategies(
        self,
        layer_idx: int,
        strategy_filter: Optional[Callable[[Dict, LayerParams], bool]] = None,
    ) -> List[Dict]:
        """
        Get strategies for a layer, optionally filtered by tile properties.

        Args:
            layer_idx: Layer identifier
            strategy_filter: Optional filter function that takes (strategy, layer_params)
                           and returns True if strategy should be included.
                           Use StrategyFilter.create_filter() to create filter functions.

        Returns:
            List of strategy result dictionaries (all strategies if no filter)

        Example:
            >>> from strategy_filter import StrategyFilter
            >>> # Get only whole_tile and minimal_tile strategies
            >>> filter_fn = StrategyFilter.create_filter(["whole_tile", "minimal_tile"])
            >>> strategies = db.get_strategies("conv1", strategy_filter=filter_fn)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT strategy_id, latency_ns, area_mm2, energy_nj,
                       ibuf_lines, obuf_lines, tiling_config,
                       ibuf_area_mm2, obuf_area_mm2, cim_area_mm2,
                       buffer_area_mm2, eap, buffer_eap
                FROM strategy_results
                WHERE layer_idx = ?
                ORDER BY strategy_id
            """,
                (layer_idx,),
            )

            results = []
            for row in cursor.fetchall():
                # Parse tiling_config - handle both string and bytes
                raw_tiling = row["tiling_config"]
                if isinstance(raw_tiling, dict):
                    # Already parsed (shouldn't happen but handle it)
                    parsed_tiling = raw_tiling
                elif isinstance(raw_tiling, str):
                    # JSON string - use json.loads for compatibility
                    # Handle potential double-encoding (data was JSON-encoded twice)
                    parsed_tiling = json.loads(raw_tiling)

                    # If result is still a string, it was double-encoded - decode again
                    if isinstance(parsed_tiling, str):
                        parsed_tiling = json.loads(parsed_tiling)
                elif isinstance(raw_tiling, bytes):
                    # Bytes - use orjson.loads
                    parsed_tiling = orjson.loads(raw_tiling)
                else:
                    raise TypeError(f"Unexpected tiling_config type: {type(raw_tiling)}")

                results.append(
                    {
                        "strategy_id": row["strategy_id"],
                        "latency_ns": row["latency_ns"],
                        "area_mm2": row["area_mm2"],
                        "energy_nj": row["energy_nj"],
                        "ibuf_lines": row["ibuf_lines"],
                        "obuf_lines": row["obuf_lines"],
                        "tiling_config": parsed_tiling,
                        "ibuf_area_mm2": row["ibuf_area_mm2"],
                        "obuf_area_mm2": row["obuf_area_mm2"],
                        "cim_area_mm2": row["cim_area_mm2"],
                        "buffer_area_mm2": row["buffer_area_mm2"],
                        "eap": row["eap"],
                        "buffer_eap": row["buffer_eap"],
                    }
                )

            # Apply filter if provided
            if strategy_filter is None:
                return results

            layer_params = self._get_layer_params(layer_idx)
            filtered = [s for s in results if strategy_filter(s, layer_params)]
            return filtered

    def get_strategy(self, layer_idx: int, strategy_id: int) -> Optional[Dict]:
        """
        Get specific strategy result

        Args:
            layer_idx: Layer identifier
            strategy_id: Strategy identifier

        Returns:
            Strategy result dictionary or None
        """
        strategies = self.get_strategies(layer_idx)
        for s in strategies:
            if s["strategy_id"] == strategy_id:
                return s
        return None

    def get_all_layers(self) -> List[str]:
        """
        Get all layer IDs in database

        Returns:
            List of layer IDs
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT layer_idx FROM layers ORDER BY layer_idx")

            return [row["layer_idx"] for row in cursor.fetchall()]

    def get_completed_strategies(self, layer_idx: int) -> List[int]:
        """
        Get list of completed strategy IDs for a layer (for resume functionality)

        Args:
            layer_idx: Layer identifier

        Returns:
            List of strategy IDs that have been simulated
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT strategy_id
                FROM strategy_results
                WHERE layer_idx = ?
                ORDER BY strategy_id
            """,
                (layer_idx,),
            )

            return [row["strategy_id"] for row in cursor.fetchall()]

    def has_strategy(self, layer_idx: int, strategy_id: int) -> bool:
        """
        Check if a specific strategy has been simulated

        Args:
            layer_idx: Layer identifier
            strategy_id: Strategy identifier

        Returns:
            True if strategy exists in database
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT 1
                FROM strategy_results
                WHERE layer_idx = ? AND strategy_id = ?
                LIMIT 1
            """,
                (layer_idx, strategy_id),
            )

            return cursor.fetchone() is not None

    def export_to_csv(self, output_path: str) -> None:
        """
        Export all results to CSV

        Args:
            output_path: Output CSV file path
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT layer_idx, strategy_id, latency_ns, area_mm2, energy_nj,
                       ibuf_lines, obuf_lines, input_tile_count, output_tile_count, tiling_config
                FROM strategy_results
                ORDER BY layer_idx, strategy_id
            """
            )

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(
                    [
                        "layer_idx",
                        "strategy_id",
                        "latency_ns",
                        "area_mm2",
                        "energy_nj",
                        "ibuf_lines",
                        "obuf_lines",
                        "input_tile_count",
                        "output_tile_count",
                        "output_tile_p",
                        "output_tile_q",
                        "input_tile_h",
                        "input_tile_w",
                        "num_output_tiles_p",
                        "num_output_tiles_q",
                        "output_tile_count",
                    ]
                )

                # Data
                for row in cursor.fetchall():
                    # Parse tiling config (handle potential double-encoding)
                    tiling_str = row["tiling_config"]
                    if isinstance(tiling_str, str):
                        tiling = orjson.loads(tiling_str)
                        # Handle double-encoded JSON
                        if isinstance(tiling, str):
                            tiling = orjson.loads(tiling)
                    else:
                        tiling = tiling_str

                    writer.writerow(
                        [
                            row["layer_idx"],
                            row["strategy_id"],
                            row["latency_ns"],
                            row["area_mm2"],
                            row["energy_nj"],
                            row["ibuf_lines"],
                            row["obuf_lines"],
                            row["input_tile_count"],
                            row["output_tile_count"],
                            tiling["output_tile_p"],
                            tiling["output_tile_q"],
                            tiling["input_tile_h"],
                            tiling["input_tile_w"],
                            tiling["num_output_tiles_p"],
                            tiling["num_output_tiles_q"],
                            tiling["output_tile_count"],
                        ]
                    )

    def get_statistics(self) -> Dict:
        """
        Get database statistics

        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total strategies
            cursor.execute("SELECT COUNT(*) as count FROM strategy_results")
            total_strategies = cursor.fetchone()["count"]

            # Total layers
            cursor.execute("SELECT COUNT(*) as count FROM layers")
            total_layers = cursor.fetchone()["count"]

            # Per-layer counts
            cursor.execute(
                """
                SELECT layer_idx, COUNT(*) as count
                FROM strategy_results
                GROUP BY layer_idx
            """
            )
            per_layer = {row["layer_idx"]: row["count"] for row in cursor.fetchall()}

            return {
                "total_strategies": total_strategies,
                "total_layers": total_layers,
                "strategies_per_layer": per_layer,
            }

    def clear(self) -> None:
        """Clear all data from database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM strategy_results")
            cursor.execute("DELETE FROM layers")

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"StrategyDatabase(path='{self.db_path}', "
            f"layers={stats['total_layers']}, "
            f"strategies={stats['total_strategies']})"
        )
