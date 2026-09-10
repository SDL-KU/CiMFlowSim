#!/usr/bin/env python3
"""
Benchmark Database for eFlowSim

Manages central SQLite database for aggregating results from multiple
preset × network combinations.

Database Schema:
- presets: Hardware preset metadata
- networks: Network configuration metadata
- combinations: Preset × network combination tracking
- strategy_generation: Strategy generation metrics
- simulation_results: Detailed simulation metrics per strategy
- optimization_results: Optimized solutions per objective
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Query Results
# ============================================================================


@dataclass
class PresetRecord:
    """Preset database record"""

    id: int
    name: str
    technology_node: Optional[str]
    cim_memory_type: Optional[str]
    description: Optional[str]
    config_hash: str
    created_at: str


@dataclass
class NetworkRecord:
    """Network database record"""

    id: int
    name: str
    num_layers: Optional[int]
    total_params: Optional[int]
    description: Optional[str]
    config_hash: str
    created_at: str


@dataclass
class CombinationRecord:
    """Combination database record"""

    id: int
    preset_id: int
    network_id: int
    workspace_path: str
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]


# ============================================================================
# Benchmark Database
# ============================================================================


class SweepDatabase:
    """
    Central database for benchmark results

    Responsibilities:
    1. Schema creation and migration
    2. Preset/Network metadata storage
    3. Combination tracking
    4. Result aggregation from individual workspace DBs
    5. Query interface for analysis
    """

    def __init__(self, db_path: Path):
        """
        Initialize benchmark database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create connection
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

        # Create schema
        self._create_schema()

        logger.info(f"Initialized benchmark database: {db_path}")

    def _create_schema(self):
        """Create database schema if not exists"""
        cursor = self.conn.cursor()

        # Presets table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS presets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                technology_node TEXT,
                cim_memory_type TEXT,
                description TEXT,
                config_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Networks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS networks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                num_layers INTEGER,
                total_params INTEGER,
                description TEXT,
                config_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Combinations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS combinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preset_id INTEGER NOT NULL REFERENCES presets(id),
                network_id INTEGER NOT NULL REFERENCES networks(id),
                workspace_path TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                UNIQUE(preset_id, network_id)
            )
        """
        )

        # Strategy generation table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_generation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preset_id INTEGER REFERENCES presets(id),
                network_id INTEGER REFERENCES networks(id),
                combination_id INTEGER REFERENCES combinations(id),
                num_strategies INTEGER,
                generation_time_sec REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Simulation results table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preset_id INTEGER REFERENCES presets(id),
                network_id INTEGER REFERENCES networks(id),
                combination_id INTEGER REFERENCES combinations(id),
                layer_idx INTEGER,
                strategy_id TEXT,

                latency_ns REAL NOT NULL,
                energy_nj REAL NOT NULL,
                area_mm2 REAL NOT NULL,
                edp REAL NOT NULL,

                compute_energy_nj REAL,
                memory_energy_nj REAL,
                communication_energy_nj REAL,
                static_energy_nj REAL,

                ibuf_area_mm2 REAL,
                obuf_area_mm2 REAL,
                cim_area_mm2 REAL,

                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(preset_id, network_id, layer_idx, strategy_id)
            )
        """
        )

        # Optimization results table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS optimization_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preset_id INTEGER REFERENCES presets(id),
                network_id INTEGER REFERENCES networks(id),
                combination_id INTEGER REFERENCES combinations(id),
                objective TEXT NOT NULL,

                total_latency_ns REAL,
                total_energy_nj REAL,
                total_area_mm2 REAL,
                total_edp REAL,

                selected_strategies TEXT,

                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(preset_id, network_id, objective)
            )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_combinations_status ON combinations(status)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sim_preset_network ON simulation_results(preset_id, network_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sim_strategy ON simulation_results(strategy_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_opt_objective ON optimization_results(objective)"
        )

        self.conn.commit()
        logger.debug("Database schema created/verified")

    def _calculate_hash(self, config_data: Dict) -> str:
        """Compute SHA256 hash of configuration for reproducibility"""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    # ========================================================================
    # Preset Operations
    # ========================================================================

    def get_or_create_preset(
        self,
        name: str,
        config_data: Dict,
        technology_node: Optional[str] = None,
        cim_memory_type: Optional[str] = None,
        description: Optional[str] = None,
    ) -> int:
        """
        Get existing preset ID or create new preset

        Args:
            name: Preset name
            config_data: Full configuration dictionary
            technology_node: Technology node (e.g., "28nm")
            cim_memory_type: CIM memory type (e.g., "SRAM")
            description: Optional description

        Returns:
            Preset ID
        """
        cursor = self.conn.cursor()
        config_hash = self._calculate_hash(config_data)

        # Check if exists
        cursor.execute("SELECT id FROM presets WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return row[0]

        # Create new
        cursor.execute(
            """
            INSERT INTO presets (name, technology_node, cim_memory_type, description, config_hash)
            VALUES (?, ?, ?, ?, ?)
        """,
            (name, technology_node, cim_memory_type, description, config_hash),
        )
        self.conn.commit()

        preset_id = cursor.lastrowid
        logger.debug(f"Created preset: {name} (id={preset_id})")
        return preset_id

    # ========================================================================
    # Network Operations
    # ========================================================================

    def get_or_create_network(
        self,
        name: str,
        config_data: Dict,
        num_layers: Optional[int] = None,
        total_params: Optional[int] = None,
        description: Optional[str] = None,
    ) -> int:
        """
        Get existing network ID or create new network

        Args:
            name: Network name
            config_data: Full configuration dictionary
            num_layers: Number of layers
            total_params: Total parameters
            description: Optional description

        Returns:
            Network ID
        """
        cursor = self.conn.cursor()
        config_hash = self._calculate_hash(config_data)

        # Check if exists
        cursor.execute("SELECT id FROM networks WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return row[0]

        # Create new
        cursor.execute(
            """
            INSERT INTO networks (name, num_layers, total_params, description, config_hash)
            VALUES (?, ?, ?, ?, ?)
        """,
            (name, num_layers, total_params, description, config_hash),
        )
        self.conn.commit()

        network_id = cursor.lastrowid
        logger.debug(f"Created network: {name} (id={network_id})")
        return network_id

    # ========================================================================
    # Combination Operations
    # ========================================================================

    def upsert_combination(
        self,
        preset_id: int,
        network_id: int,
        workspace_path: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> int:
        """
        Create or update combination record

        Args:
            preset_id: Preset ID
            network_id: Network ID
            workspace_path: Relative workspace path
            status: Combination status
            error_message: Optional error message

        Returns:
            Combination ID
        """
        cursor = self.conn.cursor()

        # Check if exists
        cursor.execute(
            "SELECT id FROM combinations WHERE preset_id = ? AND network_id = ?",
            (preset_id, network_id),
        )
        row = cursor.fetchone()

        if row:
            # Update existing
            combination_id = row[0]
            updates = {"status": status, "workspace_path": workspace_path}

            if status == "running":
                updates["started_at"] = datetime.now().isoformat()
            elif status in ("completed", "failed", "partial"):
                updates["completed_at"] = datetime.now().isoformat()

            if error_message:
                updates["error_message"] = error_message

            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
            cursor.execute(
                f"UPDATE combinations SET {set_clause} WHERE id = ?",
                (*updates.values(), combination_id),
            )
        else:
            # Create new
            started_at = datetime.now().isoformat() if status == "running" else None
            completed_at = (
                datetime.now().isoformat() if status in ("completed", "failed", "partial") else None
            )

            cursor.execute(
                """
                INSERT INTO combinations
                (preset_id, network_id, workspace_path, status, started_at, completed_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    preset_id,
                    network_id,
                    workspace_path,
                    status,
                    started_at,
                    completed_at,
                    error_message,
                ),
            )
            combination_id = cursor.lastrowid

        self.conn.commit()
        logger.debug(f"Upserted combination (id={combination_id}, status={status})")
        return combination_id

    def get_combination(self, preset_id: int, network_id: int) -> Optional[CombinationRecord]:
        """Get combination by preset and network IDs"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM combinations WHERE preset_id = ? AND network_id = ?",
            (preset_id, network_id),
        )
        row = cursor.fetchone()
        if row:
            return CombinationRecord(**dict(row))
        return None

    def list_combinations(self, status: Optional[str] = None) -> List[CombinationRecord]:
        """
        List combinations, optionally filtered by status

        Args:
            status: Optional status filter ("pending", "running", "completed", "failed")

        Returns:
            List of CombinationRecord
        """
        cursor = self.conn.cursor()
        if status:
            cursor.execute(
                "SELECT * FROM combinations WHERE status = ? ORDER BY id",
                (status,),
            )
        else:
            cursor.execute("SELECT * FROM combinations ORDER BY id")

        return [CombinationRecord(**dict(row)) for row in cursor.fetchall()]

    # ========================================================================
    # Result Aggregation
    # ========================================================================

    def insert_simulation_result(
        self,
        preset_id: int,
        network_id: int,
        combination_id: int,
        layer_idx: int,
        strategy_id: str,
        latency_ns: float,
        energy_nj: float,
        area_mm2: float,
        edp: float,
        compute_energy_nj: Optional[float] = None,
        memory_energy_nj: Optional[float] = None,
        communication_energy_nj: Optional[float] = None,
        static_energy_nj: Optional[float] = None,
        ibuf_area_mm2: Optional[float] = None,
        obuf_area_mm2: Optional[float] = None,
        cim_area_mm2: Optional[float] = None,
    ):
        """Insert simulation result (called from aggregate_to_central_db)"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO simulation_results
            (preset_id, network_id, combination_id, layer_idx, strategy_id,
             latency_ns, energy_nj, area_mm2, edp,
             compute_energy_nj, memory_energy_nj, communication_energy_nj, static_energy_nj,
             ibuf_area_mm2, obuf_area_mm2, cim_area_mm2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                preset_id,
                network_id,
                combination_id,
                layer_idx,
                strategy_id,
                latency_ns,
                energy_nj,
                area_mm2,
                edp,
                compute_energy_nj,
                memory_energy_nj,
                communication_energy_nj,
                static_energy_nj,
                ibuf_area_mm2,
                obuf_area_mm2,
                cim_area_mm2,
            ),
        )
        # NOTE: Commit is NOT called here for performance (batch insert pattern)
        # Caller must call commit_simulation_results() after all inserts

    def commit_simulation_results(self):
        """Commit all pending simulation result inserts (call after batch insert)"""
        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.debug("Closed database connection")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
