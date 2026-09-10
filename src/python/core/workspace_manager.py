"""
Workspace Manager for CiMFlowSim

Manages workspace creation, structure, and file organization for characterization experiments.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import orjson

from .exceptions import ConfigurationError, FileOperationError, build_file_error_context


def _extract_technology_node(hardware_config: dict, config_path: Path) -> str:
    """
    Extract technology node from hardware config.

    Supports two formats:
    - hardware.technology.node (wrapped format)
    - technology.node (flat format)

    Args:
        hardware_config: Hardware configuration dictionary
        config_path: Path to config file (for error messages)

    Returns:
        Technology node string (e.g., '28nm', '16nm')

    Raises:
        ConfigurationError: If technology node is missing
    """
    # Try wrapped format first: hardware.technology.node
    if "hardware" in hardware_config:
        tech_section = hardware_config["hardware"].get("technology", {})
    else:
        tech_section = hardware_config.get("technology", {})

    if not tech_section:
        raise ConfigurationError(
            f"Missing 'technology' section in hardware config: {config_path}",
            suggestions=["Add 'hardware.technology.node' or 'technology.node' field"],
        )

    if "node" not in tech_section:
        raise ConfigurationError(
            f"Missing 'node' in technology section: {config_path}",
            suggestions=["Add technology node (e.g., '28nm', '16nm')"],
        )

    return tech_section["node"]


class WorkspaceManager:
    """
    Manages experiment workspaces with separated configurations.

    Workspace structure:
        results/{network}_{hardware}/
        ├── workspace_metadata.json
        ├── hardware_config.json
        ├── network_config.json
        ├── layers/
        │   ├── L0.json
        │   └── L1.json
        ├── strategies/
        │   ├── L0_S0.json
        │   ├── L0_S1.json
        │   ├── L1_S0.json
        │   └── ...
        ├── simulations/
        │   ├── L0_S0.json
        │   └── ...
        ├── strategies.db
        └── strategies.csv
    """

    def __init__(self, workspace_path: Path):
        """
        Initialize workspace manager.

        Args:
            workspace_path: Path to workspace directory
        """
        # Convert to absolute path to support Ray parallel execution
        self.workspace_path = Path(workspace_path).resolve()
        self.layers_dir = self.workspace_path / "layers"
        self.strategies_dir = self.workspace_path / "strategies"
        self.simulations_dir = self.workspace_path / "simulations"

    @classmethod
    def create_workspace(
        cls,
        network_name: str,
        hardware_name: str,
        network_config_path: Path,
        hardware_config_path: Path,
        base_dir: Path = Path("workspaces"),
        force: bool = False,
    ) -> "WorkspaceManager":
        """
        Create a new workspace for an experiment.

        Args:
            network_name: Name of the network
            hardware_name: Name of the hardware configuration
            network_config_path: Path to network config file
            hardware_config_path: Path to hardware config file
            base_dir: Base directory for results
            force: If True, overwrite existing workspace

        Returns:
            WorkspaceManager instance for the created workspace

        Raises:
            FileExistsError: If workspace exists and force=False
        """
        workspace_name = f"{network_name}_{hardware_name}"
        workspace_path = base_dir / workspace_name

        # Check if workspace exists
        if workspace_path.exists():
            if not force:
                raise FileExistsError(
                    f"Workspace already exists: {workspace_path}\n" "Use --force to overwrite"
                )
            # Remove existing workspace
            shutil.rmtree(workspace_path)

        # Create directory structure
        workspace_path.mkdir(parents=True, exist_ok=True)
        (workspace_path / "layers").mkdir(exist_ok=True)
        (workspace_path / "strategies").mkdir(exist_ok=True)
        (workspace_path / "simulations").mkdir(exist_ok=True)

        # Copy configuration files
        shutil.copy(network_config_path, workspace_path / "network_config.json")
        shutil.copy(hardware_config_path, workspace_path / "hardware_config.json")

        # Load configs to extract metadata
        with open(network_config_path) as f:
            network_config = orjson.loads(f.read())
        with open(hardware_config_path) as f:
            hardware_config = orjson.loads(f.read())

        # Count layers
        num_layers = len(network_config.get("layers", []))

        # Extract technology node (required field)
        tech_node = _extract_technology_node(hardware_config, hardware_config_path)

        # Create metadata
        metadata = {
            "experiment_name": workspace_name,
            "created_at": datetime.now().isoformat(),
            "network": {
                "name": network_name,
                "source": str(network_config_path),
                "num_layers": num_layers,
            },
            "hardware": {
                "name": hardware_name,
                "source": str(hardware_config_path),
                "technology_node": tech_node,
            },
            "status": "initialized",
            "strategy_counts": {},
            "simulation_progress": {},
        }

        # Save metadata
        with open(workspace_path / "workspace_metadata.json", "w") as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode())

        return cls(workspace_path)

    def save_strategy(
        self, layer_idx: int, strategy_id: int, strategy_data: Dict[str, Any]
    ) -> Path:
        """
        Save a strategy configuration to workspace (input for simulation).

        Args:
            layer_idx: Layer index (0-based)
            strategy_id: Strategy ID
            strategy_data: Strategy configuration (must contain layer_idx)

        Returns:
            Path to the saved strategy file
        """
        # Extract tiling info for descriptive filename
        tiling_config = strategy_data.get("tiling_config", {})
        out_p = tiling_config.get("output_tile_p", 0)
        out_q = tiling_config.get("output_tile_q", 0)
        in_h = tiling_config.get("input_tile_h", 0)
        in_w = tiling_config.get("input_tile_w", 0)

        # Descriptive filename: L{idx}_S{id}_out{p}x{q}_in{h}x{w}.json
        filename = f"L{layer_idx}_S{strategy_id}_out{out_p}x{out_q}_in{in_h}x{in_w}.json"
        strategy_file = self.strategies_dir / filename

        # Add metadata
        strategy_with_metadata = {
            "layer_idx": layer_idx,
            "strategy_id": strategy_id,
            **strategy_data,
        }

        with open(strategy_file, "w") as f:
            f.write(orjson.dumps(strategy_with_metadata, option=orjson.OPT_INDENT_2).decode())

        return strategy_file

    def save_failed_simulation(
        self, layer_idx: int, strategy_id: int, error_message: str
    ) -> Path:
        """
        Save failed simulation information for debugging.

        Args:
            layer_idx: Layer index (0-based)
            strategy_id: Strategy ID
            error_message: Error message from simulation

        Returns:
            Path to the saved error log file
        """
        # Create failed simulations subdirectory
        failed_dir = self.simulations_dir / "failed"
        failed_dir.mkdir(parents=True, exist_ok=True)

        # Save error log
        error_file = failed_dir / f"L{layer_idx}_S{strategy_id}_error.txt"
        with open(error_file, "w") as f:
            f.write(f"Strategy: L{layer_idx}_S{strategy_id}\n")
            f.write("Status: FAILED\n")
            f.write("\nError:\n")
            f.write(error_message)

        return error_file

    def _update_metadata_field(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific field in workspace metadata.

        Args:
            section: Metadata section name (e.g., 'strategy_counts', 'simulation_progress')
            key: Key within the section
            value: Value to set
        """
        metadata = self.load_metadata()

        if section not in metadata:
            metadata[section] = {}

        metadata[section][key] = value

        metadata_path = self.workspace_path / "workspace_metadata.json"
        with open(metadata_path, "w") as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode())

    def update_strategy_count(self, layer_idx: int, count: int) -> None:
        """
        Update strategy count for a layer in metadata.

        Args:
            layer_idx: Layer index (0-based)
            count: Number of strategies generated
        """
        self._update_metadata_field("strategy_counts", f"L{layer_idx}", count)

    def load_layer_config(self, layer_idx: int) -> Dict[str, Any]:
        """
        Load layer configuration from workspace.

        Args:
            layer_idx: Layer index (0-based)

        Returns:
            Layer configuration dictionary

        Raises:
            FileNotFoundError: If layer configuration file doesn't exist
            ValueError: If layer configuration is empty or invalid
        """
        layer_file = self.layers_dir / f"L{layer_idx}.json"

        if not layer_file.exists():
            raise FileOperationError(
                f"Layer configuration file not found: {layer_file}",
                context=build_file_error_context(str(layer_file), "read"),
                suggestions=[
                    f"Ensure layer L{layer_idx} was generated in workspace",
                    "Run './efsim generate' command first",
                    "Check workspace path is correct",
                ],
            )

        with open(layer_file) as f:
            config = orjson.loads(f.read())

        if not config:
            raise ConfigurationError(
                f"Layer configuration is empty: {layer_file}",
                context={"file_path": str(layer_file), "layer_idx": layer_idx},
                suggestions=[
                    "Check that layer configuration was saved correctly",
                    "Verify JSON file is not corrupted",
                    "Re-generate layer configurations if needed",
                ],
            )

        return config

    def load_strategy_config(self, layer_idx: int, strategy_id: int) -> Dict[str, Any]:
        """
        Load strategy configuration from workspace.

        Args:
            layer_idx: Layer index (0-based)
            strategy_id: Strategy ID

        Returns:
            Strategy configuration dictionary

        Raises:
            FileOperationError: If strategy file doesn't exist
            ConfigurationError: If strategy configuration is empty or invalid
        """
        # Find the strategy file with tiling info in filename
        pattern = f"L{layer_idx}_S{strategy_id}_*.json"
        matches = list(self.strategies_dir.glob(pattern))

        if not matches:
            raise FileOperationError(
                f"Strategy file not found: L{layer_idx}_S{strategy_id}_*.json",
                context=build_file_error_context(str(self.strategies_dir / pattern), "read"),
                suggestions=[
                    f"Ensure strategy L{layer_idx}_S{strategy_id} was generated",
                    "Run './efsim generate' command first",
                ],
            )

        strategy_file = matches[0]  # Use first match

        with open(strategy_file) as f:
            config = orjson.loads(f.read())

        if not config:
            raise ConfigurationError(
                f"Strategy configuration is empty: {strategy_file}",
                context={"file_path": str(strategy_file), "layer_idx": layer_idx, "strategy_id": strategy_id},
                suggestions=["Re-generate strategies if needed"],
            )

        return config

    def load_hardware_config(self) -> Dict[str, Any]:
        """
        Load hardware configuration from workspace.

        Returns:
            Hardware configuration dictionary

        Raises:
            FileOperationError: If hardware configuration file doesn't exist
            ConfigurationError: If hardware configuration is empty or invalid
        """
        hardware_file = self.workspace_path / "hardware_config.json"

        if not hardware_file.exists():
            raise FileOperationError(
                f"Hardware configuration file not found: {hardware_file}",
                context=build_file_error_context(str(hardware_file), "read"),
                suggestions=[
                    "Ensure workspace was created properly",
                    "Run './efsim generate' to create workspace with hardware config",
                ],
            )

        with open(hardware_file) as f:
            config = orjson.loads(f.read())

        if not config:
            raise ConfigurationError(
                f"Hardware configuration is empty: {hardware_file}",
                context={"file_path": str(hardware_file)},
                suggestions=["Check that hardware config JSON is valid"],
            )

        return config

    def load_metadata(self) -> Dict[str, Any]:
        """
        Load workspace metadata.

        Returns:
            Metadata dictionary
        """
        metadata_file = self.workspace_path / "workspace_metadata.json"

        with open(metadata_file) as f:
            return orjson.loads(f.read())

    def list_strategies(self, layer_idx: int) -> List[int]:
        """
        List all strategy IDs for a layer.

        Args:
            layer_idx: Layer index (0-based)

        Returns:
            List of strategy IDs
        """
        if not self.strategies_dir.exists():
            return []

        strategy_ids = []
        # Match pattern: L{idx}_S{id}_*.json or L{idx}_S{id}.json
        pattern = f"L{layer_idx}_S*.json"
        for strategy_file in self.strategies_dir.glob(pattern):
            # Extract ID from filename: L0_S123_out2x2_in5x5.json -> 123
            # Split by '_S' and extract numeric part from second element
            parts = strategy_file.stem.split("_S")
            if len(parts) == 2:
                try:
                    # Extract only numeric part (handles both old and new formats)
                    # "123" -> 123 (old format)
                    # "123_out2x2_in5x5" -> 123 (new format)
                    id_part = parts[1].split("_")[0]  # Take first segment after _S
                    strategy_ids.append(int(id_part))
                except ValueError:
                    continue

        return sorted(strategy_ids)

    def get_database_path(self) -> Path:
        """
        Get path to strategies database.

        Returns:
            Path to strategies.db
        """
        return self.workspace_path / "strategies.db"
