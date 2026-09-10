"""
Tiling Configuration Data Structures for Aligned Tiling Strategy Framework

This module defines the core data structures for representing:
- CNN layer parameters
- Tiling configurations with alignment constraints
- Hardware constraints for validation
- Strategy descriptors for JSON serialization
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LayerParams:
    """Layer parameters for database queries.

    Attributes:
        P: Output feature map height
        Q: Output feature map width
        H: Input feature map height
        W: Input feature map width
    """

    P: int
    Q: int
    H: int
    W: int


@dataclass
class CNNLayerParams:
    """CNN layer parameters for strategy generation"""

    # Input dimensions
    H: int  # Input height
    W: int  # Input width
    C: int  # Input channels

    # Kernel dimensions
    R: int  # Kernel height
    S: int  # Kernel width
    M: int  # Output channels (number of filters)

    # Convolution parameters
    stride: int  # Stride

    # Batch parameters
    batch_size: int

    # Pooling parameters
    pool_height: int
    pool_width: int

    # Bitwidth parameters
    input_bitwidth: int
    output_bitwidth: int

    # Derived output dimensions (calculated from input)
    P: Optional[int] = None  # Output height before pooling
    Q: Optional[int] = None  # Output width before pooling
    P_pooled: Optional[int] = None  # Output height after pooling
    Q_pooled: Optional[int] = None  # Output width after pooling

    def __post_init__(self):
        """Calculate derived dimensions if not provided"""
        # Ensure valid stride (prevent division by zero)
        if self.stride <= 0:
            self.stride = 1

        # Ensure valid pooling dimensions (prevent division by zero)
        if self.pool_height <= 0:
            self.pool_height = 1
        if self.pool_width <= 0:
            self.pool_width = 1

        # Calculate output dimensions
        if self.P is None:
            self.P = (self.H - self.R) // self.stride + 1
        if self.Q is None:
            self.Q = (self.W - self.S) // self.stride + 1
        if self.P_pooled is None:
            self.P_pooled = self.P // self.pool_height
        if self.Q_pooled is None:
            self.Q_pooled = self.Q // self.pool_width

    def validate(self) -> List[str]:
        """Validate CNN layer parameters"""
        errors = []

        # Dimension checks
        if self.H <= 0 or self.W <= 0 or self.C <= 0:
            errors.append(f"Invalid input dimensions: H={self.H}, W={self.W}, C={self.C}")

        if self.R <= 0 or self.S <= 0 or self.M <= 0:
            errors.append(f"Invalid kernel dimensions: R={self.R}, S={self.S}, M={self.M}")

        if self.R > self.H or self.S > self.W:
            errors.append(f"Kernel size ({self.R}×{self.S}) exceeds input size ({self.H}×{self.W})")

        # Stride check
        if self.stride <= 0:
            errors.append(f"Invalid stride: {self.stride}")

        # Batch size check
        if self.batch_size <= 0:
            errors.append(f"Invalid batch_size: {self.batch_size}")

        # Pooling checks
        if self.pool_height <= 0 or self.pool_width <= 0:
            errors.append(f"Invalid pooling dimensions: {self.pool_height}×{self.pool_width}")

        if self.P % self.pool_height != 0:
            errors.append(
                f"Output height P={self.P} not divisible by pool_height={self.pool_height}"
            )

        if self.Q % self.pool_width != 0:
            errors.append(f"Output width Q={self.Q} not divisible by pool_width={self.pool_width}")

        # Bitwidth checks
        if self.input_bitwidth <= 0 or self.input_bitwidth > 32:
            errors.append(f"Invalid input_bitwidth: {self.input_bitwidth}")

        if self.output_bitwidth <= 0 or self.output_bitwidth > 32:
            errors.append(f"Invalid output_bitwidth: {self.output_bitwidth}")

        return errors

    @classmethod
    def from_dict(cls, data: Dict, batch_size: int | None = None) -> "CNNLayerParams":
        """Create CNNLayerParams from dictionary.

        Args:
            data: Dictionary with layer parameters
            batch_size: Optional batch_size override (useful when batch_size
                        is stored at network level, not in layer params)

        Required fields: H, W, C, R, S, M
        Optional fields (with defaults):
            - stride: 1
            - batch_size: from argument or data (required in one of them)
            - pool_height/pool_width: 1
            - input_bitwidth/output_bitwidth: 8
            - P, Q, P_pooled, Q_pooled: calculated if not provided
        """
        # batch_size: argument > data > error
        resolved_batch_size = batch_size or data.get("batch_size")
        if resolved_batch_size is None:
            raise ValueError(
                "batch_size is required: provide via argument or include in data dict"
            )

        return cls(
            H=data["H"],
            W=data["W"],
            C=data["C"],
            R=data["R"],
            S=data["S"],
            M=data["M"],
            stride=data.get("stride", 1),
            batch_size=resolved_batch_size,
            pool_height=data.get("pool_height", 1),
            pool_width=data.get("pool_width", 1),
            input_bitwidth=data.get("input_bitwidth", 8),
            output_bitwidth=data.get("output_bitwidth", 8),
            P=data.get("P"),  # calculated in __post_init__ if None
            Q=data.get("Q"),
            P_pooled=data.get("P_pooled"),
            Q_pooled=data.get("Q_pooled"),
        )


@dataclass
class TilingConfig:
    """Tiling configuration with alignment constraints"""

    # ========== Tile Dimensions ==========
    # Output tiles (P×Q coordinate system)
    output_tile_p: int
    output_tile_q: int

    # Input tiles (H×W coordinate system)
    input_tile_h: int
    input_tile_w: int

    # Input tiles mapped to output coordinate system (P×Q)
    input_tile_p: int  # Input tile in output P dimension
    input_tile_q: int  # Input tile in output Q dimension

    # ========== Directional Tile Counts ==========
    num_output_tiles_p: int  # Number of output tiles in P direction (renamed from num_tiles_p)
    num_output_tiles_q: int  # Number of output tiles in Q direction (renamed from num_tiles_q)
    num_input_tiles_p: int  # Number of input tiles in P direction
    num_input_tiles_q: int  # Number of input tiles in Q direction

    # ========== Total Tile Counts ==========
    output_tile_count: int  # Total output tiles (per batch)
    input_tile_count: int  # Total input tiles (per batch)

    # ========== Strategy Metadata ==========
    strategy_id: int  # Unique strategy ID (0-based)
    description: str

    # Case type: 1 for sub-tiling (Case 1), 2 for super-tiling (Case 2)
    case_type: int = 2  # Default to Case 2

    # ========== Total Operation Counts (across all batches) ==========
    # These are pre-calculated total counts including batch_size
    # C++ simulator can use these directly without multiplication
    total_loads: Optional[int] = None  # input_tile_count * batch_size
    total_ibuf_reads: Optional[int] = None  # output_tile_count * pixels_per_tile * batch_size
    total_cim_computes: Optional[int] = None  # Same as total_ibuf_reads
    total_obuf_writes: Optional[int] = (
        None  # output_tile_count * pooled_pixels_per_tile * batch_size
    )
    total_stores: Optional[int] = None  # output_tile_count * batch_size

    def to_dict(self) -> Dict:
        """
        Convert tiling parameters to dictionary for JSON serialization.

        Returns only the tiling parameters (12 Phase 3 fields), excluding
        strategy metadata (strategy_id, description).

        Returns:
            Dictionary with 12 tiling configuration fields

        Example:
            >>> config = TilingConfig(output_tile_p=2, output_tile_q=2, ...)
            >>> config.to_dict()
            {
                'output_tile_p': 2,
                'output_tile_q': 2,
                'input_tile_h': 5,
                'input_tile_w': 5,
                'input_tile_p': 1,
                'input_tile_q': 1,
                'num_output_tiles_p': 14,
                'num_output_tiles_q': 14,
                'num_input_tiles_p': 28,
                'num_input_tiles_q': 28,
                'output_tile_count': 196,
                'input_tile_count': 784,
                'case_type': 1
            }
        """
        result = {
            "output_tile_p": self.output_tile_p,
            "output_tile_q": self.output_tile_q,
            "input_tile_h": self.input_tile_h,
            "input_tile_w": self.input_tile_w,
            "input_tile_p": self.input_tile_p,
            "input_tile_q": self.input_tile_q,
            "num_output_tiles_p": self.num_output_tiles_p,
            "num_output_tiles_q": self.num_output_tiles_q,
            "num_input_tiles_p": self.num_input_tiles_p,
            "num_input_tiles_q": self.num_input_tiles_q,
            "output_tile_count": self.output_tile_count,
            "input_tile_count": self.input_tile_count,
            "case_type": self.case_type,
        }

        # Include total operation counts if available (Phase 3)
        if self.total_loads is not None:
            result["total_loads"] = self.total_loads
        if self.total_ibuf_reads is not None:
            result["total_ibuf_reads"] = self.total_ibuf_reads
        if self.total_cim_computes is not None:
            result["total_cim_computes"] = self.total_cim_computes
        if self.total_obuf_writes is not None:
            result["total_obuf_writes"] = self.total_obuf_writes
        if self.total_stores is not None:
            result["total_stores"] = self.total_stores

        return result


@dataclass
class StrategyDescriptor:
    """Complete strategy descriptor for a tiling strategy"""

    strategy_id: int
    description: str
    cnn_params: CNNLayerParams
    tiling_config: TilingConfig
