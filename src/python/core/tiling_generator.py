"""
Independent Input-Output Tiling Generator

Generates strategies with independent input and output tile sizing while
maintaining O(1) dependency calculation through strict alignment constraints.

Key Features:
- Two cases: Sub-tiling (Case 1) and Super-tiling (Case 2)
- Boundary case mixing: allows one dimension at boundary (d=tile) while other is strict
- Perfect alignment with no remainder
- P/Q-level alignment for Case 2

Algorithm Overview:
- Works in pooled coordinate space (P_pooled, Q_pooled) for automatic pooling alignment
- tile_p = t_pooled × pool_height guarantees tile % pool == 0
- No explicit pooling validation needed - alignment is built into the iteration
"""

import math
from typing import List, Tuple

from .logging_config import get_logger

logger = get_logger(__name__)

from .tiling import (
    CNNLayerParams,
    StrategyDescriptor,
    TilingConfig,
)

# =============================================================================
# Constants: Strategy Types
# =============================================================================

# Why: Distinguish between two fundamentally different tiling approaches
# Case 1 uses multiple small inputs for one output (data reuse in compute)
# Case 2 uses one large input for multiple outputs (data reuse in memory)
CASE_TYPE_SUB_TILING = 1  # Multiple small input tiles → 1 output tile
CASE_TYPE_SUPER_TILING = 2  # 1 large input tile → multiple output tiles

# Display names for consistent UI/logging representation
CASE1_DISPLAY_NAME = "Case 1: Sub-tiling (N:1)"
CASE2_DISPLAY_NAME = "Case 2: Super-tiling (1:N)"
CASE1_SHORT_NAME = "Sub-tiling"
CASE2_SHORT_NAME = "Super-tiling"


def get_case_display_name(case_type: int, short: bool = False) -> str:
    """Get display name for case type."""
    if short:
        return CASE1_SHORT_NAME if case_type == 1 else CASE2_SHORT_NAME
    return CASE1_DISPLAY_NAME if case_type == 1 else CASE2_DISPLAY_NAME


# =============================================================================
# Helper Functions: Divisor Calculation
# =============================================================================


def _get_divisors(n: int) -> List[int]:
    """
    Get all divisors of n in ascending order.

    Algorithm: O(√n) by checking pairs (i, n/i) simultaneously.

    Args:
        n: Number to find divisors for

    Returns:
        List of divisors [1, d1, d2, ..., n] in ascending order
    """
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)




def _calculate_input_tile_size(cnn_params: CNNLayerParams, d_p: int, d_q: int) -> Tuple[int, int]:
    """
    Calculate input tile size in H×W coordinate system

    Why: Input tile size must account for the receptive field required
    to produce d_p×d_q output pixels. The formula derives from the
    convolution sliding window operation with stride.

    Formula:
        input_h = R + (d_p - 1) × stride
        input_w = S + (d_q - 1) × stride

    Where:
        R, S: Kernel dimensions
        d_p, d_q: Output dimensions in P×Q space
        stride: Convolution stride

    Args:
        cnn_params: CNN layer parameters (R, S, stride)
        d_p: Output tile height (in P dimension)
        d_q: Output tile width (in Q dimension)

    Returns:
        (input_h, input_w): Input tile size in H×W coordinate system
    """
    input_h = cnn_params.R + (d_p - 1) * cnn_params.stride
    input_w = cnn_params.S + (d_q - 1) * cnn_params.stride
    return input_h, input_w


def _calculate_tile_counts(
    cnn_params: CNNLayerParams,
    tile_p: int,
    tile_q: int,
    input_tile_p: int,
    input_tile_q: int,
) -> dict:
    """
    Calculate directional tile counts for output and input

    Why: We need separate counts for vertical (P) and horizontal (Q)
    dimensions to correctly calculate memory access patterns and
    validate alignment constraints.

    Args:
        cnn_params: CNN layer parameters (P, Q dimensions)
        tile_p: Output tile height
        tile_q: Output tile width
        input_tile_p: Input tile height (in P dimension)
        input_tile_q: Input tile width (in Q dimension)

    Returns:
        Dictionary with tile counts:
            - num_output_tiles_p: Output tiles in P dimension
            - num_output_tiles_q: Output tiles in Q dimension
            - output_tile_count: Total output tiles
            - num_input_tiles_p: Input tiles in P dimension
            - num_input_tiles_q: Input tiles in Q dimension
    """
    num_output_tiles_p = cnn_params.P // tile_p
    num_output_tiles_q = cnn_params.Q // tile_q
    output_tile_count = num_output_tiles_p * num_output_tiles_q

    num_input_tiles_p = cnn_params.P // input_tile_p
    num_input_tiles_q = cnn_params.Q // input_tile_q

    return {
        "num_output_tiles_p": num_output_tiles_p,
        "num_output_tiles_q": num_output_tiles_q,
        "output_tile_count": output_tile_count,
        "num_input_tiles_p": num_input_tiles_p,
        "num_input_tiles_q": num_input_tiles_q,
    }


def _calculate_case1_input_tile_count(
    tile_p: int, tile_q: int, d_p: int, d_q: int, output_tile_count: int
) -> int:
    """
    Calculate total input tile count for Case 1 (Sub-tiling)

    Why: In Case 1, each output tile requires multiple input tiles.
    The number of input tiles per output is (tile_p/d_p) × (tile_q/d_q),
    which derives from the spatial subdivision of the output tile.

    Formula:
        inputs_per_output = (tile_p / d_p) × (tile_q / d_q)
        total_input_tiles = output_tile_count × inputs_per_output

    Args:
        tile_p: Output tile height
        tile_q: Output tile width
        d_p: Input→output mapping in P dimension
        d_q: Input→output mapping in Q dimension
        output_tile_count: Total number of output tiles

    Returns:
        Total number of input tile loads required
    """
    inputs_per_output = (tile_p // d_p) * (tile_q // d_q)
    return output_tile_count * inputs_per_output


def _calculate_case2_input_tile_count(n_p: int, n_q: int, output_tile_count: int) -> int:
    """
    Calculate total input tile count for Case 2 (Super-tiling)

    Why: In Case 2, one input tile produces multiple output tiles.
    The number of outputs per input is n_p × n_q, where n represents
    the multiplication factor in each dimension.

    Formula:
        outputs_per_input = n_p × n_q
        total_input_tiles = output_tile_count / outputs_per_input

    Args:
        n_p: Multiplication factor in P dimension (d_p = tile_p × n_p)
        n_q: Multiplication factor in Q dimension (d_q = tile_q × n_q)
        output_tile_count: Total number of output tiles

    Returns:
        Total number of input tile loads required
    """
    outputs_per_input = n_p * n_q
    return output_tile_count // outputs_per_input


# =============================================================================
# Helper Functions: TilingConfig Creation
# =============================================================================


def _create_tiling_config(
    tile_p: int,
    tile_q: int,
    input_h: int,
    input_w: int,
    input_tile_p: int,
    input_tile_q: int,
    tile_counts: dict,
    input_tile_count: int,
    strategy_id: int,
    description: str,
    case_type: int,
) -> TilingConfig:
    """
    Create TilingConfig object with all calculated parameters

    Why: Centralize TilingConfig creation to eliminate duplication between
    Case 1 and Case 2, ensuring consistent parameter passing and reducing
    the chance of parameter mismatch errors.

    Args:
        tile_p: Output tile height
        tile_q: Output tile width
        input_h: Input tile height (H×W system)
        input_w: Input tile width (H×W system)
        input_tile_p: Input tile height (P×Q system)
        input_tile_q: Input tile width (P×Q system)
        tile_counts: Dictionary with directional tile counts
        input_tile_count: Total input tile loads
        strategy_id: Unique strategy identifier
        description: Human-readable strategy description
        case_type: CASE_TYPE_SUB_TILING or CASE_TYPE_SUPER_TILING

    Returns:
        Configured TilingConfig object
    """
    return TilingConfig(
        output_tile_p=tile_p,
        output_tile_q=tile_q,
        input_tile_h=input_h,
        input_tile_w=input_w,
        input_tile_p=input_tile_p,
        input_tile_q=input_tile_q,
        num_output_tiles_p=tile_counts["num_output_tiles_p"],
        num_output_tiles_q=tile_counts["num_output_tiles_q"],
        num_input_tiles_p=tile_counts["num_input_tiles_p"],
        num_input_tiles_q=tile_counts["num_input_tiles_q"],
        output_tile_count=tile_counts["output_tile_count"],
        input_tile_count=input_tile_count,
        strategy_id=strategy_id,
        description=description,
        case_type=case_type,
    )


def _create_strategy_descriptor(
    strategy_id: int,
    description: str,
    cnn_params: CNNLayerParams,
    tiling_config: TilingConfig,
) -> StrategyDescriptor:
    """
    Create StrategyDescriptor with all required components

    Args:
        strategy_id: Unique strategy identifier
        description: Human-readable strategy description
        cnn_params: CNN layer parameters
        tiling_config: Configured tiling parameters

    Returns:
        Complete StrategyDescriptor ready for simulation
    """
    return StrategyDescriptor(
        strategy_id=strategy_id,
        description=description,
        cnn_params=cnn_params,
        tiling_config=tiling_config,
    )


# =============================================================================
# Main Generator Class
# =============================================================================


class IndependentTilingGenerator:
    """
    Generator for independent input-output tiling strategies

    Why: Independent tiling allows input and output tiles to have different
    sizes, enabling exploration of a richer strategy space compared to
    aligned tiling where input_tile_size = output_tile_size × factor.
    This increases design space by 3-5x while maintaining O(1) dependency
    calculation through strict alignment constraints.
    """

    def __init__(
        self,
        cnn_params: CNNLayerParams,
        starting_strategy_id: int = 0,
    ):
        """
        Initialize the generator

        Args:
            cnn_params: CNN layer parameters (H, W, C, P, Q, M, R, S, stride, pooling)
            starting_strategy_id: Starting ID for generated strategies (default: 0)

        Raises:
            ValueError: If CNN parameters are invalid
        """
        self.cnn_params = cnn_params
        self.starting_id = starting_strategy_id

        # Validate CNN parameters immediately to fail fast
        errors = self.cnn_params.validate()
        if errors:
            raise ValueError("Invalid CNN parameters:\n" + "\n".join(errors))

    def _validate_pooling_alignment(self, output_tile_p: int, output_tile_q: int) -> bool:
        """
        Check if output tile satisfies pooling alignment

        Why: Pooling operations require output tiles to be multiples of
        the pooling window size. If violated, pooling would cross tile
        boundaries, breaking the O(1) dependency calculation guarantee.

        Constraint:
            output_tile_p % pool_height == 0
            output_tile_q % pool_width == 0

        Args:
            output_tile_p: Output tile height
            output_tile_q: Output tile width

        Returns:
            True if alignment satisfied, False otherwise
        """
        return (
            output_tile_p % self.cnn_params.pool_height == 0
            and output_tile_q % self.cnn_params.pool_width == 0
        )

    def _generate_case1_strategies(
        self, tile_p: int, tile_q: int, strategy_id: int, verbose: bool = False
    ) -> Tuple[List[StrategyDescriptor], int]:
        """
        Generate Case 1 (Sub-tiling) strategies

        Why: Case 1 explores data reuse in the compute dimension by using
        multiple small input tiles to produce one output tile. This is
        optimal for input-stationary architectures where we want to maximize
        input data reuse across multiple MACs.

        Case 1 Definition (Extended):
            - Multiple small input tiles → 1 output tile
            - Condition: d_p ≤ tile_p AND d_q ≤ tile_q AND (d_p < tile_p OR d_q < tile_q)
            - Allows boundary cases where one dimension equals tile size
            - Input tiles per output: (tile_p/d_p) × (tile_q/d_q)

        Args:
            tile_p: Output tile height (must divide P)
            tile_q: Output tile width (must divide Q)
            strategy_id: Starting strategy ID for this batch
            verbose: Print generation progress for debugging

        Returns:
            (strategies, next_strategy_id): List of generated strategies and next available ID
        """
        strategies = []

        # Get all divisors including tile_p, tile_q for boundary cases
        # Why: Now allowing d ≤ tile to support boundary mixing (e.g., d_p < tile_p, d_q = tile_q)
        d_p_candidates = _get_divisors(tile_p)
        d_q_candidates = _get_divisors(tile_q)

        if verbose:
            logger.info("  Case 1 (Sub-tiling):")
            logger.info(f"    d_p candidates: {d_p_candidates}")
            logger.info(f"    d_q candidates: {d_q_candidates}")

        for d_p in d_p_candidates:
            for d_q in d_q_candidates:
                # Skip if both dimensions are at boundary (d_p = tile_p AND d_q = tile_q)
                # Why: This case is handled by Case 2 to avoid duplication
                if d_p == tile_p and d_q == tile_q:
                    continue

                # Calculate input tile size in H×W coordinate system
                input_h, input_w = _calculate_input_tile_size(self.cnn_params, d_p, d_q)

                # Input tiles in output coordinate system (P×Q)
                # Why: d_p and d_q are already in output P×Q dimensions
                input_tile_p = d_p
                input_tile_q = d_q

                # Calculate directional tile counts
                tile_counts = _calculate_tile_counts(
                    self.cnn_params, tile_p, tile_q, input_tile_p, input_tile_q
                )

                # Calculate memory access counts for Case 1 (Sub-tiling)
                input_tile_count = _calculate_case1_input_tile_count(
                    tile_p, tile_q, d_p, d_q, tile_counts["output_tile_count"]
                )

                # Format description
                description = f"Case1: {d_p}×{d_q}→tile, input={input_h}×{input_w}, tiles={tile_counts['output_tile_count']}"

                # Create tiling config
                tiling = _create_tiling_config(
                    tile_p=tile_p,
                    tile_q=tile_q,
                    input_h=input_h,
                    input_w=input_w,
                    input_tile_p=input_tile_p,
                    input_tile_q=input_tile_q,
                    tile_counts=tile_counts,
                    input_tile_count=input_tile_count,
                    strategy_id=strategy_id,
                    description=description,
                    case_type=CASE_TYPE_SUB_TILING,
                )

                # Create strategy descriptor
                descriptor = _create_strategy_descriptor(
                    strategy_id=strategy_id,
                    description=description,
                    cnn_params=self.cnn_params,
                    tiling_config=tiling,
                )

                if verbose:
                    logger.debug(f"    Strategy {strategy_id}: d=({d_p},{d_q}) → input=({input_h},{input_w})")

                strategies.append(descriptor)
                strategy_id += 1

        return strategies, strategy_id

    def _generate_case2_strategies(
        self, tile_p: int, tile_q: int, strategy_id: int, verbose: bool = False
    ) -> Tuple[List[StrategyDescriptor], int]:
        """
        Generate Case 2 (Super-tiling) strategies

        Why: Case 2 explores data reuse in the memory dimension by using
        one large input tile to produce multiple output tiles. This is
        optimal for output-stationary architectures where we want to maximize
        input data reuse across multiple output computations.

        Case 2 Definition:
            - 1 large input tile → multiple output tiles
            - Condition: d_p ≥ tile_p AND d_q ≥ tile_q AND d_p|P AND d_q|Q
            - Output tiles per input: n_p × n_q where d = tile × n

        Args:
            tile_p: Output tile height (must divide P)
            tile_q: Output tile width (must divide Q)
            strategy_id: Starting strategy ID for this batch
            verbose: Print generation progress for debugging

        Returns:
            (strategies, next_strategy_id): List of generated strategies and next available ID
        """
        strategies = []

        # Calculate k = P/tile, Q/tile
        # Why: k represents how many tiles fit in each dimension
        k_p = self.cnn_params.P // tile_p
        k_q = self.cnn_params.Q // tile_q

        # Get valid n values (divisors of k)
        # Why: n must divide k to ensure d = tile×n divides P
        # This guarantees P/Q-level alignment (no remainder)
        n_p_list = _get_divisors(k_p)
        n_q_list = _get_divisors(k_q)

        if verbose:
            logger.info("  Case 2 (Super-tiling):")
            logger.info(f"    k_p = {self.cnn_params.P}/{tile_p} = {k_p}")
            logger.info(f"    k_q = {self.cnn_params.Q}/{tile_q} = {k_q}")
            logger.info(f"    n_p candidates: {n_p_list}")
            logger.info(f"    n_q candidates: {n_q_list}")

        for n_p in n_p_list:
            d_p = tile_p * n_p

            for n_q in n_q_list:
                d_q = tile_q * n_q

                # Calculate input tile size in H×W coordinate system
                input_h, input_w = _calculate_input_tile_size(self.cnn_params, d_p, d_q)

                # Input tiles in output coordinate system (P×Q)
                # Why: d_p and d_q are already in output P×Q dimensions
                input_tile_p = d_p
                input_tile_q = d_q

                # Calculate directional tile counts
                tile_counts = _calculate_tile_counts(
                    self.cnn_params, tile_p, tile_q, input_tile_p, input_tile_q
                )

                # Calculate memory access counts for Case 2 (Super-tiling)
                input_tile_count = _calculate_case2_input_tile_count(
                    n_p, n_q, tile_counts["output_tile_count"]
                )

                # Format description
                description = f"Case2: 1→{n_p}×{n_q}tiles, input={input_h}×{input_w}, tiles={tile_counts['output_tile_count']}"

                # Create tiling config
                tiling = _create_tiling_config(
                    tile_p=tile_p,
                    tile_q=tile_q,
                    input_h=input_h,
                    input_w=input_w,
                    input_tile_p=input_tile_p,
                    input_tile_q=input_tile_q,
                    tile_counts=tile_counts,
                    input_tile_count=input_tile_count,
                    strategy_id=strategy_id,
                    description=description,
                    case_type=CASE_TYPE_SUPER_TILING,
                )

                # Create strategy descriptor
                descriptor = _create_strategy_descriptor(
                    strategy_id=strategy_id,
                    description=description,
                    cnn_params=self.cnn_params,
                    tiling_config=tiling,
                )

                if verbose:
                    logger.debug(f"    Strategy {strategy_id}: n=({n_p},{n_q}) → d=({d_p},{d_q}) → input=({input_h},{input_w})")

                strategies.append(descriptor)
                strategy_id += 1

        return strategies, strategy_id

    def generate_strategies_iterator(self, verbose: bool = False):
        """
        Iterator-based strategy generation (memory efficient)

        Why: Iterator pattern enables streaming characterization without
        holding all strategies in memory. Critical for large networks
        where strategy count can reach 1000+ per layer. Memory usage:
        O(1) instead of O(n×strategy_size).

        Yields strategies one at a time instead of building full list.
        Use this for production DSE workflows.

        Algorithm: Work in pooled space (P_pooled, Q_pooled) to automatically
        satisfy pooling alignment constraints. This eliminates the need for
        explicit pooling validation checks.

        Args:
            verbose: Print generation progress for debugging

        Yields:
            StrategyDescriptor: One valid strategy at a time
        """
        strategy_id = self.starting_id

        # Work in pooled coordinate space for automatic pooling alignment
        # Why: By iterating over divisors of P_pooled/Q_pooled and multiplying
        # by pool dimensions, all generated tile sizes automatically satisfy
        # the pooling alignment constraint (tile % pool == 0)
        P_pooled = self.cnn_params.P // self.cnn_params.pool_height
        Q_pooled = self.cnn_params.Q // self.cnn_params.pool_width

        P_pooled_divisors = _get_divisors(P_pooled)
        Q_pooled_divisors = _get_divisors(Q_pooled)

        if verbose:
            p = self.cnn_params
            logger.info(f"CNN Layer: {p.H}×{p.W}×{p.C} → {p.P}×{p.Q}×{p.M}")
            logger.info(f"Pooling: {p.pool_height}×{p.pool_width}")
            logger.info(f"P_pooled = {p.P}/{p.pool_height} = {P_pooled}")
            logger.info(f"Q_pooled = {p.Q}/{p.pool_width} = {Q_pooled}")
            logger.info(f"P_pooled divisors ({len(P_pooled_divisors)}): {P_pooled_divisors}")
            logger.info(f"Q_pooled divisors ({len(Q_pooled_divisors)}): {Q_pooled_divisors}")

        # Iterate over divisors of pooled dimensions
        # tile = t_pooled × pool guarantees pooling alignment
        for t_p_pooled in P_pooled_divisors:
            tile_p = t_p_pooled * self.cnn_params.pool_height

            for t_q_pooled in Q_pooled_divisors:
                tile_q = t_q_pooled * self.cnn_params.pool_width

                if verbose:
                    logger.info("")
                    logger.info(f"Output tile: ({tile_p}, {tile_q}) [pooled: ({t_p_pooled}, {t_q_pooled})]")

                # Generate Case 1 strategies for this output tile
                case1_strategies, strategy_id = self._generate_case1_strategies(
                    tile_p, tile_q, strategy_id, verbose
                )

                for strategy in case1_strategies:
                    yield strategy

                # Generate Case 2 strategies for this output tile
                case2_strategies, strategy_id = self._generate_case2_strategies(
                    tile_p, tile_q, strategy_id, verbose
                )

                for strategy in case2_strategies:
                    yield strategy

    def generate_all_strategies(self, verbose: bool = False) -> List[StrategyDescriptor]:
        """
        Generate all valid independent tiling strategies

        Why: Convenience method for testing and analysis where you need
        the complete strategy list. Returns all strategies at once which
        uses more memory than the iterator but enables operations like
        len(), indexing, and batch processing.

        For production DSE with large networks, prefer the iterator method
        to avoid memory exhaustion.

        Args:
            verbose: Print generation progress for debugging

        Returns:
            List of all valid strategy descriptors
        """
        strategies = list(self.generate_strategies_iterator(verbose=verbose))

        if verbose:
            logger.info("")
            logger.info(f"Generated {len(strategies)} valid strategies")

        return strategies
