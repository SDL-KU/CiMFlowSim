"""
Cross-Layer Pattern Analyzer

Extracts tiling patterns based on relative tile sizes and scores them
across all layers to identify universally good strategies.

Key concepts:
- Pattern: A ratio-based description of tiling (independent of absolute dimensions)
- Cross-layer score: How well a pattern performs across ALL layers
- Network Pareto presence: % of Pareto-optimal network combinations containing this pattern

Scoring approaches:
1. Per-layer Pareto rank: Average rank within each layer (original)
2. Network Pareto presence rate: % of Pareto-optimal combinations containing this pattern
   - Uses 7 objectives from pareto_sampling: latency vs energy/area/EAP variants
   - More meaningful for cross-layer optimization
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import defaultdict
import math

# 7 network-level objectives (same as pareto_sampling.py)
NETWORK_OBJECTIVES = [
    ("latency_ns", "energy_nj", "Latency vs Energy"),
    ("latency_ns", "buffer_area_mm2", "Latency vs Buffer Area"),
    ("latency_ns", "sum_area_mm2", "Latency vs Sum Area"),
    ("latency_ns", "peak_area_mm2", "Latency vs Peak Area"),
    ("latency_ns", "buffer_eap", "Latency vs Buffer EAP"),
    ("latency_ns", "sum_eap", "Latency vs Sum EAP"),
    ("latency_ns", "peak_eap", "Latency vs Peak EAP"),
]


@dataclass
class TilingPattern:
    """A ratio-based tiling pattern that can be applied to any layer."""

    # Output tile ratios (tile_size / full_dimension)
    output_p_ratio: float  # output_tile_p / P
    output_q_ratio: float  # output_tile_q / Q

    # Input tile ratios (tile_size / full_input_dimension)
    input_p_ratio: float   # input_tile_p / (P + R - 1)
    input_q_ratio: float   # input_tile_q / (Q + S - 1)

    # Derived properties
    is_coupled: bool = False  # input_tile == output_tile (considering kernel)
    is_full_output: bool = False  # output_tile == full dimension

    # Pattern category (for human readability)
    category: str = ""

    def __post_init__(self):
        # Determine if coupled (input covers exactly what output needs)
        # For conv: input_tile = output_tile + kernel - 1
        # Coupled means: input_tile_p == output_tile_p (after accounting for kernel)
        # We check if ratios are approximately equal
        self.is_coupled = (
            abs(self.input_p_ratio - self.output_p_ratio) < 0.05 and
            abs(self.input_q_ratio - self.output_q_ratio) < 0.05
        )

        # Full output means tile covers the entire dimension
        self.is_full_output = (
            self.output_p_ratio > 0.95 and self.output_q_ratio > 0.95
        )

        # Assign category
        self.category = self._categorize()

    def _categorize(self) -> str:
        """Categorize the pattern into human-readable labels."""
        if self.is_full_output:
            return "full"
        elif self.is_coupled:
            if self.output_p_ratio <= 0.125:
                return "coupled_minimal"
            elif self.output_p_ratio <= 0.25:
                return "coupled_quarter"
            elif self.output_p_ratio <= 0.5:
                return "coupled_half"
            else:
                return "coupled_large"
        else:
            # Super-tiling or sub-tiling
            if self.input_p_ratio > self.output_p_ratio:
                return "super_tiling"
            else:
                return "sub_tiling"

    @property
    def pattern_id(self) -> str:
        """Generate a unique pattern ID based on quantized ratios."""
        # Quantize ratios to avoid floating point issues
        # Extended granularity: 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, min
        def quantize(r: float) -> str:
            if r >= 0.75:
                return "1"
            elif r >= 0.35:
                return "1/2"
            elif r >= 0.175:
                return "1/4"
            elif r >= 0.085:
                return "1/8"
            elif r >= 0.04:
                return "1/16"
            elif r >= 0.02:
                return "1/32"
            elif r >= 0.01:
                return "1/64"
            elif r >= 0.005:
                return "1/128"
            else:
                return "min"

        out_p = quantize(self.output_p_ratio)
        out_q = quantize(self.output_q_ratio)
        in_p = quantize(self.input_p_ratio)
        in_q = quantize(self.input_q_ratio)

        return f"out_{out_p}x{out_q}_in_{in_p}x{in_q}"


@dataclass
class PatternPerformance:
    """Performance metrics for a pattern on a specific layer."""
    layer_idx: int
    strategy_id: int
    latency_ns: float
    energy_nj: float
    area_mm2: float
    buffer_eap: float = 0.0
    pareto_rank: int = 0
    efficiency_score: float = 0.0


@dataclass
class CrossLayerPattern:
    """A pattern with cross-layer performance statistics."""
    pattern: TilingPattern
    pattern_id: str

    # Performance across layers
    performances: list[PatternPerformance] = field(default_factory=list)

    # Cross-layer statistics (per-layer Pareto)
    num_layers: int = 0
    avg_pareto_rank: float = 0.0
    pareto_optimal_ratio: float = 0.0  # % of layers where rank == 1
    avg_efficiency: float = 0.0
    consistency_score: float = 0.0  # Low variance = high consistency

    # Network-level Pareto presence rate (% of Pareto combinations containing this pattern)
    # Keys: objective pair names (e.g., "latency_ns vs energy_nj")
    network_pareto_presence: dict[str, float] = field(default_factory=dict)
    avg_network_pareto_presence: float = 0.0  # Average across all 7 objectives

    # Overall cross-layer score
    overall_score: float = 0.0

    def compute_statistics(self):
        """Compute cross-layer statistics from performances."""
        if not self.performances:
            return

        self.num_layers = len(set(p.layer_idx for p in self.performances))

        ranks = [p.pareto_rank for p in self.performances]
        efficiencies = [p.efficiency_score for p in self.performances]

        self.avg_pareto_rank = sum(ranks) / len(ranks) if ranks else 0
        self.pareto_optimal_ratio = sum(1 for r in ranks if r == 1) / len(ranks) if ranks else 0
        self.avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0

        # Consistency: inverse of rank variance (normalized)
        if len(ranks) > 1:
            mean_rank = self.avg_pareto_rank
            variance = sum((r - mean_rank) ** 2 for r in ranks) / len(ranks)
            self.consistency_score = 1.0 / (1.0 + math.sqrt(variance))
        else:
            self.consistency_score = 1.0

        # Compute average network pareto presence (if available)
        if self.network_pareto_presence:
            self.avg_network_pareto_presence = (
                sum(self.network_pareto_presence.values()) /
                len(self.network_pareto_presence)
            )

        # Overall score: weighted combination
        # If network pareto presence is available, use it as primary metric
        # Otherwise fall back to per-layer metrics
        if self.avg_network_pareto_presence > 0:
            # Network-aware scoring: prioritize network pareto presence
            rank_score = 1.0 / self.avg_pareto_rank if self.avg_pareto_rank > 0 else 0
            self.overall_score = (
                0.5 * self.avg_network_pareto_presence +  # Primary: network Pareto presence
                0.2 * rank_score +                         # Per-layer rank
                0.15 * self.avg_efficiency +               # Efficiency
                0.15 * self.consistency_score              # Consistency
            )
        else:
            # Legacy scoring (no network pareto data)
            rank_score = 1.0 / self.avg_pareto_rank if self.avg_pareto_rank > 0 else 0
            self.overall_score = (
                0.3 * rank_score +
                0.3 * self.pareto_optimal_ratio +
                0.2 * self.avg_efficiency +
                0.2 * self.consistency_score
            )


class PatternAnalyzer:
    """Analyze tiling patterns across layers."""

    def __init__(self, db_path: Path, objectives: list[tuple[str, str]] = None):
        """
        Args:
            db_path: Path to strategies.db
            objectives: List of (metric, direction) tuples. Default: latency:min, energy:min
        """
        self.db_path = Path(db_path)
        self.objectives = objectives or [("latency_ns", "min"), ("energy_nj", "min")]

        # Layer parameters (needed to compute ratios)
        self._layer_params: dict[int, dict] = {}

        # Pattern data
        self._patterns: dict[str, CrossLayerPattern] = {}

        # Load layer parameters
        self._load_layer_params()

    def _load_layer_params(self):
        """Load layer parameters from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get distinct layers and their parameters from any strategy
        cursor.execute("""
            SELECT DISTINCT layer_idx,
                   input_tile_p, input_tile_q, output_tile_p, output_tile_q
            FROM strategy_results
            ORDER BY layer_idx
        """)

        # We need actual layer dimensions (P, Q, R, S)
        # These should be in the strategy files or we derive from tile ranges
        # For now, let's get max tile sizes as approximation of full dimensions
        cursor.execute("""
            SELECT layer_idx,
                   MAX(output_tile_p) as max_out_p,
                   MAX(output_tile_q) as max_out_q,
                   MAX(input_tile_p) as max_in_p,
                   MAX(input_tile_q) as max_in_q
            FROM strategy_results
            GROUP BY layer_idx
        """)

        for row in cursor.fetchall():
            layer_idx, max_out_p, max_out_q, max_in_p, max_in_q = row
            self._layer_params[layer_idx] = {
                "P": max_out_p,  # Approximate P from max output tile
                "Q": max_out_q,  # Approximate Q from max output tile
                "full_input_p": max_in_p,  # Full input dimension
                "full_input_q": max_in_q,
            }

        conn.close()

    def get_layer_indices(self) -> list[int]:
        """Get all layer indices available in the database."""
        return sorted(self._layer_params.keys())

    def get_num_layers(self) -> int:
        """Get total number of layers in the database."""
        return len(self._layer_params)

    def _create_pattern(self, layer_idx: int,
                        input_tile_p: int, input_tile_q: int,
                        output_tile_p: int, output_tile_q: int) -> TilingPattern:
        """Create a pattern from absolute tile sizes."""
        params = self._layer_params.get(layer_idx, {})

        P = params.get("P", output_tile_p)
        Q = params.get("Q", output_tile_q)
        full_in_p = params.get("full_input_p", input_tile_p)
        full_in_q = params.get("full_input_q", input_tile_q)

        return TilingPattern(
            output_p_ratio=output_tile_p / P if P > 0 else 1.0,
            output_q_ratio=output_tile_q / Q if Q > 0 else 1.0,
            input_p_ratio=input_tile_p / full_in_p if full_in_p > 0 else 1.0,
            input_q_ratio=input_tile_q / full_in_q if full_in_q > 0 else 1.0,
        )

    def analyze_all_layers(self, layer_indices: list[int] = None) -> dict[str, CrossLayerPattern]:
        """Analyze patterns across specified layers.

        Args:
            layer_indices: List of layer indices to include. None = all layers.

        Returns:
            Dictionary mapping pattern_id to CrossLayerPattern
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # First, compute Pareto ranks per layer (reuse logic from strategy_scorer)
        layer_strategies = defaultdict(list)

        # Build query with optional layer filter
        if layer_indices is not None:
            placeholders = ",".join("?" * len(layer_indices))
            query = f"""
                SELECT layer_idx, strategy_id,
                       latency_ns, energy_nj, area_mm2, buffer_area_mm2,
                       COALESCE(sram_read_energy_nj, 0) + COALESCE(sram_write_energy_nj, 0) as buffer_energy,
                       input_tile_p, input_tile_q, output_tile_p, output_tile_q
                FROM strategy_results
                WHERE layer_idx IN ({placeholders})
                ORDER BY layer_idx
            """
            cursor.execute(query, layer_indices)
        else:
            query = """
                SELECT layer_idx, strategy_id,
                       latency_ns, energy_nj, area_mm2, buffer_area_mm2,
                       COALESCE(sram_read_energy_nj, 0) + COALESCE(sram_write_energy_nj, 0) as buffer_energy,
                       input_tile_p, input_tile_q, output_tile_p, output_tile_q
                FROM strategy_results
                ORDER BY layer_idx
            """
            cursor.execute(query)

        for row in cursor.fetchall():
            (layer_idx, strategy_id, latency, energy, area, buffer_area,
             buffer_energy, in_p, in_q, out_p, out_q) = row

            buffer_eap = buffer_energy * buffer_area if buffer_area else 0

            layer_strategies[layer_idx].append({
                "strategy_id": strategy_id,
                "latency_ns": latency,
                "energy_nj": energy,
                "area_mm2": area,
                "buffer_eap": buffer_eap,
                "input_tile_p": in_p,
                "input_tile_q": in_q,
                "output_tile_p": out_p,
                "output_tile_q": out_q,
            })

        conn.close()

        # Compute Pareto ranks and efficiency for each layer
        for layer_idx, strategies in layer_strategies.items():
            self._compute_layer_scores(layer_idx, strategies)

        # Group strategies by pattern
        pattern_data: dict[str, list[tuple[int, dict]]] = defaultdict(list)

        for layer_idx, strategies in layer_strategies.items():
            for s in strategies:
                pattern = self._create_pattern(
                    layer_idx,
                    s["input_tile_p"], s["input_tile_q"],
                    s["output_tile_p"], s["output_tile_q"]
                )
                pattern_id = pattern.pattern_id
                pattern_data[pattern_id].append((layer_idx, s, pattern))

        # Create CrossLayerPattern objects
        self._patterns = {}

        for pattern_id, entries in pattern_data.items():
            # Use first pattern as representative
            _, _, pattern = entries[0]

            cross_pattern = CrossLayerPattern(
                pattern=pattern,
                pattern_id=pattern_id,
            )

            for layer_idx, s, _ in entries:
                perf = PatternPerformance(
                    layer_idx=layer_idx,
                    strategy_id=s["strategy_id"],
                    latency_ns=s["latency_ns"],
                    energy_nj=s["energy_nj"],
                    area_mm2=s["area_mm2"],
                    buffer_eap=s.get("buffer_eap", 0),
                    pareto_rank=s.get("pareto_rank", 0),
                    efficiency_score=s.get("efficiency_score", 0),
                )
                cross_pattern.performances.append(perf)

            cross_pattern.compute_statistics()
            self._patterns[pattern_id] = cross_pattern

        return self._patterns

    def compute_network_pareto_presence(
        self,
        layer_indices: list[int] = None,
        quiet: bool = False,
    ) -> None:
        """
        Compute network-level Pareto presence rate for each pattern.

        Reads from existing Pareto CSV files generated by ./efsim plot.
        If CSV files don't exist, prints a warning.

        Args:
            layer_indices: List of layer indices to include. None = all layers.
            quiet: Suppress progress output.
        """
        import csv

        if not self._patterns:
            if not quiet:
                print("No patterns loaded. Call analyze_all_layers() first.")
            return

        workspace_path = self.db_path.parent

        # Determine which Pareto CSV to read based on layer_indices
        all_layers = self.get_layer_indices()
        num_all_layers = len(all_layers)

        if layer_indices is None:
            layer_indices = all_layers

        num_target_layers = len(layer_indices)

        # Determine folder path: network_full or progressive/layers_N
        if num_target_layers == num_all_layers:
            pareto_dir = workspace_path / "plots" / "network_full" / "all"
        else:
            pareto_dir = workspace_path / "plots" / "progressive" / f"layers_{num_target_layers}" / "all"

        pareto_csv = pareto_dir / "pareto.csv"

        if not pareto_csv.exists():
            if not quiet:
                print(f"\nWarning: Pareto CSV not found: {pareto_csv}")
                print(f"  Run './efsim plot {workspace_path.name}' first to generate Pareto data.")
            return

        if not quiet:
            print(f"\nReading network Pareto presence data from: {pareto_csv.relative_to(workspace_path)}")

        # Build strategy_id -> pattern_id mapping
        strategy_to_pattern: Dict[tuple, str] = {}  # (layer_idx, strategy_id) -> pattern_id
        for pattern_id, pattern in self._patterns.items():
            for perf in pattern.performances:
                key = (perf.layer_idx, perf.strategy_id)
                strategy_to_pattern[key] = pattern_id

        # Read Pareto CSV and count pattern appearances
        pattern_pareto_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_pareto_combos: Dict[str, int] = defaultdict(int)

        with open(pareto_csv, "r") as f:
            reader = csv.DictReader(f)

            # Get layer columns (e.g., "0_strategy", "1_strategy", ...)
            fieldnames = reader.fieldnames
            layer_columns = [col for col in fieldnames if col.endswith("_strategy")]

            for row in reader:
                objective_pair = row.get("objective_pair", "").strip('"')
                total_pareto_combos[objective_pair] += 1

                # Count patterns used in this Pareto combination
                patterns_in_combo = set()
                for col in layer_columns:
                    layer_idx = int(col.replace("_strategy", ""))
                    strategy_id = int(row[col]) if row[col] else None

                    if strategy_id is not None:
                        key = (layer_idx, strategy_id)
                        pattern_id = strategy_to_pattern.get(key)
                        if pattern_id:
                            patterns_in_combo.add(pattern_id)

                # Each pattern in this combo gets credit
                for pattern_id in patterns_in_combo:
                    pattern_pareto_counts[pattern_id][objective_pair] += 1

        # Compute presence rate (appearances / total Pareto combos)
        for pattern_id, pattern in self._patterns.items():
            pattern.network_pareto_presence = {}
            for x_key, y_key, label in NETWORK_OBJECTIVES:
                objective_pair = f"{x_key} vs {y_key}"
                count = pattern_pareto_counts[pattern_id][objective_pair]
                total = total_pareto_combos.get(objective_pair, 1)
                pattern.network_pareto_presence[objective_pair] = count / total if total > 0 else 0.0

            # Recompute statistics with new network pareto data
            pattern.compute_statistics()

        if not quiet:
            # Print summary
            print(f"\nNetwork Pareto presence from existing data ({len(NETWORK_OBJECTIVES)} objectives):")
            for x_key, y_key, label in NETWORK_OBJECTIVES:
                objective_pair = f"{x_key} vs {y_key}"
                total = total_pareto_combos.get(objective_pair, 0)
                print(f"  {label}: {total} Pareto combinations")

    def _compute_layer_scores(self, layer_idx: int, strategies: list[dict]):
        """Compute Pareto rank and efficiency for strategies in a layer."""
        # Get objective values
        obj_values = []
        for s in strategies:
            values = []
            for metric, direction in self.objectives:
                val = s.get(metric, 0)
                # For minimization, lower is better
                # For maximization, higher is better (negate for consistent comparison)
                if direction == "max":
                    val = -val
                values.append(val)
            obj_values.append(values)

        # Compute Pareto ranks
        n = len(strategies)
        dominated_by = [0] * n  # Count of strategies dominating this one

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if j dominates i
                    if self._dominates(obj_values[j], obj_values[i]):
                        dominated_by[i] += 1

        # Assign ranks (rank 1 = not dominated)
        for i, s in enumerate(strategies):
            s["pareto_rank"] = dominated_by[i] + 1

        # Compute efficiency (distance to ideal point, normalized)
        if strategies:
            min_vals = [min(obj_values[i][k] for i in range(n)) for k in range(len(self.objectives))]
            max_vals = [max(obj_values[i][k] for i in range(n)) for k in range(len(self.objectives))]

            for i, s in enumerate(strategies):
                # Normalized distance to ideal (min values)
                dist = 0
                for k in range(len(self.objectives)):
                    range_k = max_vals[k] - min_vals[k]
                    if range_k > 0:
                        dist += ((obj_values[i][k] - min_vals[k]) / range_k) ** 2
                dist = math.sqrt(dist)
                # Efficiency: 1 - normalized distance (higher is better)
                s["efficiency_score"] = max(0, 1 - dist / math.sqrt(len(self.objectives)))

    def _dominates(self, a: list[float], b: list[float]) -> bool:
        """Check if a dominates b (a is better or equal in all, strictly better in at least one)."""
        dominated = True
        strictly_better = False

        for i in range(len(a)):
            if a[i] > b[i]:  # a is worse (higher value for minimization)
                dominated = False
                break
            elif a[i] < b[i]:  # a is strictly better
                strictly_better = True

        return dominated and strictly_better

    def get_top_patterns(self, n: int = 10) -> list[CrossLayerPattern]:
        """Get top N patterns by cross-layer score."""
        sorted_patterns = sorted(
            self._patterns.values(),
            key=lambda p: p.overall_score,
            reverse=True
        )
        return sorted_patterns[:n]

    def print_summary(self, top_n: int = 20):
        """Print summary of top patterns."""
        patterns = self.get_top_patterns(top_n)

        # Check if network pareto presence is available
        has_network_pres = any(p.avg_network_pareto_presence > 0 for p in patterns)

        print(f"\n{'='*100}")
        print(f"Cross-Layer Pattern Analysis (Top {top_n})")
        print(f"{'='*100}")

        if has_network_pres:
            print(f"{'Rank':<5} {'Pattern ID':<25} {'Category':<15} {'Layers':<7} "
                  f"{'NetPres':<8} {'AvgRank':<8} {'Score':<8}")
        else:
            print(f"{'Rank':<5} {'Pattern ID':<25} {'Category':<15} {'Layers':<7} "
                  f"{'Avg Rank':<9} {'Pareto%':<8} {'Score':<8}")
        print(f"{'-'*100}")

        for i, p in enumerate(patterns, 1):
            if has_network_pres:
                print(f"{i:<5} {p.pattern_id:<25} {p.pattern.category:<15} "
                      f"{p.num_layers:<7} {p.avg_network_pareto_presence*100:<8.1f} "
                      f"{p.avg_pareto_rank:<8.2f} {p.overall_score:<8.4f}")
            else:
                print(f"{i:<5} {p.pattern_id:<25} {p.pattern.category:<15} "
                      f"{p.num_layers:<7} {p.avg_pareto_rank:<9.2f} "
                      f"{p.pareto_optimal_ratio*100:<8.1f} {p.overall_score:<8.4f}")

        print(f"{'='*100}")

    def save_to_csv(self, output_path: Path):
        """Save pattern analysis to CSV."""
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if network pareto presence is available
        all_patterns = self.get_top_patterns(len(self._patterns))
        has_network_pres = any(p.avg_network_pareto_presence > 0 for p in all_patterns)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Build header with optional network presence columns
            header = [
                "rank", "pattern_id", "category",
                "output_p_ratio", "output_q_ratio", "input_p_ratio", "input_q_ratio",
                "is_coupled", "is_full_output",
                "num_layers", "avg_pareto_rank", "pareto_optimal_ratio",
                "avg_efficiency", "consistency_score",
            ]

            if has_network_pres:
                header.append("avg_network_pareto_presence")
                # Add individual objective presence rates
                for x_key, y_key, label in NETWORK_OBJECTIVES:
                    short_name = label.replace("Latency vs ", "").replace(" ", "_").lower()
                    header.append(f"net_pres_{short_name}")

            header.append("overall_score")
            writer.writerow(header)

            for i, p in enumerate(all_patterns, 1):
                row = [
                    i, p.pattern_id, p.pattern.category,
                    f"{p.pattern.output_p_ratio:.4f}",
                    f"{p.pattern.output_q_ratio:.4f}",
                    f"{p.pattern.input_p_ratio:.4f}",
                    f"{p.pattern.input_q_ratio:.4f}",
                    p.pattern.is_coupled,
                    p.pattern.is_full_output,
                    p.num_layers,
                    f"{p.avg_pareto_rank:.4f}",
                    f"{p.pareto_optimal_ratio:.4f}",
                    f"{p.avg_efficiency:.4f}",
                    f"{p.consistency_score:.4f}",
                ]

                if has_network_pres:
                    row.append(f"{p.avg_network_pareto_presence:.4f}")
                    for x_key, y_key, label in NETWORK_OBJECTIVES:
                        objective_pair = f"{x_key} vs {y_key}"
                        pres = p.network_pareto_presence.get(objective_pair, 0.0)
                        row.append(f"{pres:.4f}")

                row.append(f"{p.overall_score:.4f}")
                writer.writerow(row)

        print(f"Saved pattern analysis to: {output_path}")

    def save_detailed_csv(self, output_path: Path):
        """Save detailed per-layer performance for each pattern."""
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pattern_id", "category", "layer_idx", "strategy_id",
                "latency_ns", "energy_nj", "area_mm2", "buffer_eap",
                "pareto_rank", "efficiency_score"
            ])

            for pattern in sorted(self._patterns.values(),
                                  key=lambda p: p.overall_score, reverse=True):
                for perf in sorted(pattern.performances, key=lambda x: x.layer_idx):
                    writer.writerow([
                        pattern.pattern_id,
                        pattern.pattern.category,
                        perf.layer_idx,
                        perf.strategy_id,
                        f"{perf.latency_ns:.2f}",
                        f"{perf.energy_nj:.4f}",
                        f"{perf.area_mm2:.6f}",
                        f"{perf.buffer_eap:.6f}",
                        perf.pareto_rank,
                        f"{perf.efficiency_score:.4f}"
                    ])

        print(f"Saved detailed pattern data to: {output_path}")

    def generate_plots(self, output_dir: Path) -> list[Path]:
        """Generate visualization plots for pattern analysis."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import Counter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        all_patterns = list(self._patterns.values())

        # Check if network pareto presence data is available
        has_network_pres = any(p.avg_network_pareto_presence > 0 for p in all_patterns)

        # Category colors (shared across plots)
        category_colors = {
            'full': '#2ecc71',
            'coupled_minimal': '#3498db',
            'coupled_quarter': '#9b59b6',
            'coupled_half': '#e74c3c',
            'coupled_large': '#f39c12',
            'super_tiling': '#1abc9c',
            'sub_tiling': '#e67e22',
        }

        # ============================================================
        # 1. Category Distribution (Bar only - pie has label overlap issues)
        # ============================================================
        fig, ax = plt.subplots(figsize=(12, 6))

        categories = Counter(p.pattern.category for p in self._patterns.values())
        # Sort by count (descending)
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        labels = [c[0] for c in sorted_cats]
        sizes = [c[1] for c in sorted_cats]

        # Use category_colors for consistency
        bar_colors = [category_colors.get(label, '#95a5a6') for label in labels]
        bars = ax.bar(labels, sizes, color=bar_colors, edgecolor='white', linewidth=1)
        ax.set_xlabel("Category", fontsize=11)
        ax.set_ylabel("Number of Patterns", fontsize=11)
        ax.set_title("Pattern Category Distribution (sorted by count)", fontsize=12)
        ax.tick_params(axis='x', rotation=30)

        # Add count labels on bars
        for bar, count in zip(bars, sizes):
            pct = count / sum(sizes) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

        ax.set_ylim(0, max(sizes) * 1.15)  # Room for labels
        plt.tight_layout()
        path = output_dir / "pattern_category_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

        # ============================================================
        # 2. Per-Objective Heatmaps (7 plots: Network + Per-layer Rank)
        # ============================================================
        # Select top 30 patterns by average Network Pareto Presence
        if has_network_pres:
            patterns_with_pres = [p for p in all_patterns if p.avg_network_pareto_presence > 0]
            top_patterns = sorted(patterns_with_pres,
                                  key=lambda p: p.avg_network_pareto_presence,
                                  reverse=True)[:30]
        else:
            top_patterns = self.get_top_patterns(30)

        all_layers = sorted(set(
            perf.layer_idx for p in top_patterns for perf in p.performances
        ))

        if top_patterns and all_layers and has_network_pres:
            # Short objective labels for filenames
            obj_short_names = ["energy", "buf_area", "sum_area", "peak_area",
                              "buf_eap", "sum_eap", "peak_eap"]

            for obj_idx, (x_key, y_key, obj_label) in enumerate(NETWORK_OBJECTIVES):
                objective_pair = f"{x_key} vs {y_key}"

                # Compute Network Pareto Rank for this objective (based on presence rate)
                # Sort patterns by this objective's presence rate, assign rank
                obj_pres = [(p, p.network_pareto_presence.get(objective_pair, 0))
                            for p in top_patterns]
                obj_pres_sorted = sorted(obj_pres, key=lambda x: x[1], reverse=True)

                # Build rank mapping: pattern -> network rank for this objective
                pattern_to_net_rank = {}
                for rank, (pattern, pres) in enumerate(obj_pres_sorted, 1):
                    pattern_to_net_rank[pattern.pattern_id] = rank if pres > 0 else None

                # Create figure
                fig, ax = plt.subplots(figsize=(14, 10))

                # Matrix: rows = patterns, cols = [Network] + [L0, L1, ...]
                n_cols = 1 + len(all_layers)
                matrix = np.full((len(top_patterns), n_cols), np.nan)

                for i, pattern in enumerate(top_patterns):
                    # Column 0: Network Rank (from presence rate for this objective)
                    net_rank = pattern_to_net_rank.get(pattern.pattern_id)
                    if net_rank is not None:
                        matrix[i, 0] = net_rank

                    # Columns 1+: Per-layer Pareto Rank
                    for perf in pattern.performances:
                        j = all_layers.index(perf.layer_idx) + 1  # +1 for Network column
                        matrix[i, j] = perf.pareto_rank

                # Determine vmax (exclude NaN)
                valid_values = matrix[~np.isnan(matrix)]
                vmax = np.max(valid_values) if len(valid_values) > 0 else 30

                im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                              vmin=1, vmax=vmax)

                # X-axis labels: [Network] + [L0, L1, ...]
                col_labels = ["Network"] + [f"L{l}" for l in all_layers]
                ax.set_xticks(range(n_cols))
                ax.set_xticklabels(col_labels, fontsize=10)

                # Add vertical line to separate Network from per-layer
                ax.axvline(x=0.5, color='black', linewidth=2)

                # Y-axis labels - shorter format
                short_labels = []
                for p in top_patterns:
                    pid = p.pattern_id
                    pid = pid.replace("out_", "o").replace("in_", "i").replace("_", " ")
                    pid = pid.replace("1/", "").replace("x", "×").replace("min", "m")
                    short_labels.append(pid)
                ax.set_yticks(range(len(top_patterns)))
                ax.set_yticklabels(short_labels, fontsize=9, family='monospace')

                ax.set_xlabel("Scope (Network = all layers combined)", fontsize=11)
                ax.set_ylabel("Pattern (Top 30 by avg Pareto Presence)", fontsize=11)
                ax.set_title(f"Pareto Rank: {obj_label}\n(green=good, red=bad, white=N/A)", fontsize=12)

                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("Rank (lower = better)")

                # Text annotations
                for i in range(len(top_patterns)):
                    for j in range(n_cols):
                        if not np.isnan(matrix[i, j]):
                            rank = int(matrix[i, j])
                            color = 'white' if rank > vmax * 0.6 else 'black'
                            ax.text(j, i, str(rank), ha='center', va='center',
                                   fontsize=8, color=color, fontweight='bold' if j == 0 else 'normal')

                plt.tight_layout()
                path = output_dir / f"pattern_rank_{obj_short_names[obj_idx]}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved_files.append(path)

        # ============================================================
        # 3. Score Analysis Scatter Plots (2 panels: Score vs Rank, Rank vs NetPres)
        # ============================================================
        n_plots = 2 if has_network_pres else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
        if n_plots == 1:
            axes = [axes]  # Make it iterable

        # Plot 1: Score vs Avg Pareto Rank
        for p in all_patterns:
            color = category_colors.get(p.pattern.category, '#95a5a6')
            axes[0].scatter(p.avg_pareto_rank, p.overall_score,
                           c=color, alpha=0.6, s=50,
                           edgecolors='white', linewidths=0.3)

        axes[0].set_xlabel("Average Pareto Rank (lower = better)", fontsize=10)
        axes[0].set_ylabel("Overall Score (higher = better)", fontsize=10)
        axes[0].set_title("Score vs Average Pareto Rank", fontsize=11)

        # Plot 2: Per-layer Rank vs Network Pareto Presence
        if has_network_pres:
            for p in all_patterns:
                color = category_colors.get(p.pattern.category, '#95a5a6')
                axes[1].scatter(p.avg_pareto_rank, p.avg_network_pareto_presence * 100,
                               c=color, alpha=0.6, s=50,
                               edgecolors='white', linewidths=0.3)

            axes[1].set_xlabel("Per-layer Avg Pareto Rank (lower = better)", fontsize=10)
            axes[1].set_ylabel("Network Pareto Presence % (higher = better)", fontsize=10)
            axes[1].set_title("Per-layer Rank vs Network Presence Rate", fontsize=11)

            # Add quadrant lines and annotations
            ax1_xlim = axes[1].get_xlim()
            ax1_ylim = axes[1].get_ylim()
            mid_x = min(100, (ax1_xlim[0] + ax1_xlim[1]) / 2)
            mid_y = 5

            axes[1].axvline(x=mid_x, color='gray', linestyle='--', alpha=0.4)
            axes[1].axhline(y=mid_y, color='gray', linestyle='--', alpha=0.4)

            # Golden patterns annotation
            axes[1].text(ax1_xlim[0] + (mid_x - ax1_xlim[0]) * 0.3,
                        ax1_ylim[1] * 0.85,
                        "Golden\n(low rank, high presence)",
                        fontsize=9, color='darkgreen', fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # Legend
        category_handles = [
            plt.scatter([], [], c=color, label=cat, s=60, edgecolors='white', linewidths=0.3)
            for cat, color in category_colors.items()
        ]
        axes[0].legend(handles=category_handles, loc='upper right',
                      fontsize=8, title="Category", title_fontsize=9)

        plt.tight_layout()
        path = output_dir / "pattern_score_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

        # ============================================================
        # 4. Top 10 Patterns - Summary Table + Key Metrics
        # ============================================================
        top10 = self.get_top_patterns(10)

        if top10:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Left: Horizontal bar chart with multiple metrics (stacked approach)
            y_pos = np.arange(len(top10))

            # Shorten pattern names
            short_names = []
            for p in top10:
                pid = p.pattern_id
                pid = pid.replace("out_", "o").replace("in_", "i").replace("_", " ")
                pid = pid.replace("1/", "").replace("x", "×").replace("min", "m")
                short_names.append(pid)

            # Plot key discriminating metrics
            metrics = {
                'Avg Rank (inv)': [100 / (p.avg_pareto_rank + 1) for p in top10],  # Inverted, normalized
                'Pareto %': [p.pareto_optimal_ratio * 100 for p in top10],
            }
            if has_network_pres:
                max_pres = max(p.avg_network_pareto_presence for p in top10) or 1
                metrics['Net Pres'] = [p.avg_network_pareto_presence / max_pres * 100 for p in top10]

            bar_height = 0.8 / len(metrics)
            colors_metric = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']

            for i, (label, values) in enumerate(metrics.items()):
                axes[0].barh(y_pos + i * bar_height, values, bar_height,
                            label=label, color=colors_metric[i % len(colors_metric)])

            axes[0].set_yticks(y_pos + bar_height * (len(metrics) - 1) / 2)
            axes[0].set_yticklabels(short_names, fontsize=9, family='monospace')
            axes[0].set_xlabel("Score (normalized to 100)", fontsize=10)
            axes[0].set_title("Top 10 Patterns - Key Metrics", fontsize=12)
            axes[0].legend(loc='lower right', fontsize=9)
            axes[0].invert_yaxis()

            # Right: Overall score bar
            overall_scores = [p.overall_score for p in top10]
            bar_colors = [category_colors.get(p.pattern.category, '#95a5a6') for p in top10]

            bars = axes[1].barh(y_pos, overall_scores, color=bar_colors, edgecolor='white')
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels([f"#{i+1}" for i in range(len(top10))], fontsize=10)
            axes[1].set_xlabel("Overall Score", fontsize=10)
            axes[1].set_title("Overall Score by Rank", fontsize=12)
            axes[1].invert_yaxis()

            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, overall_scores)):
                cat = top10[i].pattern.category
                axes[1].text(score + 0.01, bar.get_y() + bar.get_height()/2,
                           f"{score:.3f} ({cat})", va='center', fontsize=9)

            # Category legend
            unique_cats = list(set(p.pattern.category for p in top10))
            cat_handles = [plt.Rectangle((0,0), 1, 1, fc=category_colors.get(c, '#95a5a6'))
                          for c in unique_cats]
            axes[1].legend(cat_handles, unique_cats, loc='lower right', fontsize=8, title="Category")

            plt.tight_layout()
            path = output_dir / "pattern_top10_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(path)

        # ============================================================
        # 5. Output Ratio vs Input Ratio 2D Plot (Quantized Grid Heatmap)
        # ============================================================
        # Use quantized buckets for clarity instead of raw scatter
        # This aggregates patterns into grid cells for better visibility

        # Define bucket boundaries (matching quantize function)
        bucket_thresholds = [0.005, 0.01, 0.02, 0.04, 0.085, 0.175, 0.35, 0.75, 1.01]
        bucket_labels = ["min", "1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1"]

        def get_bucket_idx(r: float) -> int:
            for i, thresh in enumerate(bucket_thresholds):
                if r < thresh:
                    return i
            return len(bucket_thresholds) - 1

        # Build grid: aggregate patterns by (out_bucket, in_bucket)
        n_buckets = len(bucket_labels)
        grid_count = np.zeros((n_buckets, n_buckets))
        grid_pres = np.zeros((n_buckets, n_buckets))
        grid_best_pattern = {}  # (out_idx, in_idx) -> best pattern

        for p in all_patterns:
            out_idx = get_bucket_idx(p.pattern.output_p_ratio)
            in_idx = get_bucket_idx(p.pattern.input_p_ratio)
            grid_count[in_idx, out_idx] += 1

            pres = p.avg_network_pareto_presence if has_network_pres else 0
            if pres > grid_pres[in_idx, out_idx]:
                grid_pres[in_idx, out_idx] = pres
                grid_best_pattern[(out_idx, in_idx)] = p

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Left: Pattern Count Heatmap
        im1 = axes[0].imshow(grid_count, cmap='Blues', aspect='equal', origin='lower')
        axes[0].set_xticks(range(n_buckets))
        axes[0].set_xticklabels(bucket_labels, fontsize=10, rotation=45, ha='right')
        axes[0].set_yticks(range(n_buckets))
        axes[0].set_yticklabels(bucket_labels, fontsize=10)
        axes[0].set_xlabel("Output Tile Ratio", fontsize=12)
        axes[0].set_ylabel("Input Tile Ratio", fontsize=12)
        axes[0].set_title("Pattern Count by Ratio Bucket", fontsize=13)

        # Add diagonal line (coupled zone) - bucket indices where out ≈ in
        axes[0].plot([-0.5, n_buckets-0.5], [-0.5, n_buckets-0.5], 'k--', alpha=0.5, linewidth=2)

        # Annotations for count
        for i in range(n_buckets):
            for j in range(n_buckets):
                count = int(grid_count[i, j])
                if count > 0:
                    color = 'white' if count > grid_count.max() * 0.5 else 'black'
                    axes[0].text(j, i, str(count), ha='center', va='center',
                               fontsize=8, color=color, fontweight='bold')

        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
        cbar1.set_label("Pattern Count", fontsize=11)

        # Right: Best Network Presence Heatmap
        if has_network_pres:
            grid_pres_pct = grid_pres * 100
            im2 = axes[1].imshow(grid_pres_pct, cmap='YlGn', aspect='equal', origin='lower',
                                vmin=0, vmax=max(20, grid_pres_pct.max()))
            axes[1].set_xticks(range(n_buckets))
            axes[1].set_xticklabels(bucket_labels, fontsize=10, rotation=45, ha='right')
            axes[1].set_yticks(range(n_buckets))
            axes[1].set_yticklabels(bucket_labels, fontsize=10)
            axes[1].set_xlabel("Output Tile Ratio", fontsize=12)
            axes[1].set_ylabel("Input Tile Ratio", fontsize=12)
            axes[1].set_title("Best Network Pareto Presence Rate by Bucket (%)", fontsize=13)

            # Diagonal line
            axes[1].plot([-0.5, n_buckets-0.5], [-0.5, n_buckets-0.5], 'k--', alpha=0.5, linewidth=2)

            # Annotations for presence rate
            for i in range(n_buckets):
                for j in range(n_buckets):
                    pres = grid_pres_pct[i, j]
                    if pres > 0:
                        color = 'white' if pres > 15 else 'black'
                        axes[1].text(j, i, f"{pres:.0f}", ha='center', va='center',
                                   fontsize=8, color=color, fontweight='bold')

            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
            cbar2.set_label("Network Pareto Presence %", fontsize=11)

            # Highlight top 5 cells with stars
            top_cells = sorted(
                [(grid_pres[i, j], i, j) for i in range(n_buckets) for j in range(n_buckets) if grid_pres[i, j] > 0],
                reverse=True
            )[:5]
            for rank, (pres, i, j) in enumerate(top_cells, 1):
                axes[1].scatter(j, i, marker='*', s=200, c='red', edgecolors='white', linewidths=1, zorder=10)
                axes[1].text(j + 0.3, i + 0.3, f"#{rank}", fontsize=8, color='red', fontweight='bold')

        else:
            # If no network presence data, show avg pareto rank instead
            grid_rank = np.full((n_buckets, n_buckets), np.nan)
            for p in all_patterns:
                out_idx = get_bucket_idx(p.pattern.output_p_ratio)
                in_idx = get_bucket_idx(p.pattern.input_p_ratio)
                if np.isnan(grid_rank[in_idx, out_idx]) or p.avg_pareto_rank < grid_rank[in_idx, out_idx]:
                    grid_rank[in_idx, out_idx] = p.avg_pareto_rank

            im2 = axes[1].imshow(grid_rank, cmap='RdYlGn_r', aspect='equal', origin='lower')
            axes[1].set_xticks(range(n_buckets))
            axes[1].set_xticklabels(bucket_labels, fontsize=10, rotation=45, ha='right')
            axes[1].set_yticks(range(n_buckets))
            axes[1].set_yticklabels(bucket_labels, fontsize=10)
            axes[1].set_xlabel("Output Tile Ratio", fontsize=12)
            axes[1].set_ylabel("Input Tile Ratio", fontsize=12)
            axes[1].set_title("Best Pareto Rank by Bucket (lower=better)", fontsize=13)

            # Diagonal line
            axes[1].plot([-0.5, n_buckets-0.5], [-0.5, n_buckets-0.5], 'k--', alpha=0.5, linewidth=2)

            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
            cbar2.set_label("Best Pareto Rank", fontsize=11)

        plt.tight_layout()
        path = output_dir / "pattern_ratio_space.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

        # ============================================================
        # 6. Network Pareto Presence Heatmap (7 objectives × top patterns)
        # ============================================================
        if has_network_pres:
            # Get top 20 patterns by network presence (reduced for readability)
            patterns_with_pres = [p for p in all_patterns if p.avg_network_pareto_presence > 0]
            top_by_pres = sorted(patterns_with_pres,
                                key=lambda p: p.avg_network_pareto_presence,
                                reverse=True)[:20]

            if top_by_pres:
                fig, ax = plt.subplots(figsize=(12, 10))

                # Build matrix: rows = patterns, cols = 7 objectives
                objectives = list(NETWORK_OBJECTIVES)
                matrix = np.zeros((len(top_by_pres), len(objectives)))

                for i, pattern in enumerate(top_by_pres):
                    for j, (x_key, y_key, label) in enumerate(objectives):
                        obj_pair = f"{x_key} vs {y_key}"
                        pres = pattern.network_pareto_presence.get(obj_pair, 0)
                        matrix[i, j] = pres * 100  # Convert to percentage

                # Better vmax based on actual data
                actual_max = np.max(matrix) if np.max(matrix) > 0 else 100
                im = ax.imshow(matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=max(50, actual_max))

                # Shorter objective labels
                obj_labels = ["Energy", "Buf Area", "Sum Area", "Mult Area", "Buf EAP", "Sum EAP", "Mult EAP"]
                ax.set_xticks(range(len(objectives)))
                ax.set_xticklabels(obj_labels, fontsize=11, rotation=30, ha='right')

                # Shorter pattern labels
                short_labels = []
                for p in top_by_pres:
                    pid = p.pattern_id
                    pid = pid.replace("out_", "o").replace("in_", "i").replace("_", " ")
                    pid = pid.replace("1/", "").replace("x", "×").replace("min", "m")
                    short_labels.append(pid)
                ax.set_yticks(range(len(top_by_pres)))
                ax.set_yticklabels(short_labels, fontsize=10, family='monospace')

                ax.set_xlabel("Objective (Latency vs ...)", fontsize=12)
                ax.set_ylabel("Pattern (Top 20 by presence)", fontsize=12)
                ax.set_title("Network Pareto Presence Rate by Objective (%)", fontsize=13)

                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("Presence Rate %", fontsize=11)

                # Text annotations with better contrast
                for i in range(len(top_by_pres)):
                    for j in range(len(objectives)):
                        val = matrix[i, j]
                        if val > 0:
                            color = 'white' if val > 30 else 'black'
                            ax.text(j, i, f"{val:.0f}", ha='center', va='center',
                                   fontsize=9, color=color, fontweight='bold')

                plt.tight_layout()
                path = output_dir / "pattern_network_presence_heatmap.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved_files.append(path)

        # ============================================================
        # 7. Category Performance Summary (network presence by category)
        # ============================================================
        if has_network_pres:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Left: Box plot of network presence by category
            patterns_with_pres = [p for p in all_patterns if p.avg_network_pareto_presence > 0]
            if patterns_with_pres:
                cat_pres = {}
                for p in patterns_with_pres:
                    cat = p.pattern.category
                    if cat not in cat_pres:
                        cat_pres[cat] = []
                    cat_pres[cat].append(p.avg_network_pareto_presence * 100)

                # Sort by median presence rate
                sorted_cats = sorted(cat_pres.keys(),
                                    key=lambda c: np.median(cat_pres[c]),
                                    reverse=True)

                box_data = [cat_pres[cat] for cat in sorted_cats]
                box_colors = [category_colors.get(cat, '#95a5a6') for cat in sorted_cats]

                bp = axes[0].boxplot(box_data, patch_artist=True, labels=sorted_cats)
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                axes[0].set_xlabel("Category")
                axes[0].set_ylabel("Network Pareto Presence %")
                axes[0].set_title("Network Pareto Presence Distribution by Category")
                axes[0].tick_params(axis='x', rotation=45)

            # Right: Stacked bar - count of patterns by category in different presence ranges
            pres_ranges = [(0, 5), (5, 10), (10, 20), (20, 100)]
            range_labels = ["0-5%", "5-10%", "10-20%", "20%+"]

            cat_counts = {cat: [0] * len(pres_ranges) for cat in category_colors.keys()}
            for p in patterns_with_pres:
                pres = p.avg_network_pareto_presence * 100
                cat = p.pattern.category
                if cat not in cat_counts:
                    cat_counts[cat] = [0] * len(pres_ranges)
                for i, (low, high) in enumerate(pres_ranges):
                    if low <= pres < high:
                        cat_counts[cat][i] += 1
                        break

            # Filter out empty categories
            active_cats = [cat for cat in category_colors.keys() if sum(cat_counts.get(cat, [])) > 0]
            x = np.arange(len(active_cats))
            width = 0.2

            for i, (low, high) in enumerate(pres_ranges):
                counts = [cat_counts.get(cat, [0]*len(pres_ranges))[i] for cat in active_cats]
                axes[1].bar(x + i * width, counts, width, label=range_labels[i])

            axes[1].set_xlabel("Category")
            axes[1].set_ylabel("Number of Patterns")
            axes[1].set_title("Pattern Count by Category and Presence Range")
            axes[1].set_xticks(x + width * 1.5)
            axes[1].set_xticklabels(active_cats, rotation=45, ha='right')
            axes[1].legend(title="Presence Range")

            plt.tight_layout()
            path = output_dir / "pattern_category_network_presence.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(path)

        return saved_files
