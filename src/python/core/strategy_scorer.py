"""
Strategy Scorer - Calculate usefulness scores for tiling strategies

Scoring metrics:
1. Dominance Score: How many other strategies does this one dominate?
2. Pareto Rank: Which Pareto front does this strategy belong to? (1 = best)
3. Hypervolume Contribution: How much unique area does this strategy cover on Pareto front?
4. Efficiency Score: Distance to ideal point (lower latency, lower energy)
5. Robustness Score: How often is this strategy Pareto-optimal across hardware configs?
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class StrategyScore:
    """Score data for a single strategy"""
    layer_idx: int
    strategy_id: int

    # Raw metrics
    latency_ns: float
    energy_nj: float
    area_mm2: float

    # Tiling config
    input_tile_p: int
    input_tile_q: int
    output_tile_p: int
    output_tile_q: int

    # Optional metrics (with defaults)
    buffer_area_mm2: float = 0.0
    buffer_eap: float = 0.0  # buffer_energy * buffer_area

    # Computed scores
    dominance_score: float = 0.0       # [0, 1] - fraction of strategies dominated
    pareto_rank: int = 0               # 1 = Pareto front, 2 = 2nd front, etc.
    hypervolume_contribution: float = 0.0  # Only for Pareto-optimal strategies
    efficiency_score: float = 0.0      # [0, 1] - closeness to ideal point

    # Composite score
    overall_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            'layer_idx': self.layer_idx,
            'strategy_id': self.strategy_id,
            'latency_ns': self.latency_ns,
            'energy_nj': self.energy_nj,
            'area_mm2': self.area_mm2,
            'buffer_area_mm2': self.buffer_area_mm2,
            'buffer_eap': self.buffer_eap,
            'input_tile_p': self.input_tile_p,
            'input_tile_q': self.input_tile_q,
            'output_tile_p': self.output_tile_p,
            'output_tile_q': self.output_tile_q,
            'dominance_score': self.dominance_score,
            'pareto_rank': self.pareto_rank,
            'hypervolume_contribution': self.hypervolume_contribution,
            'efficiency_score': self.efficiency_score,
            'overall_score': self.overall_score,
        }


class StrategyScorer:
    """Calculate usefulness scores for tiling strategies"""

    def __init__(self, db_path: Path, objectives: List[Tuple[str, str]] = None):
        """
        Args:
            db_path: Path to strategies.db
            objectives: List of (metric_name, direction) tuples
                        direction is 'min' or 'max'
                        Default: [('latency_ns', 'min'), ('energy_nj', 'min')]
        """
        self.db_path = Path(db_path)
        self.objectives = objectives or [
            ('latency_ns', 'min'),
            ('energy_nj', 'min'),
        ]
        self._scores: Dict[int, Dict[int, StrategyScore]] = {}  # layer_idx -> strategy_id -> score

    def _load_strategies(self, layer_idx: int) -> List[StrategyScore]:
        """Load all strategies for a layer from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT layer_idx, strategy_id,
                   latency_ns, energy_nj, area_mm2, buffer_area_mm2,
                   sram_read_energy_nj, sram_write_energy_nj,
                   input_tile_p, input_tile_q, output_tile_p, output_tile_q
            FROM strategy_results
            WHERE layer_idx = ?
        """, (layer_idx,))

        strategies = []
        for row in cursor.fetchall():
            buffer_area = row['buffer_area_mm2'] or 0.0
            sram_read = row['sram_read_energy_nj'] or 0.0
            sram_write = row['sram_write_energy_nj'] or 0.0
            buffer_energy = sram_read + sram_write
            buffer_eap = buffer_energy * buffer_area

            strategies.append(StrategyScore(
                layer_idx=row['layer_idx'],
                strategy_id=row['strategy_id'],
                latency_ns=row['latency_ns'],
                energy_nj=row['energy_nj'],
                area_mm2=row['area_mm2'],
                buffer_area_mm2=buffer_area,
                buffer_eap=buffer_eap,
                input_tile_p=row['input_tile_p'] or 0,
                input_tile_q=row['input_tile_q'] or 0,
                output_tile_p=row['output_tile_p'] or 0,
                output_tile_q=row['output_tile_q'] or 0,
            ))

        conn.close()
        return strategies

    def _get_layer_indices(self) -> List[int]:
        """Get all layer indices from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT layer_idx FROM strategy_results ORDER BY layer_idx")
        indices = [row[0] for row in cursor.fetchall()]
        conn.close()
        return indices

    def _dominates(self, a: StrategyScore, b: StrategyScore) -> bool:
        """Check if strategy a dominates strategy b (Pareto dominance)"""
        dominated = False
        for metric, direction in self.objectives:
            a_val = getattr(a, metric)
            b_val = getattr(b, metric)

            if direction == 'min':
                if a_val > b_val:
                    return False  # a is worse in this objective
                if a_val < b_val:
                    dominated = True
            else:  # max
                if a_val < b_val:
                    return False
                if a_val > b_val:
                    dominated = True

        return dominated

    def _compute_dominance_scores(self, strategies: List[StrategyScore]) -> None:
        """Compute dominance score for each strategy"""
        n = len(strategies)
        if n == 0:
            return

        for s in strategies:
            dominated_count = 0
            for other in strategies:
                if s.strategy_id != other.strategy_id:
                    if self._dominates(s, other):
                        dominated_count += 1
            s.dominance_score = dominated_count / (n - 1) if n > 1 else 0.0

    def _compute_pareto_ranks(self, strategies: List[StrategyScore]) -> None:
        """Compute Pareto rank (1 = front, 2 = 2nd front, etc.)"""
        remaining = strategies.copy()
        rank = 1

        while remaining:
            # Find non-dominated strategies in remaining set
            front = []
            for s in remaining:
                dominated = False
                for other in remaining:
                    if s.strategy_id != other.strategy_id:
                        if self._dominates(other, s):
                            dominated = True
                            break
                if not dominated:
                    front.append(s)

            # Assign rank to front
            for s in front:
                s.pareto_rank = rank

            # Remove front from remaining
            front_ids = {s.strategy_id for s in front}
            remaining = [s for s in remaining if s.strategy_id not in front_ids]
            rank += 1

    def _compute_efficiency_scores(self, strategies: List[StrategyScore]) -> None:
        """Compute efficiency score based on distance to ideal point"""
        if not strategies:
            return

        # Find ideal point (best value for each objective)
        ideal = {}
        nadir = {}  # worst point for normalization

        for metric, direction in self.objectives:
            values = [getattr(s, metric) for s in strategies]
            if direction == 'min':
                ideal[metric] = min(values)
                nadir[metric] = max(values)
            else:
                ideal[metric] = max(values)
                nadir[metric] = min(values)

        # Compute normalized distance to ideal point
        for s in strategies:
            distance_sq = 0.0
            for metric, direction in self.objectives:
                val = getattr(s, metric)
                ideal_val = ideal[metric]
                nadir_val = nadir[metric]

                # Normalize to [0, 1]
                range_val = abs(nadir_val - ideal_val)
                if range_val > 0:
                    if direction == 'min':
                        normalized = (val - ideal_val) / range_val
                    else:
                        normalized = (ideal_val - val) / range_val
                else:
                    normalized = 0.0

                distance_sq += normalized ** 2

            distance = math.sqrt(distance_sq)
            # Convert distance to score (closer = higher score)
            max_distance = math.sqrt(len(self.objectives))  # Maximum possible distance
            s.efficiency_score = 1.0 - (distance / max_distance)

    def _compute_hypervolume_contributions(self, strategies: List[StrategyScore]) -> None:
        """Compute hypervolume contribution for Pareto-optimal strategies"""
        # Get Pareto-optimal strategies
        pareto_front = [s for s in strategies if s.pareto_rank == 1]

        if len(pareto_front) <= 1:
            for s in pareto_front:
                s.hypervolume_contribution = 1.0
            return

        # Sort by first objective for 2D case
        metric1, dir1 = self.objectives[0]
        metric2, dir2 = self.objectives[1]

        # Sort by first metric (ascending for min, descending for max)
        reverse = (dir1 == 'max')
        pareto_front.sort(key=lambda s: getattr(s, metric1), reverse=reverse)

        # Find reference point (worst values with margin)
        all_vals1 = [getattr(s, metric1) for s in strategies]
        all_vals2 = [getattr(s, metric2) for s in strategies]

        if dir1 == 'min':
            ref1 = max(all_vals1) * 1.1
        else:
            ref1 = min(all_vals1) * 0.9

        if dir2 == 'min':
            ref2 = max(all_vals2) * 1.1
        else:
            ref2 = min(all_vals2) * 0.9

        # Compute total hypervolume
        total_hv = self._compute_2d_hypervolume(pareto_front, metric1, metric2, ref1, ref2)

        # Compute contribution for each point (exclusion method)
        for i, s in enumerate(pareto_front):
            # Hypervolume without this point
            other_front = pareto_front[:i] + pareto_front[i+1:]
            if other_front:
                hv_without = self._compute_2d_hypervolume(other_front, metric1, metric2, ref1, ref2)
            else:
                hv_without = 0.0

            contribution = total_hv - hv_without
            s.hypervolume_contribution = contribution / total_hv if total_hv > 0 else 0.0

    def _compute_2d_hypervolume(self, points: List[StrategyScore],
                                 metric1: str, metric2: str,
                                 ref1: float, ref2: float) -> float:
        """Compute 2D hypervolume (area dominated by points)"""
        if not points:
            return 0.0

        # Sort by first metric
        sorted_points = sorted(points, key=lambda s: getattr(s, metric1))

        hv = 0.0
        prev_val2 = ref2

        for s in sorted_points:
            val1 = getattr(s, metric1)
            val2 = getattr(s, metric2)

            width = ref1 - val1
            height = prev_val2 - val2

            if height > 0 and width > 0:
                hv += width * height

            prev_val2 = min(prev_val2, val2)

        return hv

    def _compute_overall_scores(self, strategies: List[StrategyScore],
                                 weights: Dict[str, float] = None) -> None:
        """Compute weighted overall score"""
        weights = weights or {
            'dominance': 0.2,
            'pareto_rank': 0.3,
            'efficiency': 0.3,
            'hypervolume': 0.2,
        }

        if not strategies:
            return

        # Normalize pareto_rank (lower is better)
        max_rank = max(s.pareto_rank for s in strategies)

        for s in strategies:
            # Pareto rank score: 1.0 for rank 1, decreasing for higher ranks
            rank_score = 1.0 - (s.pareto_rank - 1) / max_rank if max_rank > 1 else 1.0

            s.overall_score = (
                weights['dominance'] * s.dominance_score +
                weights['pareto_rank'] * rank_score +
                weights['efficiency'] * s.efficiency_score +
                weights['hypervolume'] * s.hypervolume_contribution
            )

    def score_layer(self, layer_idx: int) -> List[StrategyScore]:
        """Compute all scores for a single layer"""
        strategies = self._load_strategies(layer_idx)

        if not strategies:
            return []

        self._compute_dominance_scores(strategies)
        self._compute_pareto_ranks(strategies)
        self._compute_efficiency_scores(strategies)
        self._compute_hypervolume_contributions(strategies)
        self._compute_overall_scores(strategies)

        # Cache results
        self._scores[layer_idx] = {s.strategy_id: s for s in strategies}

        return strategies

    def score_all_layers(self) -> Dict[int, List[StrategyScore]]:
        """Compute scores for all layers"""
        results = {}
        for layer_idx in self._get_layer_indices():
            results[layer_idx] = self.score_layer(layer_idx)
        return results

    def save_to_db(self) -> None:
        """Save scores to database (new table)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_scores (
                layer_idx INTEGER,
                strategy_id INTEGER,
                dominance_score REAL,
                pareto_rank INTEGER,
                hypervolume_contribution REAL,
                efficiency_score REAL,
                overall_score REAL,
                PRIMARY KEY (layer_idx, strategy_id)
            )
        """)

        # Insert scores
        for layer_idx, strategies in self._scores.items():
            for s in strategies.values():
                cursor.execute("""
                    INSERT OR REPLACE INTO strategy_scores
                    (layer_idx, strategy_id, dominance_score, pareto_rank,
                     hypervolume_contribution, efficiency_score, overall_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (s.layer_idx, s.strategy_id, s.dominance_score, s.pareto_rank,
                      s.hypervolume_contribution, s.efficiency_score, s.overall_score))

        conn.commit()
        conn.close()

    def save_to_csv(self, output_path: Path) -> None:
        """Save scores to CSV file"""
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for layer_idx in sorted(self._scores.keys()):
            for s in sorted(self._scores[layer_idx].values(),
                          key=lambda x: -x.overall_score):
                rows.append(s.to_dict())

        if rows:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    def save_to_csv_per_layer(self, output_dir: Path) -> List[Path]:
        """Save scores to separate CSV files per layer, sorted by overall_score descending"""
        import csv

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for layer_idx in sorted(self._scores.keys()):
            # Sort strategies by overall_score descending
            sorted_strategies = sorted(
                self._scores[layer_idx].values(),
                key=lambda x: -x.overall_score
            )

            # Add rank column
            rows = []
            for rank, s in enumerate(sorted_strategies, 1):
                row = s.to_dict()
                row['rank'] = rank
                rows.append(row)

            if rows:
                csv_path = output_dir / f"L{layer_idx}_scores.csv"
                # Reorder columns: rank first, then others
                fieldnames = ['rank'] + [k for k in rows[0].keys() if k != 'rank']

                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

                saved_files.append(csv_path)

        return saved_files

    def get_top_strategies(self, layer_idx: int, n: int = 10) -> List[StrategyScore]:
        """Get top N strategies by overall score"""
        if layer_idx not in self._scores:
            self.score_layer(layer_idx)

        strategies = list(self._scores[layer_idx].values())
        strategies.sort(key=lambda s: -s.overall_score)
        return strategies[:n]

    def print_summary(self, layer_idx: int = None, top_n: int = 10) -> None:
        """Print summary of scores"""
        layers = [layer_idx] if layer_idx is not None else sorted(self._scores.keys())

        for li in layers:
            if li not in self._scores:
                continue

            strategies = list(self._scores[li].values())
            pareto_count = sum(1 for s in strategies if s.pareto_rank == 1)

            print(f"\n{'='*70}")
            print(f"Layer {li}: {len(strategies)} strategies, {pareto_count} Pareto-optimal")
            print(f"{'='*70}")

            print(f"\n{'Top ' + str(top_n) + ' Strategies by Overall Score':^70}")
            print("-" * 70)
            print(f"{'Rank':<5} {'ID':<8} {'Tiles':<15} {'Score':<8} {'Pareto':<7} "
                  f"{'Dom':<6} {'Eff':<6} {'HV':<6}")
            print("-" * 70)

            top = self.get_top_strategies(li, top_n)
            for rank, s in enumerate(top, 1):
                tiles = f"{s.output_tile_p}x{s.output_tile_q}/{s.input_tile_p}x{s.input_tile_q}"
                print(f"{rank:<5} {s.strategy_id:<8} {tiles:<15} {s.overall_score:<8.3f} "
                      f"{s.pareto_rank:<7} {s.dominance_score:<6.3f} "
                      f"{s.efficiency_score:<6.3f} {s.hypervolume_contribution:<6.3f}")

            # Statistics
            print(f"\n{'Score Distribution':^70}")
            print("-" * 70)
            scores = [s.overall_score for s in strategies]
            print(f"  Min: {min(scores):.4f}  Max: {max(scores):.4f}  "
                  f"Mean: {sum(scores)/len(scores):.4f}")

            # Pareto rank distribution
            rank_dist = {}
            for s in strategies:
                rank_dist[s.pareto_rank] = rank_dist.get(s.pareto_rank, 0) + 1

            print(f"\n{'Pareto Rank Distribution':^70}")
            print("-" * 70)
            for rank in sorted(rank_dist.keys())[:5]:
                pct = rank_dist[rank] / len(strategies) * 100
                print(f"  Rank {rank}: {rank_dist[rank]:>5} strategies ({pct:>5.1f}%)")
