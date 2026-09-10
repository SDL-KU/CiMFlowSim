#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Optional logging - fall back to standard logging if core.logging_config unavailable
try:
    from core.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Common constants
COLORS = {
    "LOAD": "#FF6B6B",  # Red
    "IBUF_READ": "#4ECDC4",  # Teal
    "CIM_COMPUTE": "#45B7D1",  # Blue
    "OBUF_WRITE": "#96CEB4",  # Green
    "STORE": "#FFEAA7",  # Yellow
}

OP_TYPES = ["STORE", "OBUF_WRITE", "CIM_COMPUTE", "IBUF_READ", "LOAD"]  # Reversed order for display

# Binary format constants (must match C++ GanttGenerator)
GANTT_MAGIC = 0x544E4147  # "GANT" in little-endian
GANTT_VERSION = 1
OP_TYPE_NAMES = ["LOAD", "IBUF_READ", "CIM_COMPUTE", "OBUF_WRITE", "STORE"]

# NumPy dtype for binary records (32 bytes, matches C++ GanttRecord)
GANTT_RECORD_DTYPE = np.dtype([
    ('op_type', 'u1'),       # uint8
    ('reserved', 'u1', 3),   # uint8[3] padding
    ('op_id', 'u4'),         # uint32
    ('start_time', 'f8'),    # double
    ('end_time', 'f8'),      # double
    ('src_total', 'u2'),     # uint16
    ('src_max', 'u2'),       # uint16
    ('dst_total', 'u2'),     # uint16
    ('dst_max', 'u2'),       # uint16
])


def parse_gantt_data_binary(filename):
    """
    Parse Gantt chart data from binary format (fast, preferred).

    Binary format (gantt_data.bin):
    - Header: 4 bytes magic + 4 bytes version + 4 bytes record count
    - Records: 32 bytes each (see GANTT_RECORD_DTYPE)

    Args:
        filename (str): Path to binary gantt data file

    Returns:
        list: List of operation dictionaries with timing information
    """
    with open(filename, 'rb') as f:
        # Read header
        header = np.frombuffer(f.read(12), dtype=np.uint32)
        magic, version, record_count = header

        if magic != GANTT_MAGIC:
            raise ValueError(f"Invalid magic number: {hex(magic)}, expected {hex(GANTT_MAGIC)}")
        if version != GANTT_VERSION:
            raise ValueError(f"Unsupported version: {version}, expected {GANTT_VERSION}")

        # Read all records at once (very fast!)
        records = np.fromfile(f, dtype=GANTT_RECORD_DTYPE, count=record_count)

    # Vectorized conversion to list of dicts (much faster than loop)
    # Pre-compute operation types as numpy array
    op_types = np.array(OP_TYPE_NAMES + ["UNKNOWN"])
    type_indices = np.clip(records['op_type'], 0, len(OP_TYPE_NAMES))
    types = op_types[type_indices]

    # Extract arrays (zero-copy views where possible)
    ids = records['op_id']
    starts = records['start_time']
    ends = records['end_time']
    durations = ends - starts
    src_totals = records['src_total']
    src_maxs = records['src_max']
    dst_totals = records['dst_total']
    dst_maxs = records['dst_max']

    # Build list using list comprehension (faster than append loop)
    operations = [
        {
            "type": types[i],
            "id": int(ids[i]),
            "start": float(starts[i]),
            "end": float(ends[i]),
            "duration": float(durations[i]),
            "src_total": int(src_totals[i]),
            "src_max": int(src_maxs[i]),
            "dst_total": int(dst_totals[i]),
            "dst_max": int(dst_maxs[i]),
        }
        for i in range(len(records))
    ]

    return operations


def parse_gantt_data_csv(filename):
    """
    Parse Gantt chart data from CSV format (legacy, slower).

    CSV format:
        # op,id,start,end,src_total,src_max,dst_total,dst_max
        LOAD,0,0.00,121.25,6,2,1,1

    Args:
        filename (str): Path to CSV gantt data file

    Returns:
        list: List of operation dictionaries with timing information
    """
    operations = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    op_data = {
                        "type": parts[0],
                        "id": int(parts[1]),
                        "start": float(parts[2]),
                        "end": float(parts[3]),
                        "duration": float(parts[3]) - float(parts[2]),
                    }

                    if len(parts) >= 8:
                        op_data["src_total"] = int(parts[4])
                        op_data["src_max"] = int(parts[5])
                        op_data["dst_total"] = int(parts[6])
                        op_data["dst_max"] = int(parts[7])

                    operations.append(op_data)
                except (ValueError, IndexError):
                    continue

    return operations


def parse_gantt_data(filename="gantt_data.bin"):
    """
    Parse Gantt chart data from SystemC simulation output.

    Automatically detects format:
    - .bin: Binary format (fast, preferred)
    - .txt/.csv: CSV format (legacy, slower)

    Args:
        filename (str): Path to gantt data file

    Returns:
        list: List of operation dictionaries with timing information
    """
    filename = str(filename)
    path = Path(filename)

    # Auto-detect format based on extension or file content
    if path.suffix == '.bin':
        return parse_gantt_data_binary(filename)
    elif path.suffix in ('.txt', '.csv'):
        return parse_gantt_data_csv(filename)
    else:
        # Try binary first (check magic number)
        try:
            with open(filename, 'rb') as f:
                magic = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                if magic == GANTT_MAGIC:
                    return parse_gantt_data_binary(filename)
        except Exception:
            pass
        # Fall back to CSV
        return parse_gantt_data_csv(filename)


def parse_gantt_data_numpy(filename):
    """
    Parse binary Gantt data and return numpy structured array directly.

    This is the fastest option when you don't need dict conversion.
    Use with create_gantt_chart_numpy() for 30-70x speedup over dict-based approach.

    Args:
        filename (str): Path to binary gantt_data.bin file

    Returns:
        numpy.ndarray: Structured array with fields: op_type, op_id, start_time, end_time, etc.

    Raises:
        ValueError: If file format is invalid or unsupported version
    """
    with open(filename, 'rb') as f:
        header = np.frombuffer(f.read(12), dtype=np.uint32)
        magic, version, record_count = header

        if magic != GANTT_MAGIC:
            raise ValueError(f"Invalid magic number: {hex(magic)}, expected {hex(GANTT_MAGIC)}")
        if version != GANTT_VERSION:
            raise ValueError(f"Unsupported version: {version}, expected {GANTT_VERSION}")

        return np.fromfile(f, dtype=GANTT_RECORD_DTYPE, count=record_count)


def create_gantt_chart_numpy(records):
    """
    Create Gantt chart from numpy structured array using vectorized drawing.

    This is ~30-70x faster than create_gantt_chart_simple() for large datasets.
    Uses matplotlib's broken_barh() for vectorized rectangle drawing instead of
    individual Rectangle patches.

    Performance (702K operations, 21MB):
    - Dict-based: 250 seconds (4.2 minutes)
    - Numpy-based: 7.4 seconds (33.8x faster)

    Args:
        records: numpy structured array from parse_gantt_data_numpy()

    Returns:
        matplotlib.figure.Figure: The Gantt chart figure
    """
    if len(records) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(60, 6))
        ax.set_title("Pipeline Execution Timeline (no data)")
        return fig

    # Group by op_type using numpy boolean masking (vectorized)
    TYPE_TO_IDX = {name: i for i, name in enumerate(OP_TYPE_NAMES)}
    op_groups = {}
    for op_name in OP_TYPE_NAMES:
        op_idx = TYPE_TO_IDX[op_name]
        mask = records['op_type'] == op_idx
        if np.any(mask):
            op_groups[op_name] = records[mask]

    # Calculate adaptive figure width
    max_time = float(records['end_time'].max())
    time_in_microseconds = max_time / 1000.0
    adaptive_width = max(60, min(300, time_in_microseconds * 10))

    fig, ax = plt.subplots(1, 1, figsize=(adaptive_width, 6))

    # Plot using broken_barh (vectorized, much faster than individual rectangles)
    y_pos = 0
    plotted_types = []

    for op_type in OP_TYPES:
        if op_type not in op_groups:
            continue

        data = op_groups[op_type]
        starts = data['start_time']
        durations = data['end_time'] - data['start_time']

        # broken_barh draws all rectangles at once (vectorized)
        ax.broken_barh(
            list(zip(starts, durations)),
            (y_pos - 0.4, 0.8),
            facecolors=COLORS[op_type],
            edgecolors='black',
            linewidth=0.5,
            alpha=0.8
        )
        plotted_types.append(op_type)
        y_pos += 1

    # Configure axes
    ax.set_ylim(-0.5, len(OP_TYPES) - 0.5)
    ax.set_yticks(range(len(plotted_types)))
    ax.set_yticklabels(plotted_types)
    ax.set_xlim(0, max_time * 1.05)
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_title("Pipeline Execution Timeline", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=COLORS[op_type], edgecolor="black", label=op_type)
        for op_type in plotted_types
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig


def create_gantt_chart_simple(operations):
    """Create simplified Gantt chart with only timeline (no parallelism plot)

    Note: For better performance with large datasets (>100K operations),
    use parse_gantt_data_numpy() + create_gantt_chart_numpy() instead.
    """
    # Group operations by type
    op_groups = defaultdict(list)
    for op in operations:
        op_groups[op["type"]].append(op)

    # Calculate adaptive figure width based on simulation time for optimal visualization
    # Longer simulations need wider charts to maintain readability
    max_time = max(op["end"] for op in operations) if operations else 1000
    # Scale width based on time: 1 microsecond = 10 inches, with reasonable limits
    # This ensures operations don't appear too compressed or too spread out
    time_in_microseconds = max_time / 1000.0  # Convert ns to μs
    adaptive_width = max(60, min(300, time_in_microseconds * 10))  # 60-300 inches range

    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(adaptive_width, 6))

    # Plot Gantt Chart
    y_positions = {}
    y_pos = 0

    for op_type in OP_TYPES:
        if op_type in op_groups:
            y_positions[op_type] = y_pos
            for op in op_groups[op_type]:
                # Use actual duration without artificial manipulation
                rect = patches.Rectangle(
                    (op["start"], y_pos - 0.4),
                    op["duration"],
                    0.8,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=COLORS[op_type],
                    alpha=0.8,
                )
                ax.add_patch(rect)
            y_pos += 1

    # Configure Gantt chart
    ax.set_ylim(-0.5, len(OP_TYPES) - 0.5)
    ax.set_yticks(range(len(OP_TYPES)))
    ax.set_yticklabels(OP_TYPES)
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_title("Pipeline Execution Timeline", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Set x-axis limits
    if operations:
        max_time = max(op["end"] for op in operations)
        ax.set_xlim(0, max_time * 1.05)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=COLORS[op_type], edgecolor="black", label=op_type)
        for op_type in OP_TYPES
        if op_type in op_groups
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig




def print_statistics(operations):
    """
    Print comprehensive pipeline execution statistics and bottleneck analysis.

    Statistics include:
    - Operation type breakdown (count, average duration, total time)
    - Overall execution metrics (total time, parallelism)
    - Bottleneck identification (longest operations, idle times)
    - Pipeline efficiency analysis

    Args:
        operations (list): List of operation dictionaries from parse_gantt_data
    """
    logger.info("=" * 60)
    logger.info("Pipeline Execution Statistics")
    logger.info("=" * 60)

    if not operations:
        logger.info("No operations found in data file")
        return

    # Group by type
    op_groups = defaultdict(list)
    for op in operations:
        op_groups[op["type"]].append(op)

    # Calculate statistics per operation type
    logger.info("Operation Type Statistics:")
    logger.info("-" * 60)
    logger.info(f"{'Type':<15} {'Count':<10} {'Avg Duration':<15} {'Total Time':<15}")
    logger.info("-" * 60)

    total_ops = 0
    for op_type in ["LOAD", "IBUF_READ", "CIM_COMPUTE", "OBUF_WRITE", "STORE"]:
        if op_type in op_groups:
            ops = op_groups[op_type]
            count = len(ops)
            avg_duration = np.mean([op["duration"] for op in ops])
            total_duration = sum(op["duration"] for op in ops)
            total_ops += count
            logger.info(f"{op_type:<15} {count:<10} {avg_duration:<15.2f} {total_duration:<15.2f}")

    # Overall statistics
    logger.info("-" * 60)
    logger.info("Overall Statistics:")
    logger.info("-" * 60)

    total_time = max(op["end"] for op in operations)
    logger.info(f"Total execution time: {total_time:.2f} ns")
    logger.info(f"Total operations: {total_ops}")

    # Calculate parallelism
    time_points = np.linspace(0, total_time, 1000)
    parallelism = []
    for t in time_points:
        active = sum(1 for op in operations if op["start"] <= t <= op["end"])
        parallelism.append(active)

    avg_parallelism = np.mean(parallelism)
    max_parallelism = max(parallelism)

    logger.info(f"Average parallelism: {avg_parallelism:.2f}")
    logger.info(f"Maximum parallelism: {max_parallelism:.0f}")
    logger.info(f"Pipeline efficiency: {avg_parallelism / 5 * 100:.1f}%")

    # Identify bottlenecks
    logger.info("-" * 60)
    logger.info("Bottleneck Analysis:")
    logger.info("-" * 60)

    # Find longest operations
    longest_ops = sorted(operations, key=lambda x: x["duration"], reverse=True)[:5]
    logger.info("Top 5 Longest Operations:")
    for op in longest_ops:
        logger.info(
            f"  {op['type']}-{op['id']}: {op['duration']:.2f} ns "
            f"({op['start']:.2f} - {op['end']:.2f})"
        )

    # Find idle times between operations of same type
    logger.info("Idle Time Analysis:")
    for op_type in ["LOAD", "IBUF_READ", "CIM_COMPUTE", "OBUF_WRITE", "STORE"]:
        if op_type in op_groups:
            ops = sorted(op_groups[op_type], key=lambda x: x["start"])
            idle_times = []
            for i in range(1, len(ops)):
                idle = ops[i]["start"] - ops[i - 1]["end"]
                if idle > 0:
                    idle_times.append(idle)

            if idle_times:
                avg_idle = np.mean(idle_times)
                max_idle = max(idle_times)
                logger.info(f"  {op_type}: Avg idle: {avg_idle:.2f} ns, Max idle: {max_idle:.2f} ns")


if __name__ == "__main__":
    logger.info("Pipeline Visualization Tool")
    logger.info("=" * 60)

    # Parse data
    operations = parse_gantt_data()

    if not operations:
        logger.info("No data found in gantt_data.txt")
        exit(1)

    # Print statistics
    print_statistics(operations)

    # Create visualization
    logger.info("Generating visualization...")
    fig = create_gantt_chart_simple(operations)

    # Save figure
    output_file = "pipeline_gantt.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Visualization saved to: {output_file}")

    # Show plot (optional)
    plt.show()
