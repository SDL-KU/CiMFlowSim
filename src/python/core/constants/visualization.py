#!/usr/bin/env python3
"""Visualization constants for CiMFlowSim.

Why this module exists:
- Centralizes visualization configuration
- Defines color schemes and plot parameters
"""

from typing import Dict, Final, List

# ============================================================================
# Color Schemes
# ============================================================================

PIPELINE_COLORS: Final[Dict[str, str]] = {
    "LOAD": "#FF6B6B",
    "IBUF_READ": "#4ECDC4",
    "CIM_COMPUTE": "#45B7D1",
    "OBUF_WRITE": "#96CEB4",
    "STORE": "#FFEAA7",
}
"""
Why: Consistent color mapping for pipeline operations in Gantt charts.
Used by: systemc_runner.py
"""

PIPELINE_OP_TYPES: Final[List[str]] = [
    "STORE",
    "OBUF_WRITE",
    "CIM_COMPUTE",
    "IBUF_READ",
    "LOAD",
]
"""
Why: Ordered list of pipeline operations for Gantt chart rendering.
Order ensures consistent vertical arrangement in visualizations.
Used by: systemc_runner.py
"""

# ============================================================================
# Gantt Chart Configuration
# ============================================================================

GANTT_PDF_MIN_WIDTH_INCHES: Final[int] = 20
"""
Why: Minimum PDF width in inches for Gantt charts.
Prevents text overlap in charts with few operations.
Used by: systemc_runner.py
"""

GANTT_PDF_MAX_WIDTH_INCHES: Final[int] = 200
"""
Why: Maximum PDF width in inches for Gantt charts.
Prevents excessively large files for charts with many operations.
Used by: systemc_runner.py
"""

GANTT_PDF_HEIGHT_INCHES: Final[int] = 6
"""
Why: Fixed PDF height in inches for Gantt charts.
Provides consistent aspect ratio across all charts.
Used by: systemc_runner.py
"""

GANTT_PDF_TIME_SCALE: Final[int] = 8
"""
Why: Pixels per time unit for Gantt chart width calculation.
Balances readability vs file size.
Used by: systemc_runner.py
"""

GANTT_PDF_DPI: Final[int] = 100
"""
Why: DPI for Gantt chart PDF generation.
100 DPI balances quality and file size for technical diagrams.
Used by: systemc_runner.py
"""

# ============================================================================
# Progress Bar Configuration
# ============================================================================

PROGRESS_BAR_LENGTH: Final[int] = 40
"""
Why: Standard progress bar length in characters.
40 characters provides good readability without excessive width.
Used by: simulate.py, pareto_common.py
"""

PROGRESS_UPDATE_PERCENT: Final[int] = 1
"""
Why: Update progress bar every N percent of completion.
1% provides smooth updates while minimizing log output.
Used by: simulate.py, pareto_common.py
"""
