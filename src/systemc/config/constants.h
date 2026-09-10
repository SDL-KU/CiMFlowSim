/**
 * @file constants.h
 * @brief Global constants for CiMFlowSim validation and configuration
 *
 * This file defines compile-time constants used throughout the simulator:
 * - Tensor memory system parameters (bit widths, atomic constraints)
 * - File output naming conventions
 * - Pipeline configuration defaults
 */

#ifndef CONFIG_CONSTANTS_H
#define CONFIG_CONSTANTS_H

namespace validation_constants {
// Tiling strategy configuration constants
constexpr int DOUBLE_BUFFER_COUNT = 2;  // Double buffering for IBUF/OBUF

// File output naming conventions
constexpr const char* SIMULATION_LOG_FILENAME = "simulation_log.txt";     // Simulation stdout/stderr log
constexpr const char* GANTT_DATA_FILENAME = "gantt_data.bin";             // Pipeline timeline data (binary)
constexpr const char* GANTT_DATA_CSV_FILENAME = "gantt_data.csv";         // Pipeline timeline data (CSV, legacy)
constexpr const char* DEBUG_STRATEGY_FILENAME = "debug_strategy.txt";     // Strategy debugging
constexpr const char* EXECUTION_TRACE_FILENAME = "execution_trace.log";   // Detailed execution log
constexpr const char* DEPENDENCY_GRAPH_FILENAME = "dependency_graph.txt"; // Dependency graph output
}  // namespace validation_constants

// Pipeline configuration constants
namespace pipeline_config {
// Memory hierarchy naming conventions
constexpr const char* IBUF_PREFIX = "IBUF_";                  // Input buffer naming prefix
constexpr const char* OBUF_PREFIX = "OBUF_";                  // Output buffer naming prefix
constexpr const char* EXTERNAL_MEM_NAME = "EXTERNAL_MEMORY";  // External memory identifier
}  // namespace pipeline_config

namespace config_defaults {
// Default layer parameters when not specified
constexpr int DEFAULT_POOL_HEIGHT = 1;  // No pooling by default
constexpr int DEFAULT_POOL_WIDTH = 1;   // No pooling by default
}  // namespace config_defaults

#endif  // CONFIG_CONSTANTS_H