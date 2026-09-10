/**
 * @file gantt_generator.h
 * @brief Gantt chart data generation for pipeline visualization
 *
 * This module generates Gantt chart data files from SystemC simulation timelines:
 * - Extracts operation timing data from pipeline simulation results
 * - Outputs compact binary format for fast Python loading via numpy
 * - Organizes operations in execution order for clear timeline visualization
 *
 * Binary Format (gantt_data.bin):
 * - Header: 4 bytes magic "GANT", 4 bytes version, 4 bytes record count
 * - Records: 32 bytes each (see GanttRecord struct)
 * - ~50% smaller than CSV, 10-50x faster to load in Python via numpy
 *
 * Field sizing rationale:
 * - op_id: uint32 (max observed: ~50000, could grow to millions for large networks)
 * - start/end: double (ns timing can exceed 30M, need >7 significant digits)
 * - memory lines: uint16 (typically small, max ~65535)
 */

#ifndef OUTPUT_GANTT_GENERATOR_H
#define OUTPUT_GANTT_GENERATOR_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../config/constants.h"

namespace output {

/**
 * @brief Binary record structure for Gantt data (32 bytes, packed)
 *
 * Field sizes chosen for:
 * - Minimal size while supporting realistic data ranges
 * - Natural alignment for efficient memory access
 * - Direct numpy dtype mapping in Python
 */
#pragma pack(push, 1)
struct GanttRecord {
    uint8_t op_type;       // Operation type (0=LOAD, 1=IBUF_READ, 2=CIM_COMPUTE, 3=OBUF_WRITE, 4=STORE)
    uint8_t reserved[3];   // Padding for alignment
    uint32_t op_id;        // Operation ID (supports up to 4B operations)
    double start_time;     // Start time in ns (double for precision with large values)
    double end_time;       // End time in ns
    uint16_t src_total;    // Source total lines
    uint16_t src_max;      // Source max lines per bank
    uint16_t dst_total;    // Destination total lines
    uint16_t dst_max;      // Destination max lines per bank
};
#pragma pack(pop)

static_assert(sizeof(GanttRecord) == 32, "GanttRecord must be 32 bytes");

/**
 * @class GanttGenerator
 * @brief Static utility class for generating Gantt chart visualization data
 */
class GanttGenerator {
  public:
    using TimelineData = std::map<std::string, std::vector<std::vector<double>>>;

    /**
     * @brief Generate binary Gantt data file from simulation timeline
     * @param timeline Map of operation names to timing records
     * @param filename Output filename (default: gantt_data.bin)
     */
    static void create_gantt_data(
        const TimelineData& timeline,
        const std::string& filename = validation_constants::GANTT_DATA_FILENAME);

    // Binary format identifiers
    static constexpr uint32_t MAGIC = 0x544E4147;  // "GANT" in little-endian
    static constexpr uint32_t VERSION = 1;

  private:
    static uint8_t operation_name_to_type(const std::string& name);
    static const std::vector<std::string> OPERATION_ORDER;
};

}  // namespace output

#endif  // OUTPUT_GANTT_GENERATOR_H
