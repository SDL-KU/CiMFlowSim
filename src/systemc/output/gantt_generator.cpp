#include "gantt_generator.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace output {

// Define operation order for consistent Gantt chart output
const std::vector<std::string> GanttGenerator::OPERATION_ORDER = {
    "load",   "ibuf_read", "cim_compute", "obuf_write", "store",
    "compute"  // Legacy compute operations for backwards compatibility
};

uint8_t GanttGenerator::operation_name_to_type(const std::string& name) {
    if (name == "load") return 0;
    if (name == "ibuf_read") return 1;
    if (name == "cim_compute" || name == "compute") return 2;
    if (name == "obuf_write") return 3;
    if (name == "store") return 4;
    return 255;  // Unknown
}

void GanttGenerator::create_gantt_data(const TimelineData& timeline, const std::string& filename) {
    // Count total records
    uint32_t total_records = 0;
    for (const auto& operation_name : OPERATION_ORDER) {
        if (timeline.find(operation_name) != timeline.end()) {
            total_records += static_cast<uint32_t>(timeline.at(operation_name).size());
        }
    }

    // Open file in binary mode
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create Gantt data file: " << filename << std::endl;
        return;
    }

    // Write header: magic (4 bytes) + version (4 bytes) + record count (4 bytes)
    file.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
    file.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
    file.write(reinterpret_cast<const char*>(&total_records), sizeof(total_records));

    // Write records
    GanttRecord record;
    std::memset(&record, 0, sizeof(record));  // Zero-initialize including padding

    for (const auto& operation_name : OPERATION_ORDER) {
        if (timeline.find(operation_name) == timeline.end()) {
            continue;
        }

        uint8_t op_type = operation_name_to_type(operation_name);
        const auto& records = timeline.at(operation_name);

        for (const auto& r : records) {
            if (r.size() < 3) continue;  // Need at least op_id, start, end

            record.op_type = op_type;
            record.op_id = static_cast<uint32_t>(r[0]);
            record.start_time = r[1];
            record.end_time = r[2];

            // Memory line info (optional)
            if (r.size() >= 7) {
                record.src_total = static_cast<uint16_t>(r[3]);
                record.src_max = static_cast<uint16_t>(r[4]);
                record.dst_total = static_cast<uint16_t>(r[5]);
                record.dst_max = static_cast<uint16_t>(r[6]);
            } else {
                record.src_total = 0;
                record.src_max = 0;
                record.dst_total = 0;
                record.dst_max = 0;
            }

            file.write(reinterpret_cast<const char*>(&record), sizeof(record));
        }
    }

    file.close();

    // Calculate file size for reporting
    size_t file_size = 12 + (total_records * sizeof(GanttRecord));  // header + records
    double size_mb = file_size / (1024.0 * 1024.0);

    std::cout << "Gantt chart data saved to " << filename
              << " (" << total_records << " records, " << std::fixed << std::setprecision(2)
              << size_mb << " MB)" << std::endl;
}

}  // namespace output
