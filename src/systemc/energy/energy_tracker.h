/**
 * @file energy_tracker.h
 * @brief Energy tracking system for CNN accelerator simulation
 *
 * This module tracks computational operations and memory accesses
 * to enable energy consumption analysis. It collects statistics during
 * simulation and outputs them in JSON format for Python-based energy calculation.
 */

#ifndef ENERGY_TRACKER_H
#define ENERGY_TRACKER_H

#include <atomic>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

#include "../json.hpp"

using json = nlohmann::json;

/**
 * @class EnergyTracker
 * @brief Tracks operations and memory accesses for energy calculation
 *
 * Thread-safe tracking of:
 * - Computational operations (MAC, pooling, activation)
 * - Memory accesses (read/write per hierarchy level)
 * - Data movement (bytes transferred between memory levels)
 */
class EnergyTracker {
  public:
    // Operation types
    enum class OpType {
        MAC,         // Multiply-Accumulate
        POOLING,     // Max/Average pooling
        ACTIVATION,  // ReLU, sigmoid, etc.
        COMPARISON   // For max pooling comparisons
    };

    // Memory hierarchy levels
    enum class MemLevel {
        EXTERNAL,    // DRAM
        IBUF,        // Input buffer (SRAM)
        OBUF,        // Output buffer (SRAM)
        WEIGHT_BUF,  // Weight buffer (SRAM)
        CIM          // Compute-in-memory
    };

    // Access types
    enum class AccessType { READ, WRITE };

  private:
    // Thread-safe counters using atomic
    std::atomic<uint64_t> mac_operations_{0};
    std::atomic<uint64_t> pooling_operations_{0};
    std::atomic<uint64_t> activation_operations_{0};
    std::atomic<uint64_t> comparison_operations_{0};

    // Memory access counters (using mutex for complex updates)
    mutable std::mutex access_mutex_;
    std::map<MemLevel, std::map<AccessType, uint64_t>> memory_accesses_;

    // Data movement tracking (bytes)
    mutable std::mutex movement_mutex_;
    std::map<std::pair<MemLevel, MemLevel>, uint64_t> data_movement_;

    // Additional metrics
    std::atomic<double> total_time_ns_{0};

    // Pipeline operation counts
    std::atomic<uint64_t> pipeline_loads_{0};
    std::atomic<uint64_t> pipeline_ibuf_reads_{0};
    std::atomic<uint64_t> pipeline_cim_computes_{0};
    std::atomic<uint64_t> pipeline_obuf_writes_{0};
    std::atomic<uint64_t> pipeline_stores_{0};

    // Buffer peak usage
    std::atomic<uint64_t> ibuf_peak_lines_{0};
    std::atomic<uint64_t> obuf_peak_lines_{0};

  public:
    // Stage statistics for bottleneck analysis (public for external use)
    // Must be defined before private members that use it
    struct StageStats {
        uint64_t count = 0;           // Number of operations
        double total_busy_ns = 0.0;   // Sum of (end - start) for all operations
        double min_start_ns = 0.0;    // Earliest start time
        double max_end_ns = 0.0;      // Latest end time
        double avg_duration_ns = 0.0; // Average operation duration
        double utilization = 0.0;     // total_busy / (max_end - min_start)
    };

  private:
    // Stage statistics storage
    mutable std::mutex stage_mutex_;
    std::map<std::string, StageStats> stage_statistics_;
    std::string bottleneck_stage_;
    double bottleneck_utilization_ = 0.0;

  public:
    // Constructor
    EnergyTracker() {
        // Initialize memory access counters
        for (auto level : {MemLevel::EXTERNAL, MemLevel::IBUF, MemLevel::OBUF, MemLevel::WEIGHT_BUF,
                           MemLevel::CIM}) {
            memory_accesses_[level][AccessType::READ] = 0;
            memory_accesses_[level][AccessType::WRITE] = 0;
        }
    }

    // ============================================================================
    // Operation Tracking Interface
    // ============================================================================

    /**
     * Track computational operations
     */
    void track_operation(OpType op_type, uint64_t count = 1) {
        switch (op_type) {
            case OpType::MAC:
                mac_operations_ += count;
                break;
            case OpType::POOLING:
                pooling_operations_ += count;
                break;
            case OpType::ACTIVATION:
                activation_operations_ += count;
                break;
            case OpType::COMPARISON:
                comparison_operations_ += count;
                break;
        }
    }

    // ============================================================================
    // Memory Access Tracking Interface
    // ============================================================================

    /**
     * Track memory access
     * @param level Memory hierarchy level
     * @param type Read or Write
     * @param count Number of accesses
     */
    void track_memory_access(MemLevel level, AccessType type, uint64_t count = 1) {
        std::lock_guard<std::mutex> lock(access_mutex_);
        memory_accesses_[level][type] += count;
    }

    /**
     * Track data movement between memory levels
     * @param from Source memory level
     * @param to Destination memory level
     * @param bytes Number of bytes transferred
     */
    void track_data_movement(MemLevel from, MemLevel to, uint64_t bytes) {
        std::lock_guard<std::mutex> lock(movement_mutex_);
        data_movement_[{from, to}] += bytes;
    }

    // ============================================================================
    // Timing Tracking
    // ============================================================================

    void set_total_time_ns(double time_ns) { total_time_ns_ = time_ns; }

    // ============================================================================
    // Pipeline Operation Tracking
    // ============================================================================

    void set_pipeline_operations(uint64_t loads, uint64_t ibuf_reads, uint64_t cim_computes,
                                  uint64_t obuf_writes, uint64_t stores) {
        pipeline_loads_ = loads;
        pipeline_ibuf_reads_ = ibuf_reads;
        pipeline_cim_computes_ = cim_computes;
        pipeline_obuf_writes_ = obuf_writes;
        pipeline_stores_ = stores;
    }

    // ============================================================================
    // Buffer Usage Tracking
    // ============================================================================

    void set_buffer_peak_usage(uint64_t ibuf_peak, uint64_t obuf_peak) {
        ibuf_peak_lines_ = ibuf_peak;
        obuf_peak_lines_ = obuf_peak;
    }

    // ============================================================================
    // Memory Line Statistics (from TensorMemorySystem)
    // ============================================================================

    void set_memory_line_stats(MemLevel level, AccessType type, uint64_t lines) {
        std::lock_guard<std::mutex> lock(access_mutex_);
        memory_accesses_[level][type] = lines;
    }

    // ============================================================================
    // Stage Statistics for Bottleneck Analysis
    // ============================================================================

    /**
     * Set stage statistics computed from timeline data
     * @param stage_name Name of the pipeline stage (load, ibuf_read, cim_compute, obuf_write, store)
     * @param stats Pre-computed statistics for this stage
     */
    void set_stage_statistics(const std::string& stage_name, const StageStats& stats) {
        std::lock_guard<std::mutex> lock(stage_mutex_);
        stage_statistics_[stage_name] = stats;
    }

    /**
     * Set the identified bottleneck stage
     */
    void set_bottleneck(const std::string& stage_name, double utilization) {
        std::lock_guard<std::mutex> lock(stage_mutex_);
        bottleneck_stage_ = stage_name;
        bottleneck_utilization_ = utilization;
    }

    // ============================================================================
    // Statistics Output
    // ============================================================================

    /**
     * Get all statistics as JSON
     */
    json get_statistics_json() const {
        json stats;

        // Operations
        stats["operations"]["mac_ops"] = mac_operations_.load();
        stats["operations"]["pooling_ops"] = pooling_operations_.load();
        stats["operations"]["activation_ops"] = activation_operations_.load();
        stats["operations"]["comparison_ops"] = comparison_operations_.load();

        // Memory accesses (line-based for all buffers)
        {
            std::lock_guard<std::mutex> lock(access_mutex_);

            // IBUF/OBUF: report as line counts (accesses to SRAM lines)
            stats["memory_line_accesses"]["ibuf_read_lines"] =
                memory_accesses_.at(MemLevel::IBUF).at(AccessType::READ);
            stats["memory_line_accesses"]["ibuf_write_lines"] =
                memory_accesses_.at(MemLevel::IBUF).at(AccessType::WRITE);
            stats["memory_line_accesses"]["obuf_read_lines"] =
                memory_accesses_.at(MemLevel::OBUF).at(AccessType::READ);
            stats["memory_line_accesses"]["obuf_write_lines"] =
                memory_accesses_.at(MemLevel::OBUF).at(AccessType::WRITE);

            // EXTERNAL: report as line counts (same as IBUF/OBUF for consistency)
            stats["memory_line_accesses"]["external_read_lines"] =
                memory_accesses_.at(MemLevel::EXTERNAL).at(AccessType::READ);
            stats["memory_line_accesses"]["external_write_lines"] =
                memory_accesses_.at(MemLevel::EXTERNAL).at(AccessType::WRITE);
        }

        // DEPRECATED: Keep old format for backward compatibility (will be removed later)
        {
            std::lock_guard<std::mutex> lock(access_mutex_);
            for (const auto& [level, access_map] : memory_accesses_) {
                std::string level_name = memory_level_to_string(level);
                for (const auto& [type, count] : access_map) {
                    std::string access_name = (type == AccessType::READ) ? "reads" : "writes";
                    stats["memory_accesses"][level_name + "_" + access_name] = count;
                }
            }
        }

        // DEPRECATED: Data movement (bytes) - keep for backward compatibility
        {
            std::lock_guard<std::mutex> lock(movement_mutex_);
            for (const auto& [path, bytes] : data_movement_) {
                std::string from_name = memory_level_to_string(path.first);
                std::string to_name = memory_level_to_string(path.second);
                stats["data_movement"][from_name + "_to_" + to_name + "_bytes"] = bytes;
            }
        }

        // Pipeline operations
        stats["pipeline"]["loads"] = pipeline_loads_.load();
        stats["pipeline"]["ibuf_reads"] = pipeline_ibuf_reads_.load();
        stats["pipeline"]["cim_computes"] = pipeline_cim_computes_.load();
        stats["pipeline"]["obuf_writes"] = pipeline_obuf_writes_.load();
        stats["pipeline"]["stores"] = pipeline_stores_.load();

        // Buffer peak usage
        stats["buffer_usage"]["ibuf_peak_lines"] = ibuf_peak_lines_.load();
        stats["buffer_usage"]["obuf_peak_lines"] = obuf_peak_lines_.load();

        // Timing
        stats["timing"]["total_time_ns"] = total_time_ns_.load();

        // Summary statistics
        uint64_t total_ops =
            mac_operations_ + pooling_operations_ + activation_operations_ + comparison_operations_;
        stats["summary"]["total_operations"] = total_ops;

        uint64_t total_memory_accesses = 0;
        {
            std::lock_guard<std::mutex> lock(access_mutex_);
            for (const auto& [level, access_map] : memory_accesses_) {
                for (const auto& [type, count] : access_map) {
                    total_memory_accesses += count;
                }
            }
        }
        stats["summary"]["total_memory_accesses"] = total_memory_accesses;

        // Stage statistics for bottleneck analysis
        {
            std::lock_guard<std::mutex> lock(stage_mutex_);
            for (const auto& [stage_name, stage_stats] : stage_statistics_) {
                stats["stage_statistics"][stage_name]["count"] = stage_stats.count;
                stats["stage_statistics"][stage_name]["total_busy_ns"] = stage_stats.total_busy_ns;
                stats["stage_statistics"][stage_name]["min_start_ns"] = stage_stats.min_start_ns;
                stats["stage_statistics"][stage_name]["max_end_ns"] = stage_stats.max_end_ns;
                stats["stage_statistics"][stage_name]["avg_duration_ns"] = stage_stats.avg_duration_ns;
                stats["stage_statistics"][stage_name]["utilization"] = stage_stats.utilization;
            }

            // Bottleneck identification
            if (!bottleneck_stage_.empty()) {
                stats["bottleneck"]["stage"] = bottleneck_stage_;
                stats["bottleneck"]["utilization"] = bottleneck_utilization_;
            }
        }

        return stats;
    }

    /**
     * Print statistics to stdout
     */
    void print_statistics() const {
        json stats = get_statistics_json();
        std::cout << "\n=== Energy Statistics JSON ===" << std::endl;
        std::cout << stats.dump(2) << std::endl;
        std::cout << "=== End Energy Statistics JSON ===" << std::endl;
    }

    /**
     * Write statistics to JSON file
     * @param filepath Path to output file (e.g., "simulations/L0_S0/simulation_statistics.json")
     */
    void write_statistics_to_file(const std::string& filepath) const {
        json stats = get_statistics_json();

        std::ofstream out_file(filepath);
        if (!out_file.is_open()) {
            std::cerr << "Warning: Cannot write energy statistics to " << filepath << std::endl;
            return;
        }

        out_file << stats.dump(2) << std::endl;
        out_file.close();
    }

    /**
     * Reset all counters
     */
    void reset() {
        mac_operations_ = 0;
        pooling_operations_ = 0;
        activation_operations_ = 0;
        comparison_operations_ = 0;
        total_time_ns_ = 0;

        pipeline_loads_ = 0;
        pipeline_ibuf_reads_ = 0;
        pipeline_cim_computes_ = 0;
        pipeline_obuf_writes_ = 0;
        pipeline_stores_ = 0;

        ibuf_peak_lines_ = 0;
        obuf_peak_lines_ = 0;

        {
            std::lock_guard<std::mutex> lock(access_mutex_);
            for (auto& [level, access_map] : memory_accesses_) {
                for (auto& [type, count] : access_map) {
                    count = 0;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(movement_mutex_);
            data_movement_.clear();
        }

        {
            std::lock_guard<std::mutex> lock(stage_mutex_);
            stage_statistics_.clear();
            bottleneck_stage_.clear();
            bottleneck_utilization_ = 0.0;
        }
    }

  private:
    /**
     * Convert memory level enum to string for JSON output
     */
    static std::string memory_level_to_string(MemLevel level) {
        switch (level) {
            case MemLevel::EXTERNAL:
                return "external";
            case MemLevel::IBUF:
                return "ibuf";
            case MemLevel::OBUF:
                return "obuf";
            case MemLevel::WEIGHT_BUF:
                return "weight_buf";
            case MemLevel::CIM:
                return "cim";
            default:
                return "unknown";
        }
    }
};

#endif  // ENERGY_TRACKER_H
