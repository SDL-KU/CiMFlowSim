/**
 * @file pipeline_simulator.h
 * @brief Core SystemC-based CNN accelerator pipeline simulator
 *
 * This file defines the main PipelineSimulator class that orchestrates:
 * - Multi-strategy CNN convolution execution (Strategies 1-5)
 * - SystemC-based pipeline modeling with realistic timing
 * - Memory hierarchy simulation (External, IBUF, CIM, OBUF)
 * - Operation dependency management and scheduling
 * - Performance analysis and Gantt chart generation
 *
 * Key Components:
 * - Strategy Pattern: Pluggable memory access strategies
 * - Tensor Memory System: Realistic buffer and external memory simulation
 * - Timeline Tracking: Detailed operation timing for analysis
 * - Debug/Logging: Comprehensive debugging and performance logging
 */

#ifndef PIPELINE_SIMULATOR_H
#define PIPELINE_SIMULATOR_H

// Logging control - set to 0 to disable all logs for better performance
// Enable for detailed debugging, disable for performance measurements
#ifndef ENABLE_VERBOSE_LOGS
#define ENABLE_VERBOSE_LOGS 0  // Disabled for performance (set to 1 for debugging)
#endif

#if ENABLE_VERBOSE_LOGS
#define VERBOSE_LOG(msg)               \
    do {                               \
        std::cout << msg << std::endl; \
    } while (0)
#define OPERATION_LOG(op, msg, idx)           \
    do {                                      \
        std::cout << "[" << op << "]";        \
        if (idx >= 0)                         \
            std::cout << " " << idx;          \
        std::cout << " " << msg << std::endl; \
    } while (0)
#define DEBUG_LOG(msg)                 \
    do {                               \
        std::cout << msg << std::endl; \
    } while (0)
#define INFO_LOG(msg)                  \
    do {                               \
        std::cout << msg << std::endl; \
    } while (0)
#define STRATEGY_LOG(msg)              \
    do {                               \
        std::cout << msg << std::endl; \
    } while (0)
#else
// Empty macros - compiler optimizes these away completely
#define VERBOSE_LOG(msg) \
    do {                 \
    } while (0)
#define OPERATION_LOG(op, msg, idx) \
    do {                            \
    } while (0)
#define DEBUG_LOG(msg) \
    do {               \
    } while (0)
#define INFO_LOG(msg) \
    do {              \
    } while (0)
#define STRATEGY_LOG(msg) \
    do {                  \
    } while (0)
#endif

// Always show results and errors regardless of verbose setting
#define RESULT_LOG(msg)                \
    do {                               \
        std::cout << msg << std::endl; \
    } while (0)
#define ERROR_LOG(msg)                 \
    do {                               \
        std::cerr << msg << std::endl; \
    } while (0)

// Forward declarations and includes for DebugFileLogger
#include <fstream>
#include <iostream>

#include "config/constants.h"
#include "energy/energy_tracker.h"
#include "output/memory_metadata_writer.h"

// Debug file logging system
class DebugFileLogger {
  private:
    inline static std::string debug_filename_ = validation_constants::DEBUG_STRATEGY_FILENAME;
    inline static bool debug_enabled_ = false;

  public:
    static void
    set_debug_file(const std::string& filename = validation_constants::DEBUG_STRATEGY_FILENAME) {
        debug_filename_ = filename;
        debug_enabled_ = true;
    }

    static void disable_debug() { debug_enabled_ = false; }

    static void log_debug(const std::string& message) {
        if (!debug_enabled_)
            return;

        std::ofstream debug_file(debug_filename_, std::ios::app);
        if (debug_file.is_open()) {
            debug_file << "[DEBUG] " << message << std::endl;
            debug_file.close();
        }
    }

    static void log_obuf_write_start(int total_writes) {
        log_debug("OBUF_WRITE thread started: total_obuf_writes=" + std::to_string(total_writes));
    }

    static void log_obuf_write_iteration(int idx, int total) {
        log_debug("OBUF_WRITE iteration: idx=" + std::to_string(idx) + "/" + std::to_string(total));
    }

    static void log_obuf_write_dependencies_satisfied(int idx) {
        log_debug("OBUF_WRITE dependencies satisfied for idx=" + std::to_string(idx));
    }

    static void log_obuf_write_before_region_access(int idx) {
        log_debug("About to call get_obuf_write_region for idx=" + std::to_string(idx));
    }

    static void log_obuf_write_after_region_access(int idx) {
        log_debug("get_obuf_write_tensor_access returned for idx=" + std::to_string(idx));
    }
};

#include <systemc.h>

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cnn_strategy.h"
#include "memory/tensor_memory_system.h"

// Operation types for type-safe operation handling
// OperationType is defined in cnn_strategy.h

// Convert OperationType to string for logging
inline const char* operation_type_to_string(OperationType type) {
    switch (type) {
        case OperationType::LOAD:
            return "load";
        case OperationType::IBUF_READ:
            return "ibuf_read";
        case OperationType::CIM_COMPUTE:
            return "cim_compute";
        case OperationType::OBUF_WRITE:
            return "obuf_write";
        case OperationType::STORE:
            return "store";
        default:
            return "unknown";
    }
}

// CNN Parameters
struct CNNParams {
    int H, W, C;     // Input dimensions (padding already included)
    int R, S, M;     // Filter dimensions
    int stride;      // Convolution stride
    int batch_size;  // Batch size
    int input_bitwidth, output_bitwidth;
    int pool_height, pool_width;  // Pooling kernel dimensions

    // Derived parameters
    int P, Q;                // Convolution output dimensions
    int P_pooled, Q_pooled;  // Final output dimensions after pooling

    CNNParams(int h, int w, int c, int r, int s, int m, int str = 1, int batch = 1, int in_bits = 8,
              int out_bits = 8, int pool_h = 1, int pool_w = 1)
        : H(h),
          W(w),
          C(c),
          R(r),
          S(s),
          M(m),
          stride(str),
          batch_size(batch),
          input_bitwidth(in_bits),
          output_bitwidth(out_bits),
          pool_height(pool_h),
          pool_width(pool_w) {
        P = (H - R) / stride + 1;
        Q = (W - S) / stride + 1;
        P_pooled = P / pool_height;
        Q_pooled = Q / pool_width;
    }
};

// Hardware Configuration
struct HWConfig {
    double compute_time_ns;  // Pure computation time (MAC operations)

    // Memory configurations for different buffer types
    MemoryTimingConfig ibuf_config;      // IBUF memory configuration
    MemoryTimingConfig obuf_config;      // OBUF memory configuration
    MemoryTimingConfig external_config;  // External memory configuration

    // Bank and line configurations for different memory types
    BankLineConfig ibuf_bank_config;      // IBUF bank/line configuration
    BankLineConfig obuf_bank_config;      // OBUF bank/line configuration
    BankLineConfig external_bank_config;  // External memory bank/line configuration

    HWConfig(double compute, const MemoryTimingConfig& ibuf, const MemoryTimingConfig& obuf,
             const MemoryTimingConfig& external, const BankLineConfig& ibuf_bank,
             const BankLineConfig& obuf_bank, const BankLineConfig& external_bank)
        : compute_time_ns(compute),
          ibuf_config(ibuf),
          obuf_config(obuf),
          external_config(external),
          ibuf_bank_config(ibuf_bank),
          obuf_bank_config(obuf_bank),
          external_bank_config(external_bank) {}
};

// Main Pipeline Simulator
SC_MODULE(PipelineSimulator) {
    // Parameters
    CNNParams params;
    HWConfig hw_config;
    std::string output_dir;

    // Strategy implementation
    std::unique_ptr<CNNStrategy> strategy;

    // Multiple buffer instances (using standalone unified memory system)
    std::vector<std::unique_ptr<TensorMemorySystem>> ibufs;
    std::vector<std::unique_ptr<TensorMemorySystem>> obufs;
    std::unique_ptr<TensorMemorySystem> external_memory;  // External DRAM/storage

    // Events for thread synchronization

    SC_HAS_PROCESS(PipelineSimulator);

    // Progress counters (public for JSON interface)
    int load_idx, ibuf_read_idx, cim_compute_idx, obuf_write_idx, store_idx;
    int total_loads, total_ibuf_reads, total_cim_computes, total_obuf_writes, total_stores;

    // Operation completion events (single event per operation type since they're sequential)
    // Note: Using std::map instead of unordered_map - with only 5 operation types,
    // tree traversal is faster than hash computation overhead
    std::map<OperationType, sc_event> last_completed_event;

    // Timeline recording
    struct TimelineRecord {
        OperationType operation;  // Use enum instead of string for efficiency
        int id;
        sc_time start_time;
        sc_time end_time;
        std::vector<Dependency> dependencies;  // Dependencies for this operation

        // Memory line access information (total and max per bank)
        int source_total_lines = 0;          // Total lines read from source (for energy)
        int source_max_lines_per_bank = 0;   // Max lines per bank read from source (for timing)
        int dest_total_lines = 0;            // Total lines written to destination (for energy)
        int dest_max_lines_per_bank = 0;     // Max lines per bank written to destination (for timing)
    };
    std::vector<TimelineRecord> timeline;

    // Execution trace logging
    std::ofstream execution_trace_file;

    // Energy tracking
    EnergyTracker energy_tracker;

    // Memory metadata for visualization (lightweight alternative to simulation_log.txt)
    output::MemoryMetadataWriter memory_metadata;

    // Macro-based logging replaces log_operation function for better performance

    // Convert sc_time to nanoseconds for display
    double to_ns(const sc_time& t) const {
        return t.to_seconds() * 1e9;
    }

    // Helper to create multiple buffers with consistent naming and specific timing config
    void create_buffers(std::vector<std::unique_ptr<TensorMemorySystem>> & buffers,
                        const std::string& prefix, int count,
                        const MemoryTimingConfig& timing_config,
                        const BankLineConfig& bank_line_config) {
        for (int i = 0; i < count; i++) {
            std::string buf_name = prefix + std::to_string(i);
            buffers.push_back(std::make_unique<TensorMemorySystem>(
                buf_name.c_str(), PortConfiguration::single_port(), timing_config,
                bank_line_config));
        }
    }

    // Dependency-based scheduling (replacing buffer access count control)
    // Since operations are performed sequentially, we only need the last completed index
    std::map<OperationType, int> last_completed_idx;

    // Dependency getter function map to avoid switch statements
    typedef std::function<std::vector<Dependency>(CNNStrategy*, int, const CNNParams&)>
        DependencyGetter;
    std::map<OperationType, DependencyGetter> dependency_getters;

    // Constructor for Tiling Strategy
    PipelineSimulator(sc_module_name name, const CNNParams& p, const HWConfig& hw,
                      const TilingConfig& tiling, const std::string& output_directory = ".",
                      bool save_logs = false)
        : sc_module(name),
          params(p),
          hw_config(hw),
          output_dir(output_directory),
          load_idx(0),
          ibuf_read_idx(0),
          cim_compute_idx(0),
          obuf_write_idx(0),
          store_idx(0) {
        // Create tiling strategy instance
        strategy.reset(create_tiling_strategy(tiling));
        if (!strategy) {
            SC_REPORT_ERROR("PipelineSimulator", "Failed to create tiling strategy");
            return;
        }

        // Calculate total operations based on tiling configuration
        total_loads = strategy->get_total_loads(params);
        int total_computes = strategy->get_total_computes(params);
        total_stores = strategy->get_total_stores(params);

        total_ibuf_reads = total_computes;
        total_cim_computes = total_computes;
        total_obuf_writes = strategy->get_total_obuf_writes(params);

        // Initialize execution trace file only if save_logs is enabled
        if (save_logs) {
            std::string trace_path = output_dir + "/" + validation_constants::EXECUTION_TRACE_FILENAME;
            execution_trace_file.open(trace_path);
            if (!execution_trace_file.is_open()) {
                SC_REPORT_WARNING("PipelineSimulator",
                                  ("Failed to open execution trace file: " + trace_path).c_str());
            }
        }

        // Create memory subsystems with strategy-specific configuration
        int ibuf_count = strategy->get_ibuf_count();
        int obuf_count = strategy->get_obuf_count();

        // Create input and output buffers using helper function with specific configs
        create_buffers(ibufs, pipeline_config::IBUF_PREFIX, ibuf_count, hw_config.ibuf_config,
                       hw_config.ibuf_bank_config);
        create_buffers(obufs, pipeline_config::OBUF_PREFIX, obuf_count, hw_config.obuf_config,
                       hw_config.obuf_bank_config);

        // Create external memory (SINGLE_PORT - typical DRAM configuration)
        external_memory = std::make_unique<TensorMemorySystem>(
            pipeline_config::EXTERNAL_MEM_NAME, PortConfiguration::single_port(),
            hw_config.external_config, hw_config.external_bank_config);

        // Initialize dependency getters (same as numbered strategies)
        dependency_getters[OperationType::LOAD] = [](CNNStrategy* s, int idx, const CNNParams& p) {
            return s->get_load_dependencies(idx, p);
        };
        dependency_getters[OperationType::IBUF_READ] = [](CNNStrategy* s, int idx,
                                                          const CNNParams& p) {
            return s->get_ibuf_read_dependencies(idx, p);
        };
        dependency_getters[OperationType::CIM_COMPUTE] = [](CNNStrategy* s, int idx,
                                                            const CNNParams& p) {
            return s->get_compute_dependencies(idx, p);
        };
        dependency_getters[OperationType::OBUF_WRITE] = [](CNNStrategy* s, int idx,
                                                           const CNNParams& p) {
            return s->get_obuf_write_dependencies(idx, p);
        };
        dependency_getters[OperationType::STORE] = [](CNNStrategy* s, int idx, const CNNParams& p) {
            return s->get_store_dependencies(idx, p);
        };

        // Register tensors in external memory
        std::vector<int> input_shape = {params.batch_size, params.C, params.H, params.W};
        std::vector<int> output_shape = {params.batch_size, params.M, params.P_pooled,
                                         params.Q_pooled};
        external_memory->register_tensor("input_tensor", input_shape, params.input_bitwidth);
        external_memory->register_tensor("output_tensor", output_shape, params.output_bitwidth);

        register_strategy_buffers();

        // Initialize memory metadata writer with hardware configuration
        memory_metadata.set_hardware_config(
            hw_config.external_bank_config.num_banks,
            hw_config.external_bank_config.bits_per_line,
            hw_config.ibuf_bank_config.num_banks,
            hw_config.ibuf_bank_config.bits_per_line,
            hw_config.obuf_bank_config.num_banks,
            hw_config.obuf_bank_config.bits_per_line
        );

        // Start pipeline threads
        SC_THREAD(load_thread);
        SC_THREAD(ibuf_read_thread);
        SC_THREAD(cim_compute_thread);
        SC_THREAD(obuf_write_thread);
        SC_THREAD(store_thread);

        INFO_LOG("Pipeline Simulator initialized (Tiling Strategy):");
        INFO_LOG("  Tiling: " << tiling.output_tile_p << "×" << tiling.output_tile_q << " output, "
                              << tiling.input_tile_h << "×" << tiling.input_tile_w << " input, "
                              << tiling.output_tile_count << " tiles");
        INFO_LOG("  Total operations: L=" << total_loads << ", IR=" << total_ibuf_reads << ", CC="
                                          << total_cim_computes << ", OW=" << total_obuf_writes
                                          << ", S=" << total_stores);
        INFO_LOG("  Buffers: " << ibuf_count << " IBUFs, " << obuf_count << " OBUFs");
    }

    // Destructor
    ~PipelineSimulator() {
        // Close execution trace file
        if (execution_trace_file.is_open()) {
            execution_trace_file.close();
        }
        // unique_ptr will automatically clean up buffers
    }

    // Helper function to log tensor access
    void log_tensor_access(const std::string& operation, int op_id, const std::string& action,
                           const std::string& tensor_id,
                           const std::vector<std::pair<int, int>>& ranges,
                           int buffer_idx = -1) {
        if (execution_trace_file.is_open()) {
            execution_trace_file << to_ns(sc_time_stamp()) << " " << operation << "[" << op_id
                                 << "] " << action << " " << tensor_id;

            // Add buffer index if provided
            if (buffer_idx >= 0) {
                // Extract buffer type from action (e.g., "READ_IBUF" -> "IBUF")
                if (action.find("IBUF") != std::string::npos) {
                    execution_trace_file << " @IBUF[" << buffer_idx << "]";
                } else if (action.find("OBUF") != std::string::npos) {
                    execution_trace_file << " @OBUF[" << buffer_idx << "]";
                }
            }

            execution_trace_file << ": ";
            for (size_t i = 0; i < ranges.size(); i++) {
                if (i > 0)
                    execution_trace_file << "×";
                execution_trace_file << "[" << ranges[i].first << ":" << ranges[i].second << "]";
            }
            execution_trace_file << std::endl;
        }
    }

    // Load thread
    void load_thread() {
        while (load_idx < total_loads) {
            // Get target buffer index from strategy
            int ibuf_idx = strategy->get_load_ibuf_idx(load_idx, params);

            TensorMemorySystem* target_ibuf = ibufs[ibuf_idx].get();

            // Wait for dependencies to be satisfied (event-driven approach)
            auto dependencies = wait_for_dependencies(OperationType::LOAD, load_idx);

            sc_time start_time = sc_time_stamp();
            OPERATION_LOG("LOAD",
                          "Starting load " << load_idx << " at " << to_ns(start_time) << "ns", -1);

            // Generate external read access using new strategy function
            auto [external_tensor_id, external_ranges] =
                strategy->get_external_read_access(load_idx, params);
            // Generate IBUF write access using new strategy function
            auto [ibuf_tensor_id, ibuf_ranges] = strategy->get_ibuf_write_access(load_idx, params);

            // Log tensor accesses with buffer index
            log_tensor_access("LOAD", load_idx, "READ_EXTERNAL", external_tensor_id,
                              external_ranges);
            log_tensor_access("LOAD", load_idx, "WRITE_IBUF", ibuf_tensor_id, ibuf_ranges,
                              ibuf_idx);

            // Streaming transfer: read from external memory while writing to IBUF
            sc_event external_read_event, ibuf_write_event;
            external_memory->read(external_tensor_id, external_ranges, external_read_event);
            target_ibuf->write(ibuf_tensor_id, ibuf_ranges, ibuf_write_event);

            // Wait for both operations to complete (DMA-style streaming)
            wait(external_read_event & ibuf_write_event);

            sc_time end_time = sc_time_stamp();

            OPERATION_LOG("LOAD",
                          "Completed load " << load_idx << " to IBUF[" << ibuf_idx << "] at "
                                            << to_ns(end_time) << "ns",
                          -1);

            // Calculate memory line accesses (unified for energy tracking and Gantt visualization)
            auto ext_info = external_memory->get_line_access_info_for_tensor_range(external_tensor_id, external_ranges);
            auto ibuf_info = target_ibuf->get_line_access_info_for_tensor_range(ibuf_tensor_id, ibuf_ranges);

            // Track energy using line-based counts (Python energy calculator will multiply by bits_per_line)
            energy_tracker.track_memory_access(EnergyTracker::MemLevel::EXTERNAL,
                                              EnergyTracker::AccessType::READ, ext_info.total_lines);
            energy_tracker.track_memory_access(EnergyTracker::MemLevel::IBUF,
                                              EnergyTracker::AccessType::WRITE, ibuf_info.total_lines);

            // Record timeline with memory line information (total and max per bank)
            record_operation_timeline(OperationType::LOAD, load_idx, start_time, end_time, dependencies,
                                     ext_info.total_lines, ext_info.max_lines_per_bank,
                                     ibuf_info.total_lines, ibuf_info.max_lines_per_bank);

            // Mark operation as completed (replacing buffer access count system)
            mark_operation_completed(OperationType::LOAD, load_idx);

            load_idx++;
        }
    }

    // IBUF Read thread (CIM register input)
    void ibuf_read_thread() {
        while (ibuf_read_idx < total_ibuf_reads) {
            // Wait for dependencies: load[i] complete AND cim_compute[i-2] complete (reverse
            // dependency)
            auto dependencies = wait_for_dependencies(OperationType::IBUF_READ, ibuf_read_idx);

            sc_time start_time = sc_time_stamp();

            OPERATION_LOG(
                "IBUF_READ",
                "Starting IBUF read " << ibuf_read_idx << " at " << to_ns(start_time) << "ns", -1);

            // Get buffer index from strategy
            int read_ibuf_idx = strategy->get_compute_ibuf_idx(ibuf_read_idx, params);

            TensorMemorySystem* source_ibuf = ibufs[read_ibuf_idx].get();

            // Generate IBUF read access using new strategy function
            auto [input_tensor_id, input_ranges] =
                strategy->get_ibuf_read_access(ibuf_read_idx, params);

            // Log tensor access with buffer index
            log_tensor_access("IBUF_READ", ibuf_read_idx, "READ_IBUF", input_tensor_id,
                              input_ranges, read_ibuf_idx);

            // Perform IBUF read (async) - TensorMemorySystem version
            sc_event ibuf_read_completion_event;
            source_ibuf->read(input_tensor_id, input_ranges, ibuf_read_completion_event);

            // Wait for read operation completion
            wait(ibuf_read_completion_event);

            sc_time end_time = sc_time_stamp();

            OPERATION_LOG("IBUF_READ",
                          "Completed IBUF read " << ibuf_read_idx << " from IBUF[" << read_ibuf_idx
                                                 << "] at " << to_ns(end_time) << "ns",
                          -1);

            // Calculate memory line accesses (unified for energy tracking and Gantt visualization)
            auto ibuf_info = source_ibuf->get_line_access_info_for_tensor_range(input_tensor_id, input_ranges);

            // Track energy using line-based counts
            energy_tracker.track_memory_access(EnergyTracker::MemLevel::IBUF,
                                              EnergyTracker::AccessType::READ, ibuf_info.total_lines);

            // Record timeline with memory line information (source: IBUF, dest: CIM registers - not tracked as lines)
            record_operation_timeline(OperationType::IBUF_READ, ibuf_read_idx, start_time,
                                      end_time, dependencies,
                                      ibuf_info.total_lines, ibuf_info.max_lines_per_bank,
                                      0, 0);

            // Mark operation as completed
            mark_operation_completed(OperationType::IBUF_READ, ibuf_read_idx);

            ibuf_read_idx++;
        }
    }

    // CIM Compute thread (pure computation)
    void cim_compute_thread() {
        while (cim_compute_idx < total_cim_computes) {
            // Wait for dependencies: ibuf_read[i] complete
            auto dependencies = wait_for_dependencies(OperationType::CIM_COMPUTE, cim_compute_idx);

            sc_time start_time = sc_time_stamp();

            OPERATION_LOG("CIM_COMPUTE",
                          "Starting CIM compute " << cim_compute_idx << " at " << to_ns(start_time)
                                                  << "ns",
                          -1);

            // Track MAC operations for this compute tile
            // For each output pixel, we perform R*S*C*M MAC operations
            // (R*S*C for each of M output channels)
            uint64_t macs_per_compute = params.R * params.S * params.C * params.M;
            energy_tracker.track_operation(EnergyTracker::OpType::MAC, macs_per_compute);

            // Pure computation time
            wait(hw_config.compute_time_ns, SC_NS);

            sc_time end_time = sc_time_stamp();

            OPERATION_LOG("CIM_COMPUTE",
                          "Completed CIM compute " << cim_compute_idx << " in "
                                                   << hw_config.compute_time_ns << "ns at "
                                                   << to_ns(end_time) << "ns",
                          -1);

            // Record timeline (no memory access for pure compute)
            record_operation_timeline(OperationType::CIM_COMPUTE, cim_compute_idx, start_time,
                                      end_time, dependencies, 0, 0, 0, 0);

            // Mark operation as completed
            mark_operation_completed(OperationType::CIM_COMPUTE, cim_compute_idx);

            cim_compute_idx++;
        }
    }

    // OBUF Write thread (CIM register output) - Pooling group based
    void obuf_write_thread() {
        while (obuf_write_idx < total_obuf_writes) {
            // Wait for dependencies (pooling group handled by strategy-specific dependencies)
            auto dependencies = wait_for_dependencies(OperationType::OBUF_WRITE, obuf_write_idx);

            sc_time start_time = sc_time_stamp();

            OPERATION_LOG("OBUF_WRITE",
                          "Starting OBUF write " << obuf_write_idx << " (pooling group) at "
                                                 << to_ns(start_time) << "ns",
                          -1);

            // Get buffer index from strategy
            int write_obuf_idx = strategy->get_compute_obuf_idx(obuf_write_idx, params);

            TensorMemorySystem* target_obuf = obufs[write_obuf_idx].get();

            // Generate pooled output region using strategy-specific function
            auto [output_tensor_id, output_ranges] =
                strategy->get_obuf_write_access(obuf_write_idx, params);

            // Log tensor access with buffer index
            log_tensor_access("OBUF_WRITE", obuf_write_idx, "WRITE_OBUF", output_tensor_id,
                              output_ranges, write_obuf_idx);

            // Track pooling operations (if pooling is enabled)
            if (params.pool_height > 1 || params.pool_width > 1) {
                // Calculate pooled output elements for operation counting
                uint64_t write_elements = 1;
                for (const auto& range : output_ranges) {
                    write_elements *= (range.second - range.first);
                }
                // Each pooled output requires pool_height * pool_width comparisons/operations
                uint64_t pooling_ops = write_elements * params.pool_height * params.pool_width;
                energy_tracker.track_operation(EnergyTracker::OpType::POOLING, pooling_ops);
            }

            // Perform OBUF write (async) - TensorMemorySystem version
            sc_event obuf_write_completion_event;
            target_obuf->write(output_tensor_id, output_ranges, obuf_write_completion_event);

            // Wait for write operation completion
            wait(obuf_write_completion_event);

            sc_time end_time = sc_time_stamp();

            OPERATION_LOG("OBUF_WRITE",
                          "Completed OBUF write " << obuf_write_idx << " to OBUF[" << write_obuf_idx
                                                  << "] at " << to_ns(end_time) << "ns",
                          -1);

            // Calculate memory line accesses (unified for energy tracking and Gantt visualization)
            auto obuf_info = target_obuf->get_line_access_info_for_tensor_range(output_tensor_id, output_ranges);

            // Track energy using line-based counts
            energy_tracker.track_memory_access(EnergyTracker::MemLevel::OBUF,
                                              EnergyTracker::AccessType::WRITE, obuf_info.total_lines);

            // Record timeline with memory line information (source: CIM registers - not tracked as lines, dest: OBUF)
            record_operation_timeline(OperationType::OBUF_WRITE, obuf_write_idx, start_time,
                                      end_time, dependencies,
                                      0, 0,
                                      obuf_info.total_lines, obuf_info.max_lines_per_bank);

            // Mark operation as completed
            mark_operation_completed(OperationType::OBUF_WRITE, obuf_write_idx);

            obuf_write_idx++;
        }
    }

    // Store thread
    void store_thread() {
        while (store_idx < total_stores) {
            // Wait for dependencies to be satisfied (event-driven approach)
            auto dependencies = wait_for_dependencies(OperationType::STORE, store_idx);

            sc_time start_time = sc_time_stamp();

            OPERATION_LOG(
                "STORE", "Starting store " << store_idx << " at " << to_ns(start_time) << "ns", -1);

            // Get target buffer index from strategy
            int obuf_idx = strategy->get_store_obuf_idx(store_idx, params);

            TensorMemorySystem* source_obuf = obufs[obuf_idx].get();

            // Dependencies are already handled by wait_for_dependencies above
            // No need for buffer polling - pure event-driven approach

            // Generate OBUF read access using new strategy function
            auto [obuf_tensor_id, obuf_ranges] = strategy->get_obuf_read_access(store_idx, params);
            // Generate external write access using new strategy function
            auto [external_tensor_id, external_ranges] =
                strategy->get_external_write_access(store_idx, params);

            // Log tensor accesses with buffer index
            log_tensor_access("STORE", store_idx, "READ_OBUF", obuf_tensor_id, obuf_ranges, obuf_idx);
            log_tensor_access("STORE", store_idx, "WRITE_EXTERNAL", external_tensor_id,
                              external_ranges);

            // Streaming transfer: read from OBUF while writing to external memory
            sc_event obuf_read_event, external_write_event;
            source_obuf->read(obuf_tensor_id, obuf_ranges, obuf_read_event);
            external_memory->write(external_tensor_id, external_ranges, external_write_event);

            // Wait for both operations to complete (DMA-style streaming)
            wait(obuf_read_event & external_write_event);

            sc_time end_time = sc_time_stamp();

            OPERATION_LOG("STORE",
                          "Completed store " << store_idx << " from OBUF[" << obuf_idx << "] at "
                                             << to_ns(end_time) << "ns",
                          -1);

            // Calculate memory line accesses (unified for energy tracking and Gantt visualization)
            auto obuf_info = source_obuf->get_line_access_info_for_tensor_range(obuf_tensor_id, obuf_ranges);
            auto ext_info = external_memory->get_line_access_info_for_tensor_range(external_tensor_id, external_ranges);

            // Track energy using line-based counts
            energy_tracker.track_memory_access(EnergyTracker::MemLevel::OBUF,
                                              EnergyTracker::AccessType::READ, obuf_info.total_lines);
            energy_tracker.track_memory_access(EnergyTracker::MemLevel::EXTERNAL,
                                              EnergyTracker::AccessType::WRITE, ext_info.total_lines);

            // Record timeline with memory line information (source: OBUF, dest: External)
            record_operation_timeline(OperationType::STORE, store_idx, start_time, end_time, dependencies,
                                     obuf_info.total_lines, obuf_info.max_lines_per_bank,
                                     ext_info.total_lines, ext_info.max_lines_per_bank);

            // Mark operation as completed (replacing buffer access count system)
            mark_operation_completed(OperationType::STORE, store_idx);

            store_idx++;
        }
    }

    // Get simulation results (optimized with enum-based storage)
    std::map<std::string, std::vector<std::vector<double>>> get_timeline() const {
        std::map<std::string, std::vector<std::vector<double>>> result;

        // First pass: count operations for each type to reserve space
        std::map<OperationType, size_t> operation_counts;
        for (const auto& record : timeline) {
            operation_counts[record.operation]++;
        }

        // Reserve space for each operation type
        for (const auto& pair : operation_counts) {
            std::string op_name = operation_type_to_string(pair.first);
            result[op_name].reserve(pair.second);
        }

        // Second pass: populate the data
        for (const auto& record : timeline) {
            std::string op_name = operation_type_to_string(record.operation);
            result[op_name].push_back({
                static_cast<double>(record.id),
                record.start_time.to_seconds() * 1e9,                    // Convert to ns for output
                record.end_time.to_seconds() * 1e9,                      // Convert to ns for output
                static_cast<double>(record.source_total_lines),          // Source total lines (energy)
                static_cast<double>(record.source_max_lines_per_bank),   // Source max per bank (timing)
                static_cast<double>(record.dest_total_lines),            // Dest total lines (energy)
                static_cast<double>(record.dest_max_lines_per_bank)      // Dest max per bank (timing)
            });
        }

        return result;
    }

    // Write dependency information to separate file (for critical path analysis)
    void write_dependencies_to_file(const std::string& filename) const {
        std::ofstream file(filename);  // Create new file (not append)
        if (!file.is_open()) {
            std::cerr << "Error: Cannot create dependency graph file: " << filename << std::endl;
            return;
        }

        file << "# Pipeline Dependency Graph" << std::endl;
        file << "# Format: operation_type operation_id depends_on: op_type op_id, ..." << std::endl;
        file << "# Use for critical path analysis and pipeline optimization" << std::endl;

        for (const auto& record : timeline) {
            if (!record.dependencies.empty()) {
                std::string op_name = operation_type_to_string(record.operation);
                std::transform(op_name.begin(), op_name.end(), op_name.begin(), ::toupper);

                file << op_name << " " << record.id << " depends_on: ";
                for (size_t i = 0; i < record.dependencies.size(); i++) {
                    std::string dep_op_name = operation_type_to_string(record.dependencies[i].operation_type);
                    std::transform(dep_op_name.begin(), dep_op_name.end(), dep_op_name.begin(), ::toupper);

                    file << dep_op_name << " " << record.dependencies[i].operation_id;
                    if (i < record.dependencies.size() - 1) {
                        file << ", ";
                    }
                }
                file << std::endl;
            }
        }

        file.close();
    }

  private:
    // Register tensors in buffers based on strategy-specific shapes
    void register_strategy_buffers() {
        // Get buffer shapes from strategy
        std::vector<int> ibuf_shape = strategy->get_ibuf_shape(params);
        std::vector<int> obuf_shape = strategy->get_obuf_shape(params);

        // Register input tensor in all input buffers
        for (auto& ibuf : ibufs) {
            ibuf->register_tensor("input_tensor", ibuf_shape, params.input_bitwidth);
        }

        // Register output tensor in all output buffers
        for (auto& obuf : obufs) {
            obuf->register_tensor("output_tensor", obuf_shape, params.output_bitwidth);
        }
    }

    // Helper function to get dependencies for operation type
    std::vector<Dependency> get_dependencies_for_operation(OperationType operation_type,
                                                           int operation_id) {
        auto it = dependency_getters.find(operation_type);
        if (it != dependency_getters.end()) {
            return it->second(strategy.get(), operation_id, params);
        }
        return {};
    }

    // Wait for specific dependencies to be satisfied (event-driven approach)
    std::vector<Dependency> wait_for_dependencies(OperationType operation_type, int operation_id) {
        std::vector<Dependency> dependencies =
            get_dependencies_for_operation(operation_type, operation_id);

        // Wait for each unsatisfied dependency
        for (const auto& dep : dependencies) {
            while (!is_operation_completed(dep.operation_type, dep.operation_id)) {
                wait(last_completed_event[dep.operation_type]);
            }
        }

        return dependencies;
    }

    bool is_operation_completed(OperationType operation_type, int operation_id) {
        auto it = last_completed_idx.find(operation_type);
        if (it != last_completed_idx.end()) {
            return it->second >= operation_id;
        }
        return false;
    }

    void mark_operation_completed(OperationType operation_type, int operation_id) {
        last_completed_idx[operation_type] = operation_id;
        last_completed_event[operation_type].notify(
            SC_ZERO_TIME);  // Notify completion with delta delay
    }

    // Common helper function for timeline recording
    void record_operation_timeline(OperationType operation_type, int operation_id,
                                   sc_time start_time, sc_time end_time,
                                   const std::vector<Dependency>& dependencies = {},
                                   int src_total = 0, int src_max = 0,
                                   int dst_total = 0, int dst_max = 0) {
        // Using initializer list for efficient construction
        TimelineRecord record{operation_type, operation_id, start_time, end_time, dependencies};
        record.source_total_lines = src_total;
        record.source_max_lines_per_bank = src_max;
        record.dest_total_lines = dst_total;
        record.dest_max_lines_per_bank = dst_max;
        timeline.push_back(record);
    }

    /**
     * @brief Compute stage statistics from timeline data for bottleneck analysis
     *
     * For each pipeline stage (LOAD, IBUF_READ, CIM_COMPUTE, OBUF_WRITE, STORE):
     * - count: number of operations
     * - total_busy_ns: sum of (end - start) for all operations
     * - min_start_ns / max_end_ns: span of the stage
     * - utilization: total_busy / span (higher = busier = potential bottleneck)
     */
    void compute_stage_statistics() {
        if (timeline.empty()) return;

        // Group timeline records by operation type
        std::map<OperationType, std::vector<const TimelineRecord*>> stage_records;
        for (const auto& record : timeline) {
            stage_records[record.operation].push_back(&record);
        }

        // Compute statistics for each stage
        double max_utilization = 0.0;
        std::string bottleneck_stage;

        for (const auto& [op_type, records] : stage_records) {
            if (records.empty()) continue;

            std::string stage_name = operation_type_to_string(op_type);
            EnergyTracker::StageStats stats;
            stats.count = records.size();

            double min_start = std::numeric_limits<double>::max();
            double max_end = 0.0;
            double total_busy = 0.0;

            for (const auto* rec : records) {
                double start_ns = rec->start_time.to_seconds() * 1e9;
                double end_ns = rec->end_time.to_seconds() * 1e9;
                double duration = end_ns - start_ns;

                total_busy += duration;
                if (start_ns < min_start) min_start = start_ns;
                if (end_ns > max_end) max_end = end_ns;
            }

            stats.total_busy_ns = total_busy;
            stats.min_start_ns = min_start;
            stats.max_end_ns = max_end;
            stats.avg_duration_ns = (stats.count > 0) ? total_busy / stats.count : 0.0;

            // Utilization = total busy time / span
            double span = max_end - min_start;
            stats.utilization = (span > 0) ? total_busy / span : 0.0;

            energy_tracker.set_stage_statistics(stage_name, stats);

            // Track bottleneck (highest utilization stage)
            if (stats.utilization > max_utilization) {
                max_utilization = stats.utilization;
                bottleneck_stage = stage_name;
            }
        }

        // Set bottleneck
        if (!bottleneck_stage.empty()) {
            energy_tracker.set_bottleneck(bottleneck_stage, max_utilization);
        }
    }

  public:
    // Print final statistics
    void print_statistics() {
        sc_time total_time = sc_time_stamp();
        double latency_ns = to_ns(total_time);

        RESULT_LOG("\n=== Simulation Results ===");
        RESULT_LOG("Total simulation time: " << latency_ns << " ns");
        RESULT_LOG("Operations completed:");
        RESULT_LOG("  Loads: " << load_idx << "/" << total_loads);
        RESULT_LOG("  IBUF Reads: " << ibuf_read_idx << "/" << total_ibuf_reads);
        RESULT_LOG("  CIM Computes: " << cim_compute_idx << "/" << total_cim_computes);
        RESULT_LOG("  OBUF Writes: " << obuf_write_idx << "/" << total_obuf_writes);
        RESULT_LOG("  Stores: " << store_idx << "/" << total_stores);

        // Update energy tracker with timing information
        energy_tracker.set_total_time_ns(latency_ns);

        // Update energy tracker with pipeline operation counts
        energy_tracker.set_pipeline_operations(load_idx, ibuf_read_idx, cim_compute_idx,
                                                obuf_write_idx, store_idx);

        // Update energy tracker with buffer peak usage
        auto [ibuf_peak, obuf_peak] = calculate_buffer_depth();
        energy_tracker.set_buffer_peak_usage(ibuf_peak, obuf_peak);

        // Update energy tracker with actual memory line statistics from TensorMemorySystem
        // Aggregate stats from all IBUF instances
        uint64_t total_ibuf_read_lines = 0, total_ibuf_write_lines = 0;
        for (const auto& ibuf : ibufs) {
            auto stats = ibuf->get_access_stats();
            total_ibuf_read_lines += stats.total_read_lines;
            total_ibuf_write_lines += stats.total_write_lines;
        }

        // Aggregate stats from all OBUF instances
        uint64_t total_obuf_read_lines = 0, total_obuf_write_lines = 0;
        for (const auto& obuf : obufs) {
            auto stats = obuf->get_access_stats();
            total_obuf_read_lines += stats.total_read_lines;
            total_obuf_write_lines += stats.total_write_lines;
        }

        // Get external memory stats
        auto ext_stats = external_memory->get_access_stats();

        // Set aggregated line stats to energy tracker
        energy_tracker.set_memory_line_stats(EnergyTracker::MemLevel::IBUF,
                                             EnergyTracker::AccessType::READ,
                                             total_ibuf_read_lines);
        energy_tracker.set_memory_line_stats(EnergyTracker::MemLevel::IBUF,
                                             EnergyTracker::AccessType::WRITE,
                                             total_ibuf_write_lines);

        energy_tracker.set_memory_line_stats(EnergyTracker::MemLevel::OBUF,
                                             EnergyTracker::AccessType::READ,
                                             total_obuf_read_lines);
        energy_tracker.set_memory_line_stats(EnergyTracker::MemLevel::OBUF,
                                             EnergyTracker::AccessType::WRITE,
                                             total_obuf_write_lines);

        energy_tracker.set_memory_line_stats(EnergyTracker::MemLevel::EXTERNAL,
                                             EnergyTracker::AccessType::READ,
                                             ext_stats.total_read_lines);
        energy_tracker.set_memory_line_stats(EnergyTracker::MemLevel::EXTERNAL,
                                             EnergyTracker::AccessType::WRITE,
                                             ext_stats.total_write_lines);

        // Print buffer usage statistics in JSON format for Python parsing
        print_buffer_usage_json();

        // Compute stage statistics from timeline for bottleneck analysis
        compute_stage_statistics();

        // Print energy statistics in JSON format for Python parsing
        energy_tracker.print_statistics();

        // Write simulation statistics to JSON file for Python analysis
        // File: <output_dir>/simulation_statistics.json (~1KB)
        // Why: Single source of truth - Python reads this file directly for all metrics
        //      No need for results.json or stdout parsing
        std::string energy_stats_path = output_dir + "/simulation_statistics.json";
        energy_tracker.write_statistics_to_file(energy_stats_path);

        // Write memory metadata for visualization
        // File: <output_dir>/memory_metadata.json (~5-10KB)
        // Why: Lightweight alternative to 3MB+ simulation_log.txt for memory layout visualization
        write_memory_metadata();
    }

    /**
     * @brief Collect and write memory metadata for visualization
     */
    void write_memory_metadata() {
        // Collect tensor information from external memory
        // For visualization, we show layout for a single sample (batch=1)
        if (external_memory) {
            auto ext_tensors = external_memory->get_all_tensor_info();
            for (const auto& [tensor_id, info] : ext_tensors) {
                std::string key = std::string("external_") + (tensor_id == "input_tensor" ? "input" : "output");

                // Normalize to single sample for visualization
                std::vector<int> single_sample_shape = info.max_shape;
                int total_lines_per_sample = info.total_lines;
                if (!single_sample_shape.empty() && single_sample_shape[0] > 1) {
                    int batch_size = single_sample_shape[0];
                    single_sample_shape[0] = 1;  // Set batch to 1
                    total_lines_per_sample = info.total_lines / batch_size;
                }

                output::TensorMetadata meta{
                    tensor_id,
                    single_sample_shape,
                    total_lines_per_sample,
                    info.channel_groups_per_line,
                    hw_config.external_bank_config.bits_per_line,
                    info.element_bitwidth
                };
                memory_metadata.add_tensor(key, meta);
            }
        }

        // Collect tensor information from IBUF
        if (!ibufs.empty() && ibufs[0]) {
            auto ibuf_tensors = ibufs[0]->get_all_tensor_info();
            for (const auto& [tensor_id, info] : ibuf_tensors) {
                output::TensorMetadata meta{
                    tensor_id,
                    info.max_shape,
                    info.total_lines,
                    info.channel_groups_per_line,
                    hw_config.ibuf_bank_config.bits_per_line,
                    info.element_bitwidth
                };
                memory_metadata.add_tensor("ibuf", meta);
            }
        }

        // Collect tensor information from OBUF
        if (!obufs.empty() && obufs[0]) {
            auto obuf_tensors = obufs[0]->get_all_tensor_info();
            for (const auto& [tensor_id, info] : obuf_tensors) {
                output::TensorMetadata meta{
                    tensor_id,
                    info.max_shape,
                    info.total_lines,
                    info.channel_groups_per_line,
                    hw_config.obuf_bank_config.bits_per_line,
                    info.element_bitwidth
                };
                memory_metadata.add_tensor("obuf", meta);
            }
        }

        // Collect access patterns from timeline
        for (const auto& record : timeline) {
            switch (record.operation) {
                case OperationType::LOAD:
                    // External read, IBUF write
                    if (record.source_max_lines_per_bank > 0) {
                        memory_metadata.record_access("external", "read", record.source_max_lines_per_bank);
                    }
                    if (record.dest_max_lines_per_bank > 0) {
                        memory_metadata.record_access("ibuf", "write", record.dest_max_lines_per_bank);
                    }
                    break;
                case OperationType::IBUF_READ:
                    if (record.source_max_lines_per_bank > 0) {
                        memory_metadata.record_access("ibuf", "read", record.source_max_lines_per_bank);
                    }
                    break;
                case OperationType::OBUF_WRITE:
                    if (record.dest_max_lines_per_bank > 0) {
                        memory_metadata.record_access("obuf", "write", record.dest_max_lines_per_bank);
                    }
                    break;
                case OperationType::STORE:
                    // OBUF read, External write
                    if (record.source_max_lines_per_bank > 0) {
                        memory_metadata.record_access("obuf", "read", record.source_max_lines_per_bank);
                    }
                    if (record.dest_max_lines_per_bank > 0) {
                        memory_metadata.record_access("external", "write", record.dest_max_lines_per_bank);
                    }
                    break;
                default:
                    break;
            }
        }

        // Write to file
        std::string metadata_path = output_dir + "/memory_metadata.json";
        memory_metadata.write_to_file(metadata_path);
    }

  private:
    // Calculate buffer depth based on registered tensors
    std::pair<int, int> calculate_buffer_depth() const {
        int total_ibuf_lines = 0;
        int total_obuf_lines = 0;

        // Get total allocated lines from IBUF (assuming single IBUF)
        if (!ibufs.empty() && ibufs[0]) {
            total_ibuf_lines = ibufs[0]->get_total_allocated_lines();
        }

        // Get total allocated lines from OBUF (assuming single OBUF)
        if (!obufs.empty() && obufs[0]) {
            total_obuf_lines = obufs[0]->get_total_allocated_lines();
        }

        return {total_ibuf_lines, total_obuf_lines};
    }

    // Print buffer usage in JSON format
    void print_buffer_usage_json() const {
        std::cout << "\n=== Buffer Usage JSON ===" << std::endl;
        std::cout << "{" << std::endl;

        // Calculate buffer depth based on registered tensors
        auto [ibuf_lines, obuf_lines] = calculate_buffer_depth();

        std::cout << "  \"ibuf_peak_lines\": " << ibuf_lines << "," << std::endl;
        std::cout << "  \"obuf_peak_lines\": " << obuf_lines << std::endl;
        std::cout << "}" << std::endl;
        std::cout << "=== End Buffer Usage JSON ===" << std::endl;
    }
};

#endif  // PIPELINE_SIMULATOR_H
