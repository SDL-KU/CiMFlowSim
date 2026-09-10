/**
 * @file tensor_memory_system.h
 * @brief Tensor-based unified memory system for CNN accelerator simulation
 *
 * This file implements a comprehensive tensor memory management system that provides:
 * - Multi-dimensional tensor registration and access tracking
 * - Memory bank and line allocation for optimal hardware modeling
 * - Port-based access control (single-port, dual-port, multi-port)
 * - Timing-accurate memory access simulation
 * - Automatic line allocation based on tensor shapes and element sizes
 *
 * Key Features:
 * - Tensor Registry: Manages tensor shapes, bit widths, and memory layout
 * - Bank Distribution: Distributes memory lines across multiple banks
 * - Access Patterns: Tracks read/write patterns for performance analysis
 * - Channel Atomicity: Treats channel groups as atomic units for realistic memory access
 * - Traversal Modes: Supports row-major and column-major memory layouts
 */

#ifndef TENSOR_MEMORY_SYSTEM_H
#define TENSOR_MEMORY_SYSTEM_H

#include <systemc.h>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "memory_types.h"

// Use same logging control as pipeline_simulator.h
#ifndef ENABLE_VERBOSE_LOGS
#define ENABLE_VERBOSE_LOGS 0
#endif

/**
 * @enum TraversalMode
 * @brief Memory traversal patterns for tensor data layout
 */
enum class TraversalMode {
    ROW_MAJOR,    // Height first, then width (H→W order) - most common for images
    COLUMN_MAJOR  // Width first, then height (W→H order) - alternative layout
};

/**
 * @class TensorMemorySystem
 * @brief SystemC module implementing tensor-based unified memory system
 *
 * This SystemC module provides a comprehensive memory management system for CNN accelerators:
 * - Manages multi-dimensional tensors with automatic memory allocation
 * - Provides bank-interleaved memory access for parallel processing
 * - Supports multiple port configurations (single, dual, multi-port)
 * - Implements timing-accurate memory access simulation
 * - Tracks access patterns for performance analysis
 */
SC_MODULE(TensorMemorySystem) {
    // ============================================================================
    // Tensor Information Structure
    // ============================================================================

    /**
     * @struct TensorInfo
     * @brief Complete information about registered tensors
     */
    struct TensorInfo {
        std::string tensor_id;
        std::vector<int> max_shape;  // Maximum dimensions [batch, channel, height, width]
        int element_bitwidth;     // Bits per element (e.g., 8-bit int, 16-bit float, 32-bit float)
        TraversalMode traversal;  // Traversal mode for the tensor (height vs width order)
        int global_line_base;     // Starting global line ID
        int total_lines;          // Total allocated lines
        int channel_groups_per_line;  // Channel groups that fit in one line (all channels atomic)

        TensorInfo() = default;

        TensorInfo(const std::string& id, const std::vector<int>& shape, int bits,
                   TraversalMode mode, int base, int lines, int cgpl)
            : tensor_id(id),
              max_shape(shape),
              element_bitwidth(bits),
              traversal(mode),
              global_line_base(base),
              total_lines(lines),
              channel_groups_per_line(cgpl) {}

        int get_channels() const { return max_shape.size() > 1 ? max_shape[1] : 1; }
        int get_height() const { return max_shape.size() > 2 ? max_shape[2] : 1; }
        int get_width() const { return max_shape.size() > 3 ? max_shape[3] : 1; }
        int get_pixel_bits() const { return element_bitwidth * get_channels(); }

        std::string to_string() const {
            std::string shape_str = "[";
            for (size_t i = 0; i < max_shape.size(); ++i) {
                if (i > 0)
                    shape_str += ",";
                shape_str += std::to_string(max_shape[i]);
            }
            shape_str += "]";
            return tensor_id + shape_str + " @ lines " + std::to_string(global_line_base) + "-" +
                   std::to_string(global_line_base + total_lines - 1);
        }

        int calculate_total_elements() const {
            int total = 1;
            for (int dim : max_shape) {
                total *= dim;
            }
            return total;
        }
    };

    // ============================================================================
    // Tensor Access Range Structure
    // ============================================================================
    struct TensorRange {
        std::string tensor_id;
        std::vector<std::pair<int, int>> ranges;  // [start, end) for each dimension
        sc_time access_time;
        bool is_write;

        TensorRange(const std::string& id, const std::vector<std::pair<int, int>>& r, sc_time time,
                    bool write)
            : tensor_id(id), ranges(r), access_time(time), is_write(write) {}

        std::string to_string() const {
            std::string range_str = "[";
            for (size_t i = 0; i < ranges.size(); ++i) {
                if (i > 0)
                    range_str += ",";
                range_str +=
                    std::to_string(ranges[i].first) + ":" + std::to_string(ranges[i].second);
            }
            range_str += "]";
            return tensor_id + range_str + (is_write ? " (W)" : " (R)");
        }
    };

    // ============================================================================
    // Request Structure for Unified System
    // ============================================================================
    struct UnifiedMemoryRequest {
        enum Type { READ, WRITE };

        int id;
        Type type;
        std::string tensor_id;
        std::vector<std::pair<int, int>> ranges;
        sc_event* completion_event;

        UnifiedMemoryRequest() : id(0), type(READ), completion_event(nullptr) {}

        UnifiedMemoryRequest(int req_id, Type req_type, const std::string& tid,
                             const std::vector<std::pair<int, int>>& rng, sc_event* event)
            : id(req_id), type(req_type), tensor_id(tid), ranges(rng), completion_event(event) {}

        friend std::ostream& operator<<(std::ostream& os, const UnifiedMemoryRequest& req) {
            return os << "UnifiedRequest(id=" << req.id
                      << ", type=" << (req.type == READ ? "READ" : "WRITE")
                      << ", tensor=" << req.tensor_id << ")";
        }
    };

    // ============================================================================
    // Memory Statistics Structure
    // ============================================================================
    struct MemoryStats {
        int total_registered_tensors;
        int total_allocated_lines;
        int total_tensor_accesses;
        int total_memory_bytes;
    };

    // Configuration
    PortConfiguration port_config;
    MemoryTimingConfig timing_config;
    BankLineConfig bank_line_config;

    // Request queues based on port configuration
    sc_fifo<UnifiedMemoryRequest> request_queue;        // For SINGLE_PORT, TRUE_DUAL_PORT
    sc_fifo<UnifiedMemoryRequest> read_request_queue;   // For SIMPLE_DUAL_PORT
    sc_fifo<UnifiedMemoryRequest> write_request_queue;  // For SIMPLE_DUAL_PORT

    // Statistics
    mutable MemoryAccessPattern access_stats;
    std::atomic<int> request_counter;

    // Bank usage tracking for conflict modeling
    mutable std::map<int, sc_time> bank_last_access;

  private:
    // Tensor registry
    std::map<std::string, TensorInfo> tensor_registry_;

    // Tensor access history
    std::map<std::string, std::vector<TensorRange>> tensor_accesses_;

    // Global line allocation tracking
    int next_available_line_;

    // Memory configuration
    int num_banks_;
    int line_bits_;

  public:
    SC_HAS_PROCESS(TensorMemorySystem);

    // Constructor
    TensorMemorySystem(sc_module_name name, const PortConfiguration& ports,
                       const MemoryTimingConfig& timing, const BankLineConfig& bl_config)
        : sc_module(name),
          port_config(ports),
          timing_config(timing),
          bank_line_config(bl_config),
          request_queue("request_queue", 256),
          read_request_queue("read_request_queue", 256),
          write_request_queue("write_request_queue", 256),
          request_counter(0),
          next_available_line_(0),
          num_banks_(bl_config.num_banks),
          line_bits_(bl_config.bits_per_line) {
        // Start worker threads based on port configuration
        if (port_config.num_rw_ports > 0) {
            SC_THREAD(process_rw_requests);
        }
        if (port_config.num_read_ports > 0) {
            SC_THREAD(process_read_requests);
        }
        if (port_config.num_write_ports > 0) {
            SC_THREAD(process_write_requests);
        }
    }

    // ============================================================================
    // Tensor Registration Interface
    // ============================================================================
    void register_tensor(const std::string& tensor_id, const std::vector<int>& max_shape,
                         int element_bitwidth = 64,
                         TraversalMode traversal = TraversalMode::ROW_MAJOR) {
        // Check if tensor already exists
        if (tensor_registry_.find(tensor_id) != tensor_registry_.end()) {
            std::string error_msg =
                "Tensor '" + tensor_id +
                "' is already registered. Duplicate tensor registration not allowed.";
            std::cerr << "Error: " << error_msg << std::endl;
            throw std::runtime_error(error_msg);
        }

        // Calculate memory requirements
        int spatial_pixels = calculate_spatial_pixels(max_shape);

        // Compute number of channels (based on tensor dimensions)
        int channels;
        if (max_shape.size() == 1) {
            channels = 1;  // 1D: [N] -> 1 channel
        } else if (max_shape.size() == 2) {
            channels = 1;  // 2D: [H, W] -> 1 channel
        } else if (max_shape.size() == 3) {
            channels = max_shape[0];  // 3D: [C, H, W] -> C channels
        } else if (max_shape.size() == 4) {
            channels = max_shape[1];  // 4D: [B, C, H, W] -> C channels
        } else {
            channels = 1;  // default
        }
        int channel_group_bits =
            element_bitwidth * channels;  // Channel group bits (all channels atomic)
        int channel_groups_per_line = line_bits_ / channel_group_bits;

        // Store original value for DEBUG output (negative for multi-line pixels)
        int debug_groups_per_line = channel_groups_per_line;

        if (channel_groups_per_line < 1) {
            // If channel group is larger than line, each group needs multiple lines
            int lines_per_group = (channel_group_bits + line_bits_ - 1) / line_bits_;
            int total_lines_needed = spatial_pixels * lines_per_group;
            channel_groups_per_line = 1;  // Special case: 1 group per multiple lines
            debug_groups_per_line = -lines_per_group;  // Negative to indicate lines per pixel

            // Store tensor info - for multi-line groups, store lines_per_group as negative value
            tensor_registry_[tensor_id] =
                TensorInfo(tensor_id, max_shape, element_bitwidth, traversal, next_available_line_,
                           total_lines_needed, -lines_per_group);
        } else {
            // Normal case: multiple groups per line
            int total_lines_needed =
                (spatial_pixels + channel_groups_per_line - 1) / channel_groups_per_line;

            // Store tensor info
            tensor_registry_[tensor_id] =
                TensorInfo(tensor_id, max_shape, element_bitwidth, traversal, next_available_line_,
                           total_lines_needed, channel_groups_per_line);
        }

        // Update next available line
        next_available_line_ += tensor_registry_[tensor_id].total_lines;

#if ENABLE_VERBOSE_LOGS
        // DEBUG output AFTER processing multi-line case
        std::cout << "[DEBUG] Tensor: " << tensor_id << ", spatial_pixels=" << spatial_pixels
                  << ", channels=" << channels << ", channel_group_bits=" << channel_group_bits
                  << ", line_bits=" << line_bits_ << ", groups_per_line=" << debug_groups_per_line
                  << std::endl;

        std::cout << "[TensorMemorySystem] Registered tensor '" << tensor_id << "' shape=";
        std::cout << "[";
        for (size_t i = 0; i < max_shape.size(); ++i) {
            if (i > 0)
                std::cout << ",";
            std::cout << max_shape[i];
        }
        std::cout << "], " << element_bitwidth << "-bit elements, " << channels
                  << " channels (atomic), ";
        std::cout << (traversal == TraversalMode::ROW_MAJOR ? "ROW_MAJOR" : "COLUMN_MAJOR");
        std::cout << " -> lines " << tensor_registry_[tensor_id].global_line_base << "-";
        std::cout << (next_available_line_ - 1) << " (" << tensor_registry_[tensor_id].total_lines
                  << " total)" << std::endl;
#endif
        // Suppress unused variable warning when logs disabled
        (void)debug_groups_per_line;
    }

    // ============================================================================
    // Tensor Access Interface
    // ============================================================================
    void read(const std::string& tensor_id, const std::vector<std::pair<int, int>>& ranges,
              sc_event& completion_event) {
        submit_tensor_request(tensor_id, ranges, completion_event, UnifiedMemoryRequest::READ);
    }

    void write(const std::string& tensor_id, const std::vector<std::pair<int, int>>& ranges,
               sc_event& completion_event) {
        submit_tensor_request(tensor_id, ranges, completion_event, UnifiedMemoryRequest::WRITE);
    }

    // ============================================================================
    // Tensor Information Interface
    // ============================================================================
    bool is_tensor_registered(const std::string& tensor_id) const {
        return tensor_registry_.find(tensor_id) != tensor_registry_.end();
    }

    const TensorInfo& get_tensor_info(const std::string& tensor_id) const {
        validate_tensor_exists(tensor_id);
        return tensor_registry_.at(tensor_id);
    }

    std::vector<std::string> get_registered_tensors() const {
        std::vector<std::string> tensor_names;
        for (const auto& pair : tensor_registry_) {
            tensor_names.push_back(pair.first);
        }
        return tensor_names;
    }

    // Get information about all registered tensors
    std::map<std::string, TensorInfo> get_all_tensor_info() const {
        return tensor_registry_;
    }

    // Get total allocated lines for all registered tensors
    int get_total_allocated_lines() const {
        int total_lines = 0;
        for (const auto& [tensor_name, tensor_info] : tensor_registry_) {
            total_lines += tensor_info.total_lines;
        }
        return total_lines;
    }

    void print_tensor_registry() const {
#if ENABLE_VERBOSE_LOGS
        std::cout << "Registered tensors:" << std::endl;
        for (const auto& pair : tensor_registry_) {
            std::cout << "  " << pair.second.to_string() << std::endl;
        }
#endif
    }

    // ============================================================================
    // Advanced Features
    // ============================================================================

    // Tensor access history management
    const std::vector<TensorRange>& get_tensor_accesses(const std::string& tensor_id) const {
        static const std::vector<TensorRange> empty_accesses;
        auto it = tensor_accesses_.find(tensor_id);
        if (it != tensor_accesses_.end()) {
            return it->second;
        }
        return empty_accesses;
    }

    void clear_tensor_accesses(const std::string& tensor_id) {
        tensor_accesses_[tensor_id].clear();
    }

    void clear_all_tensor_accesses() {
        tensor_accesses_.clear();
    }

    // Bank and line calculations
    int get_bank_id(int global_line_id) const {
        return global_line_id % num_banks_;
    }

    int get_local_line_id(int global_line_id) const {
        return global_line_id / num_banks_;
    }

    int get_global_line_id_for_tensor(const std::string& tensor_id,
                                      const std::vector<int>& coordinates) const {
        validate_tensor_exists(tensor_id);
        const auto& tensor_info = tensor_registry_.at(tensor_id);

        // Compute the pixel index within the tensor
        int pixel_index = calculate_tensor_pixel_index(tensor_info, coordinates);

        // Compute channel group index
        int channel_group_line_offset;
        if (tensor_info.channel_groups_per_line > 0) {
            // Normal case: multiple groups per line
            channel_group_line_offset = pixel_index / tensor_info.channel_groups_per_line;
        } else {
            // Special case: each group spans multiple lines
            int lines_per_group = -tensor_info.channel_groups_per_line;
            channel_group_line_offset = pixel_index * lines_per_group;
        }

        // Global line ID = tensor base + offset
        return tensor_info.global_line_base + channel_group_line_offset;
    }

    // Timing calculations
    // Helper to calculate accessed lines for a tensor range
    std::set<int> get_accessed_lines_for_tensor_range(
        const std::string& tensor_id, const std::vector<std::pair<int, int>>& ranges) const {
        validate_tensor_exists(tensor_id);
        const auto& tensor_info = tensor_registry_.at(tensor_id);

        // Collect indices of all lines being accessed
        std::set<int> accessed_lines;

        // Multi-line case: one pixel spans multiple lines
        int lines_per_pixel = 1;
        if (tensor_info.channel_groups_per_line < 0) {
            lines_per_pixel = -tensor_info.channel_groups_per_line;
        }

        // Helper lambda to insert all lines for a pixel
        auto insert_pixel_lines = [&](int start_line_id) {
            for (int offset = 0; offset < lines_per_pixel; offset++) {
                accessed_lines.insert(start_line_id + offset);
            }
        };

        // Iterate over all dimension combinations (including batch)
        if (tensor_info.max_shape.size() == 4) {
            // 4D: [batch, channel, height, width]
            for (int b = ranges[0].first; b < ranges[0].second; b++) {
                for (int h = ranges[2].first; h < ranges[2].second; h++) {
                    for (int w = ranges[3].first; w < ranges[3].second; w++) {
                        // Compute line index for the channel group at each (b,h,w) position
                        std::vector<int> coords = {b, 0, h, w};  // channel=0 (channels are atomic)
                        int line_id = get_global_line_id_for_tensor(tensor_id, coords);
                        insert_pixel_lines(line_id);
                    }
                }
            }
        } else if (tensor_info.max_shape.size() == 3) {
            // 3D: [channel, height, width]
            for (int h = ranges[1].first; h < ranges[1].second; h++) {
                for (int w = ranges[2].first; w < ranges[2].second; w++) {
                    std::vector<int> coords = {0, h, w};  // channel=0 (channels are atomic)
                    int line_id = get_global_line_id_for_tensor(tensor_id, coords);
                    insert_pixel_lines(line_id);
                }
            }
        } else if (tensor_info.max_shape.size() == 2) {
            // 2D: [height, width]
            for (int h = ranges[0].first; h < ranges[0].second; h++) {
                for (int w = ranges[1].first; w < ranges[1].second; w++) {
                    std::vector<int> coords = {h, w};
                    int line_id = get_global_line_id_for_tensor(tensor_id, coords);
                    insert_pixel_lines(line_id);
                }
            }
        }

        return accessed_lines;
    }

    // Structure to hold both timing and energy line counts
    struct LineAccessInfo {
        int total_lines;          // For energy calculation (all accesses)
        int max_lines_per_bank;   // For timing calculation (parallel access bottleneck)
    };

    // Get line access information in one pass (efficient version)
    LineAccessInfo get_line_access_info_for_tensor_range(
        const std::string& tensor_id, const std::vector<std::pair<int, int>>& ranges) const {

        std::set<int> accessed_lines = get_accessed_lines_for_tensor_range(tensor_id, ranges);

        // Calculate both total and max per bank in a single pass
        std::map<int, int> lines_per_bank;
        for (int line_id : accessed_lines) {
            int bank_id = get_bank_id(line_id);
            lines_per_bank[bank_id]++;
        }

        int max_lines_per_bank = 0;
        for (const auto& [bank_id, line_count] : lines_per_bank) {
            max_lines_per_bank = std::max(max_lines_per_bank, line_count);
        }

        return LineAccessInfo{
            static_cast<int>(accessed_lines.size()),  // total_lines
            max_lines_per_bank                         // max_lines_per_bank
        };
    }


    // ============================================================================
    // Statistics and Configuration
    // ============================================================================
    void reset() {
        access_stats = MemoryAccessPattern();
        bank_last_access.clear();
        clear_all_tensor_accesses();
    }

    MemoryAccessPattern get_access_stats() const {
        return access_stats;
    }

    MemoryStats get_memory_stats() const {
        MemoryStats stats;
        stats.total_registered_tensors = tensor_registry_.size();
        stats.total_allocated_lines = next_available_line_;

        stats.total_tensor_accesses = 0;
        for (const auto& [name, accesses] : tensor_accesses_) {
            stats.total_tensor_accesses += accesses.size();
        }

        // Estimate memory usage (rough calculation)
        stats.total_memory_bytes = stats.total_allocated_lines * (line_bits_ / 8);

        return stats;
    }

  private:
    // ============================================================================
    // Helper Functions
    // ============================================================================

    void validate_tensor_exists(const std::string& tensor_id) const {
        if (tensor_registry_.find(tensor_id) == tensor_registry_.end()) {
            throw std::runtime_error("Tensor '" + tensor_id + "' not registered");
        }
    }

    void validate_tensor_range_bounds(const std::string& tensor_id,
                                      const std::vector<std::pair<int, int>>& ranges) const {
        const auto& tensor_info = tensor_registry_.at(tensor_id);

        if (ranges.size() != tensor_info.max_shape.size()) {
            throw std::invalid_argument("Range dimensions don't match tensor shape");
        }

        for (size_t i = 0; i < ranges.size(); ++i) {
            if (ranges[i].first < 0 || ranges[i].second > tensor_info.max_shape[i] ||
                ranges[i].first >= ranges[i].second) {
                throw std::out_of_range("Range [" + std::to_string(ranges[i].first) + ":" +
                                        std::to_string(ranges[i].second) +
                                        "] is invalid for dimension " + std::to_string(i) +
                                        " with size " + std::to_string(tensor_info.max_shape[i]));
            }
        }
    }

    int calculate_spatial_pixels(const std::vector<int>& shape) const {
        // Calculate spatial pixels (including batch dimension for proper memory allocation)
        int spatial_pixels = 1;
        if (shape.size() == 1) {
            spatial_pixels = shape[0];  // 1D: [N] - take N
        } else if (shape.size() == 2) {
            spatial_pixels = shape[0] * shape[1];  // 2D: [H, W] - take H×W
        } else if (shape.size() == 3) {
            spatial_pixels = shape[1] * shape[2];  // 3D: [C, H, W] - take H×W
        } else if (shape.size() == 4) {
            spatial_pixels = shape[0] * shape[2] * shape[3];  // 4D: [B, C, H, W] - take B×H×W
        }
        return spatial_pixels;
    }

    int calculate_tensor_pixel_index(const TensorInfo& tensor_info,
                                     const std::vector<int>& coordinates) const {
        const auto& shape = tensor_info.max_shape;
        int pixel_index = 0;

        if (shape.size() == 2) {
            // 2D: [h, w]
            int h = coordinates[0];
            int w = coordinates[1];
            pixel_index = h * shape[1] + w;
        } else if (shape.size() == 3) {
            // 3D: [c, h, w] - spatial indexing only (channels are atomic)
            int h = coordinates[1];
            int w = coordinates[2];
            pixel_index = h * shape[2] + w;
        } else if (shape.size() == 4) {
            // 4D: [b, c, h, w] - include batch dimension for proper memory layout
            int b = coordinates[0];
            int h = coordinates[2];
            int w = coordinates[3];

            // ROW_MAJOR traversal: batch first, then spatial
            int spatial_size = shape[2] * shape[3];  // H * W
            pixel_index = b * spatial_size + h * shape[3] + w;
        } else {
            throw std::invalid_argument("Invalid coordinate dimensions");
        }

        return pixel_index;
    }

    // ============================================================================
    // Request Submission
    // ============================================================================
    void submit_tensor_request(const std::string& tensor_id,
                               const std::vector<std::pair<int, int>>& ranges,
                               sc_event& completion_event, UnifiedMemoryRequest::Type type) {
        UnifiedMemoryRequest req(++request_counter, type, tensor_id, ranges, &completion_event);
        submit_generic_request(req);
    }

    void submit_generic_request(const UnifiedMemoryRequest& req) {
        if (port_config.num_rw_ports > 0) {
            // Use general request queue for read/write ports
            request_queue.write(req);
        } else if (req.type == UnifiedMemoryRequest::READ && port_config.num_read_ports > 0) {
            // Use dedicated read queue
            read_request_queue.write(req);
        } else if (req.type == UnifiedMemoryRequest::WRITE && port_config.num_write_ports > 0) {
            // Use dedicated write queue
            write_request_queue.write(req);
        } else {
            // Fallback to general queue
            request_queue.write(req);
        }
    }

    // ============================================================================
    // Worker Threads for Port Processing
    // ============================================================================
    void process_rw_requests() {
        while (true) {
            UnifiedMemoryRequest req = request_queue.read();
            process_tensor_request(req);
        }
    }

    void process_read_requests() {
        while (true) {
            UnifiedMemoryRequest req = read_request_queue.read();
            if (req.type == UnifiedMemoryRequest::READ) {
                process_tensor_request(req);
            }
        }
    }

    void process_write_requests() {
        while (true) {
            UnifiedMemoryRequest req = write_request_queue.read();
            if (req.type == UnifiedMemoryRequest::WRITE) {
                process_tensor_request(req);
            }
        }
    }

    void process_tensor_request(const UnifiedMemoryRequest& req) {
        validate_tensor_exists(req.tensor_id);
        validate_tensor_range_bounds(req.tensor_id, req.ranges);

        // Calculate line access info in a single pass (efficient)
        auto line_info = get_line_access_info_for_tensor_range(req.tensor_id, req.ranges);

        // Calculate access time based on maximum lines per bank (timing: parallel access)
        double line_time_ns = timing_config.ns_per_clock_cycle();  // Time per line access
        double total_time_ns =
            timing_config.base_latency_ns() + (line_time_ns * line_info.max_lines_per_bank);
        sc_time access_time = sc_time(total_time_ns, SC_NS);

        // Use total_lines for energy calculation
        int total_lines = line_info.total_lines;

        // Record the access with calculated time
        tensor_accesses_[req.tensor_id].emplace_back(req.tensor_id, req.ranges, access_time,
                                                     req.type == UnifiedMemoryRequest::WRITE);

        // Update access statistics
        if (req.type == UnifiedMemoryRequest::READ) {
            access_stats.total_reads++;
            access_stats.total_read_lines += total_lines;
        } else {
            access_stats.total_writes++;
            access_stats.total_write_lines += total_lines;
        }

#if ENABLE_VERBOSE_LOGS
        std::cout << "[TensorMemorySystem:" << name() << "] "
                  << (req.type == UnifiedMemoryRequest::READ ? "Read from" : "Write to")
                  << " tensor '" << req.tensor_id << "' ranges: ";
        for (size_t i = 0; i < req.ranges.size(); ++i) {
            if (i > 0)
                std::cout << ",";
            std::cout << "[" << req.ranges[i].first << ":" << req.ranges[i].second << "]";
        }
        std::cout << " (access_time: " << access_time
                  << ", max_lines_per_bank: " << line_info.max_lines_per_bank
                  << ", base_latency: " << timing_config.base_latency_ns()
                  << "ns, line_time: " << line_time_ns << "ns)" << std::endl;
#endif

        // Wait for the calculated access time, then notify immediately
        wait(access_time);
        req.completion_event->notify(SC_ZERO_TIME);
    }
};

#endif  // STANDALONE_UNIFIED_MEMORY_H