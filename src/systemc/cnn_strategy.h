/**
 * @file cnn_strategy.h
 * @brief CNN accelerator memory access strategy interface and implementations
 *
 * This file defines the Strategy Pattern for CNN convolution execution strategies
 * using Independent Tiling with flexible, JSON-based configuration.
 *
 * Key Concepts:
 * - Memory Hierarchy: External Memory ↔ IBUF ↔ CIM Array ↔ OBUF ↔ External Memory
 * - Double Buffering: Pipeline-safe buffer management with alternating indices
 * - Dependency Management: Operation scheduling based on data dependencies
 * - Independent Tiling: Flexible tile sizes configured via JSON
 */

#ifndef CNN_STRATEGY_H
#define CNN_STRATEGY_H

#include <stdexcept>
#include <string>
#include <vector>

#include "memory/memory_types.h"

// Forward declarations (definitions in pipeline_simulator.h)
struct CNNParams;

/**
 * @brief Helper structure for compute operation index decomposition
 * @details Converts linear compute index to multi-dimensional coordinates
 */
struct ComputeIndices {
    int batch_idx;  // Batch index in range [0, batch_size)
    int p, q;       // Output spatial position [0, P) x [0, Q)
    int local_idx;  // Position within current batch

    /**
     * @brief Convert linear compute index to multi-dimensional coordinates
     * @param idx Linear compute operation index
     * @param params CNN layer parameters
     * @return ComputeIndices structure with decomposed coordinates
     */
    static ComputeIndices from_compute_idx(int idx, const CNNParams& params);
};

/**
 * @brief Operation types in the CNN accelerator pipeline
 * @details Defines the five-stage pipeline: LOAD → IBUF_READ → CIM_COMPUTE → OBUF_WRITE → STORE
 */
enum class OperationType {
    LOAD,         // External Memory → IBUF transfer
    IBUF_READ,    // IBUF → CIM Array data read
    CIM_COMPUTE,  // Convolution computation in CIM array
    OBUF_WRITE,   // CIM Array → OBUF result write
    STORE         // OBUF → External Memory transfer
};

/**
 * @brief Dependency structure for operation scheduling
 * @details Specifies which operation must complete before current operation can start
 */
struct Dependency {
    OperationType operation_type;  // Type of prerequisite operation
    int operation_id;              // Index of the prerequisite operation

    /**
     * @brief Construct a dependency relationship
     * @param op_type Type of the prerequisite operation
     * @param op_id Index of the prerequisite operation
     */
    Dependency(OperationType op_type, int op_id) : operation_type(op_type), operation_id(op_id) {}
};

/**
 * @class CNNStrategy
 * @brief Abstract base class for CNN accelerator memory access strategies
 *
 * This class defines the interface for different memory access strategies.
 * Each concrete strategy implements:
 * - Memory access patterns (which data to load/store when)
 * - Buffer management (IBUF/OBUF allocation and indexing)
 * - Operation dependencies (ensuring pipeline correctness)
 * - Tensor region calculations (memory access bounds)
 */
class CNNStrategy {
  public:
    virtual ~CNNStrategy() = default;

    // ========================================================================
    // Memory Access Pattern Functions
    // ========================================================================

    /**
     * @brief Get external memory read access pattern for load operation
     * @param load_idx Index of the load operation
     * @param params CNN layer parameters
     * @return Pair of tensor name and access ranges [(start, end), ...]
     */
    virtual std::pair<std::string, std::vector<std::pair<int, int>>>
    get_external_read_access(int load_idx, const CNNParams& params) const = 0;

    /**
     * @brief Get external memory write access pattern for store operation
     * @param store_idx Index of the store operation
     * @param params CNN layer parameters
     * @return Pair of tensor name and access ranges
     */
    virtual std::pair<std::string, std::vector<std::pair<int, int>>>
    get_external_write_access(int store_idx, const CNNParams& params) const = 0;

    /**
     * @brief Get IBUF write access pattern (External → IBUF)
     * @param load_idx Index of the load operation
     * @param params CNN layer parameters
     * @return Pair of tensor name and buffer access ranges
     */
    virtual std::pair<std::string, std::vector<std::pair<int, int>>>
    get_ibuf_write_access(int load_idx, const CNNParams& params) const = 0;

    /**
     * @brief Get IBUF read access pattern (IBUF → CIM)
     * @param compute_idx Index of the compute operation
     * @param params CNN layer parameters
     * @return Pair of tensor name and buffer access ranges
     */
    virtual std::pair<std::string, std::vector<std::pair<int, int>>>
    get_ibuf_read_access(int compute_idx, const CNNParams& params) const = 0;

    /**
     * @brief Get OBUF write access pattern (CIM → OBUF)
     * @param obuf_write_idx Index of the OBUF write operation
     * @param params CNN layer parameters
     * @return Pair of tensor name and buffer access ranges
     */
    virtual std::pair<std::string, std::vector<std::pair<int, int>>>
    get_obuf_write_access(int obuf_write_idx, const CNNParams& params) const = 0;

    /**
     * @brief Get OBUF read access pattern (OBUF → External)
     * @param store_idx Index of the store operation
     * @param params CNN layer parameters
     * @return Pair of tensor name and buffer access ranges
     */
    virtual std::pair<std::string, std::vector<std::pair<int, int>>>
    get_obuf_read_access(int store_idx, const CNNParams& params) const = 0;

    // ========================================================================
    // Buffer Shape Functions
    // ========================================================================

    /**
     * @brief Get IBUF buffer dimensions for tensor memory system
     * @param params CNN layer parameters
     * @return Vector of dimensions [batch, channels, height, width]
     * @details Used by tensor memory system to allocate proper buffer space
     */
    virtual std::vector<int> get_ibuf_shape(const CNNParams& params) const = 0;

    /**
     * @brief Get OBUF buffer dimensions for tensor memory system
     * @param params CNN layer parameters
     * @return Vector of dimensions [batch, channels, height, width]
     * @details Used by tensor memory system to allocate proper buffer space
     */
    virtual std::vector<int> get_obuf_shape(const CNNParams& params) const = 0;

    // ========================================================================
    // Operation Count Information Functions
    // ========================================================================

    /**
     * @brief Get total number of load operations (External Memory → IBUF)
     * @param params CNN layer parameters
     * @return Total load operations for the entire layer computation
     */
    virtual int get_total_loads(const CNNParams& params) const = 0;

    /**
     * @brief Get total number of compute operations (convolution in CIM array)
     * @param params CNN layer parameters
     * @return Total compute operations for the entire layer computation
     */
    virtual int get_total_computes(const CNNParams& params) const = 0;

    /**
     * @brief Get total number of store operations (OBUF → External Memory)
     * @param params CNN layer parameters
     * @return Total store operations for the entire layer computation
     */
    virtual int get_total_stores(const CNNParams& params) const = 0;

    /**
     * @brief Get total number of OBUF write operations (CIM → OBUF)
     * @param params CNN layer parameters
     * @return Total OBUF writes, typically organized by pooling groups
     * @details Often fewer than computes due to pooling aggregation
     */
    virtual int get_total_obuf_writes(const CNNParams& params) const = 0;

    // ========================================================================
    // Buffer Assignment Functions
    // ========================================================================

    /**
     * @brief Get IBUF index for load operation (double buffering)
     * @param load_idx Index of the load operation
     * @param params CNN layer parameters
     * @return Buffer index (0 or 1 for double buffering)
     */
    virtual int get_load_ibuf_idx(int load_idx, const CNNParams& params) const = 0;

    /**
     * @brief Get IBUF index for compute operation read
     * @param ibuf_read_idx Index of the IBUF read operation
     * @param params CNN layer parameters
     * @return Buffer index (0 or 1 for double buffering)
     */
    virtual int get_compute_ibuf_idx(int ibuf_read_idx, const CNNParams& params) const = 0;

    /**
     * @brief Get OBUF index for compute operation write
     * @param obuf_write_idx Index of the OBUF write operation
     * @param params CNN layer parameters
     * @return Buffer index (0 or 1 for double buffering)
     */
    virtual int get_compute_obuf_idx(int obuf_write_idx, const CNNParams& params) const = 0;

    /**
     * @brief Get OBUF index for store operation read
     * @param store_idx Index of the store operation
     * @param params CNN layer parameters
     * @return Buffer index (0 or 1 for double buffering)
     */
    virtual int get_store_obuf_idx(int store_idx, const CNNParams& params) const = 0;

    // ========================================================================
    // Buffer Configuration Functions
    // ========================================================================

    /**
     * @brief Get number of IBUF buffers needed (typically 2 for double buffering)
     * @return Number of IBUF instances required
     */
    virtual int get_ibuf_count() const = 0;

    /**
     * @brief Get number of OBUF buffers needed (typically 2 for double buffering)
     * @return Number of OBUF instances required
     */
    virtual int get_obuf_count() const = 0;

    /**
     * @brief Get IBUF size per buffer (in elements)
     * @param params CNN layer parameters
     * @return Size of each IBUF buffer in elements
     * @details Strategy-specific: varies from R×S×C to H×W×C
     */
    virtual int get_ibuf_size(const CNNParams& params) const = 0;

    /**
     * @brief Get OBUF size per buffer (in elements)
     * @param params CNN layer parameters
     * @return Size of each OBUF buffer in elements
     * @details Strategy-specific: varies from single pixel to full output
     */
    virtual int get_obuf_size(const CNNParams& params) const = 0;

    // ========================================================================
    // Dependency Calculation Functions for Pipeline Scheduling
    // ========================================================================

    /**
     * @brief Get dependencies for load operation
     * @param load_idx Index of the load operation
     * @param params CNN layer parameters
     * @return Vector of dependencies that must complete before this load
     * @details Ensures pipeline safety and prevents buffer conflicts
     */
    virtual std::vector<Dependency> get_load_dependencies(int load_idx,
                                                          const CNNParams& params) const = 0;

    /**
     * @brief Get dependencies for IBUF read operation
     * @param ibuf_read_idx Index of the IBUF read operation
     * @param params CNN layer parameters
     * @return Vector of dependencies that must complete before this read
     * @details Ensures data is available in IBUF before compute reads it
     */
    virtual std::vector<Dependency> get_ibuf_read_dependencies(int ibuf_read_idx,
                                                               const CNNParams& params) const = 0;

    /**
     * @brief Get dependencies for compute operation
     * @param compute_idx Index of the compute operation
     * @param params CNN layer parameters
     * @return Vector of dependencies that must complete before this compute
     * @details Ensures input data is ready and output buffer is available
     */
    virtual std::vector<Dependency> get_compute_dependencies(int compute_idx,
                                                             const CNNParams& params) const = 0;

    /**
     * @brief Get dependencies for OBUF write operation
     * @param obuf_write_idx Index of the OBUF write operation
     * @param params CNN layer parameters
     * @return Vector of dependencies that must complete before this write
     * @details Ensures compute results are ready and buffer space is available
     */
    virtual std::vector<Dependency> get_obuf_write_dependencies(int obuf_write_idx,
                                                                const CNNParams& params) const = 0;

    /**
     * @brief Get dependencies for store operation
     * @param store_idx Index of the store operation
     * @param params CNN layer parameters
     * @return Vector of dependencies that must complete before this store
     * @details Ensures OBUF data is ready before transferring to external memory
     */
    virtual std::vector<Dependency> get_store_dependencies(int store_idx,
                                                           const CNNParams& params) const = 0;
};

// ========================================================================
// Independent Tiling Strategy
// ========================================================================

/**
 * @brief Tiling configuration for Independent Tiling strategy
 * @details Defines tile sizes and counts for flexible CNN execution
 */
struct TilingConfig {
    // ========== Tile Dimensions ==========
    // Output tiles (P×Q coordinate system)
    int output_tile_p;  // Output tile height
    int output_tile_q;  // Output tile width

    // Input tiles (H×W coordinate system)
    int input_tile_h;  // Input tile height (receptive field)
    int input_tile_w;  // Input tile width (receptive field)

    // Input tiles mapped to output coordinate system (P×Q)
    int input_tile_p;  // Input tile in output P dimension: (input_tile_h - R) / stride + 1
    int input_tile_q;  // Input tile in output Q dimension: (input_tile_w - S) / stride + 1

    // ========== Directional Tile Counts ==========
    int num_output_tiles_p;  // Number of output tiles in P direction
    int num_output_tiles_q;  // Number of output tiles in Q direction
    int num_input_tiles_p;   // Number of input tiles in P direction: P / input_tile_p
    int num_input_tiles_q;   // Number of input tiles in Q direction: Q / input_tile_q

    // ========== Total Tile Counts ==========
    int output_tile_count;  // Total: num_output_tiles_p × num_output_tiles_q (per batch)
    int input_tile_count;   // Total: num_input_tiles_p × num_input_tiles_q (per batch)

    // ========== Tiling Case Type ==========
    int case_type;  // 1: Case 1 (Sub-tiling), 2: Case 2 (Super-tiling)

    // ========== Total Operation Counts (across all batches) ==========
    // Pre-calculated in Python, C++ can use directly without batch_size multiplication
    int total_loads;         // input_tile_count * batch_size
    int total_ibuf_reads;    // output_tile_count * pixels_per_tile * batch_size
    int total_cim_computes;  // Same as total_ibuf_reads
    int total_obuf_writes;   // output_tile_count * pooled_pixels_per_tile * batch_size
    int total_stores;        // output_tile_count * batch_size

    /**
     * @brief Default constructor
     */
    TilingConfig()
        : output_tile_p(0),
          output_tile_q(0),
          input_tile_h(0),
          input_tile_w(0),
          input_tile_p(0),
          input_tile_q(0),
          num_output_tiles_p(0),
          num_output_tiles_q(0),
          num_input_tiles_p(0),
          num_input_tiles_q(0),
          output_tile_count(0),
          input_tile_count(0),
          case_type(0) {}

    /**
     * @brief Constructor with all parameters
     */
    TilingConfig(int out_p, int out_q, int in_h, int in_w, int in_p, int in_q, int out_tiles_p,
                 int out_tiles_q, int in_tiles_p, int in_tiles_q, int out_count, int in_count,
                 int case_t = 0, int t_loads = 0, int t_ibuf_reads = 0, int t_cim_computes = 0,
                 int t_obuf_writes = 0, int t_stores = 0)
        : output_tile_p(out_p),
          output_tile_q(out_q),
          input_tile_h(in_h),
          input_tile_w(in_w),
          input_tile_p(in_p),
          input_tile_q(in_q),
          num_output_tiles_p(out_tiles_p),
          num_output_tiles_q(out_tiles_q),
          num_input_tiles_p(in_tiles_p),
          num_input_tiles_q(in_tiles_q),
          output_tile_count(out_count),
          input_tile_count(in_count),
          case_type(case_t),
          total_loads(t_loads),
          total_ibuf_reads(t_ibuf_reads),
          total_cim_computes(t_cim_computes),
          total_obuf_writes(t_obuf_writes),
          total_stores(t_stores) {}
};

/**
 * @class TilingStrategy
 * @brief Independent Tiling Strategy - Flexible tile-based processing
 *
 * Memory Access Pattern:
 * - Load: Input tile (input_tile_h × input_tile_w × C) per tile
 * - Compute: Output tile pixels (output_tile_p × output_tile_q)
 * - Store: Output tile (output_tile_p × output_tile_q × M) per tile
 *
 * Characteristics:
 * - IBUF size: input_tile_h × input_tile_w × C
 * - OBUF size: output_tile_p × output_tile_q × M
 * - Flexibility: Tile sizes configurable per layer
 * - Trade-off: Balanced between memory usage and bandwidth
 */
class TilingStrategy : public CNNStrategy {
  private:
    TilingConfig tiling_;

    // Helper methods for case detection and ratio calculation
    bool is_case1() const;                        // Case 1: Sub-tiling (multiple inputs → 1 output)
    int get_input_tiles_per_output_tile() const;  // Case 1: how many input tiles per output tile
    int get_output_tiles_per_input_tile() const;  // Case 2: how many output tiles per input tile
    void validate_constraints() const;            // Validate Case 1/2 constraints
    void validate_pooling_alignment(const CNNParams& params) const;  // Validate pooling alignment

    // Ratio-based dependency calculation helpers
    double get_operation_ratio(int target_count, int source_count) const;
    int calculate_dependency_from_ratio(double ratio, int target_idx, int sources_per_target) const;

  public:
    /**
     * @brief Constructor with tiling configuration
     * @param tiling Tiling configuration parameters
     */
    explicit TilingStrategy(const TilingConfig& tiling) : tiling_(tiling) {
        validate_constraints();
    }

    int get_total_loads(const CNNParams& params) const override;
    int get_total_computes(const CNNParams& params) const override;
    int get_total_stores(const CNNParams& params) const override;
    int get_total_obuf_writes(const CNNParams& params) const override;

    int get_load_ibuf_idx(int load_idx, const CNNParams& params) const override;
    int get_compute_ibuf_idx(int ibuf_read_idx, const CNNParams& params) const override;
    int get_compute_obuf_idx(int obuf_write_idx, const CNNParams& params) const override;
    int get_store_obuf_idx(int store_idx, const CNNParams& params) const override;

    int get_ibuf_count() const override;
    int get_obuf_count() const override;
    int get_ibuf_size(const CNNParams& params) const override;
    int get_obuf_size(const CNNParams& params) const override;

    std::vector<Dependency> get_load_dependencies(int load_idx,
                                                  const CNNParams& params) const override;
    std::vector<Dependency> get_ibuf_read_dependencies(int ibuf_read_idx,
                                                       const CNNParams& params) const override;
    std::vector<Dependency> get_compute_dependencies(int compute_idx,
                                                     const CNNParams& params) const override;
    std::vector<Dependency> get_obuf_write_dependencies(int obuf_write_idx,
                                                        const CNNParams& params) const override;
    std::vector<Dependency> get_store_dependencies(int store_idx,
                                                   const CNNParams& params) const override;

    // Buffer-specific access functions
    std::pair<std::string, std::vector<std::pair<int, int>>>
    get_external_read_access(int load_idx, const CNNParams& params) const override;
    std::pair<std::string, std::vector<std::pair<int, int>>>
    get_external_write_access(int store_idx, const CNNParams& params) const override;
    std::pair<std::string, std::vector<std::pair<int, int>>>
    get_ibuf_write_access(int load_idx, const CNNParams& params) const override;
    std::pair<std::string, std::vector<std::pair<int, int>>>
    get_ibuf_read_access(int compute_idx, const CNNParams& params) const override;
    std::pair<std::string, std::vector<std::pair<int, int>>>
    get_obuf_write_access(int obuf_write_idx, const CNNParams& params) const override;
    std::pair<std::string, std::vector<std::pair<int, int>>>
    get_obuf_read_access(int store_idx, const CNNParams& params) const override;

    // Buffer shape functions
    std::vector<int> get_ibuf_shape(const CNNParams& params) const override;
    std::vector<int> get_obuf_shape(const CNNParams& params) const override;

    // Accessor for tiling configuration
    const TilingConfig& get_tiling_config() const { return tiling_; }
};

// ========================================================================
// Factory Function
// ========================================================================

/**
 * @brief Create a TilingStrategy instance with given configuration
 * @param tiling Tiling configuration parameters
 * @return Pointer to newly created TilingStrategy (caller owns memory)
 *
 * Usage:
 * ```cpp
 * TilingConfig tiling(24, 24, 28, 28, 1, 1, 1);
 * std::unique_ptr<CNNStrategy> strategy(create_tiling_strategy(tiling));
 * int loads = strategy->get_total_loads(params);
 * ```
 */
CNNStrategy* create_tiling_strategy(const TilingConfig& tiling);

#endif  // CNN_STRATEGY_H
