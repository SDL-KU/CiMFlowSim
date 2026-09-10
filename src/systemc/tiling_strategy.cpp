/**
 * @file tiling_strategy.cpp
 * @brief Implementation of Independent Tiling Strategy
 *
 * Independent Tiling Strategy
 * ===========================
 *
 * Memory Access Pattern:
 * - Loads input tile (input_tile_h×input_tile_w×C) per tile
 * - Computes all output pixels in the tile (output_tile_p×output_tile_q)
 * - Stores output tile (output_tile_p×output_tile_q×M) per tile
 *
 * Characteristics:
 * - IBUF size: input_tile_h × input_tile_w × C
 * - OBUF size: output_tile_p × output_tile_q × M
 * - Flexibility: Configurable tile sizes per layer
 * - Trade-off: Balanced memory usage vs. bandwidth
 *
 * Buffer Management:
 * - Double buffering for both IBUF and OBUF
 * - Alternates between buffer indices {0, 1}
 * - Tile-level buffer switching for pipeline safety
 */

#include <algorithm>  // for std::min
#include <cmath>      // for sqrt() in Case 2 super-tiling calculation
#include <fstream>    // for debug file output

#include "pipeline_simulator.h"

// ============================================================================
// Independent Tiling Strategy - Memory Operations
// ============================================================================

/**
 * @brief Get total number of load operations for Tiling Strategy
 * @details Each tile requires one load of its input tile
 * @return batch_size × total_tiles (one load per tile)
 */
int TilingStrategy::get_total_loads([[maybe_unused]] const CNNParams& params) const {
    // REQUIRED: Use pre-calculated total from JSON (Phase 3)
    // NO FALLBACK - if this is 0, strategy JSON is invalid
    if (tiling_.total_loads == 0) {
        throw std::runtime_error(
            "Invalid strategy: total_loads is 0. Strategy JSON must include Phase 3 total "
            "operation counts.");
    }
    return tiling_.total_loads;
}

/**
 * @brief Get total number of compute operations for Tiling Strategy
 * @details Each tile produces output_tile_p × output_tile_q pixels
 * @return batch_size × total_tiles × output_tile_p × output_tile_q
 */
int TilingStrategy::get_total_computes([[maybe_unused]] const CNNParams& params) const {
    // REQUIRED: Use pre-calculated total from JSON (Phase 3)
    if (tiling_.total_cim_computes == 0) {
        throw std::runtime_error(
            "Invalid strategy: total_cim_computes is 0. Strategy JSON must include Phase 3 total "
            "operation counts.");
    }
    return tiling_.total_cim_computes;
}

/**
 * @brief Get total number of store operations for Tiling Strategy
 * @details Each tile produces one output tile to store
 * @return batch_size × total_tiles (one store per tile)
 */
int TilingStrategy::get_total_stores([[maybe_unused]] const CNNParams& params) const {
    // REQUIRED: Use pre-calculated total from JSON (Phase 3)
    if (tiling_.total_stores == 0) {
        throw std::runtime_error(
            "Invalid strategy: total_stores is 0. Strategy JSON must include Phase 3 total "
            "operation counts.");
    }
    return tiling_.total_stores;
}

/**
 * @brief Get total number of OBUF write operations for Tiling Strategy
 * @details OBUF writes happen per pooling group within each tile
 * @return total_computes / (pool_height × pool_width)
 */
int TilingStrategy::get_total_obuf_writes([[maybe_unused]] const CNNParams& params) const {
    // REQUIRED: Use pre-calculated total from JSON (Phase 3)
    if (tiling_.total_obuf_writes == 0) {
        throw std::runtime_error(
            "Invalid strategy: total_obuf_writes is 0. Strategy JSON must include Phase 3 total "
            "operation counts.");
    }
    return tiling_.total_obuf_writes;
}

// ========================================================================
// Buffer Management Functions
// ========================================================================

int TilingStrategy::get_load_ibuf_idx(int load_idx,
                                      [[maybe_unused]] const CNNParams& params) const {
    // LOAD is always <= IBUF_READ, so LOAD switches every time
    return load_idx % validation_constants::DOUBLE_BUFFER_COUNT;
}

int TilingStrategy::get_compute_ibuf_idx(int ibuf_read_idx,
                                         [[maybe_unused]] const CNNParams& params) const {
    // IBUF_READ is always >= LOAD, so switch every (ibuf_reads / loads) times
    // This ensures IBUF_READ uses the same buffer that LOAD wrote to
    int ibuf_reads_per_load = tiling_.total_ibuf_reads / tiling_.total_loads;
    return (ibuf_read_idx / ibuf_reads_per_load) % validation_constants::DOUBLE_BUFFER_COUNT;
}

int TilingStrategy::get_compute_obuf_idx(int obuf_write_idx,
                                         [[maybe_unused]] const CNNParams& params) const {
    // OBUF_WRITE is always >= STORE, so switch every (obuf_writes / stores) times
    // This ensures OBUF_WRITE uses the same buffer that STORE will read from
    int obuf_writes_per_store = tiling_.total_obuf_writes / tiling_.total_stores;
    return (obuf_write_idx / obuf_writes_per_store) % validation_constants::DOUBLE_BUFFER_COUNT;
}

int TilingStrategy::get_store_obuf_idx(int store_idx,
                                       [[maybe_unused]] const CNNParams& params) const {
    // STORE is always <= OBUF_WRITE, so STORE switches every time
    return store_idx % validation_constants::DOUBLE_BUFFER_COUNT;
}

int TilingStrategy::get_ibuf_count() const {
    return validation_constants::DOUBLE_BUFFER_COUNT;
}

int TilingStrategy::get_obuf_count() const {
    return validation_constants::DOUBLE_BUFFER_COUNT;
}

int TilingStrategy::get_ibuf_size(const CNNParams& params) const {
    // IBUF holds one input tile
    return tiling_.input_tile_h * tiling_.input_tile_w * params.C;
}

int TilingStrategy::get_obuf_size(const CNNParams& params) const {
    // OBUF holds one output tile
    return tiling_.output_tile_p * tiling_.output_tile_q * params.M;
}

// ========================================================================
// Helper Methods for Case Detection and Ratio Calculation
// ========================================================================

/**
 * @brief Check if this is Case 1 (Sub-tiling: multiple LOADs → 1 output tile)
 * @return true if total_input_loads > output_tile_count (Case 1), false otherwise (Case 2)
 */
bool TilingStrategy::is_case1() const {
    return tiling_.input_tile_count > tiling_.output_tile_count;
}

/**
 * @brief Get the number of LOAD operations required per output tile (Case 1)
 * @return loads_per_output_tile ratio for Case 1
 * @details Example: 576 loads / 144 output tiles = 4 loads per output tile
 */
int TilingStrategy::get_input_tiles_per_output_tile() const {
    // Case 1: Multiple input loads combine to produce one output tile
    return tiling_.input_tile_count / tiling_.output_tile_count;
}

/**
 * @brief Get the number of output tiles produced per LOAD operation (Case 2)
 * @return output_tiles_per_load ratio for Case 2
 * @details Example: 144 output tiles / 36 loads = 4 output tiles per load
 */
int TilingStrategy::get_output_tiles_per_input_tile() const {
    // Case 2: One input load produces multiple output tiles
    return tiling_.output_tile_count / tiling_.input_tile_count;
}

/**
 * @brief Validate Case 1 and Case 2 constraints
 * @throws std::runtime_error if constraints are violated
 *
 * Note: Independent Tiling does NOT require strict constraints.
 * The old constraints were based on Aligned Tiling assumptions.
 * We only validate basic sanity checks here.
 */
void TilingStrategy::validate_constraints() const {
    // Basic sanity checks only
    if (tiling_.output_tile_count == 0) {
        throw std::runtime_error("output_tile_count must be > 0");
    }
    if (tiling_.input_tile_count == 0) {
        throw std::runtime_error("input_tile_count must be > 0");
    }
    if (tiling_.output_tile_p == 0 || tiling_.output_tile_q == 0) {
        throw std::runtime_error("output_tile dimensions must be > 0");
    }
    if (tiling_.input_tile_h == 0 || tiling_.input_tile_w == 0) {
        throw std::runtime_error("input_tile dimensions must be > 0");
    }

    // Note: We do NOT enforce ratio constraints because Independent Tiling
    // allows flexible input/output tile relationships that don't follow
    // the strict sub-tiling or super-tiling patterns of Aligned Tiling.
}

/**
 * @brief Validate pooling alignment constraints
 * @param params CNN parameters with pooling settings
 * @throws std::runtime_error if pooling constraints are violated
 */
void TilingStrategy::validate_pooling_alignment(const CNNParams& params) const {
    // Check if output tile dimensions are divisible by pooling dimensions
    if (tiling_.output_tile_p % params.pool_height != 0) {
        throw std::runtime_error(
            "Pooling alignment violation: output_tile_p (" + std::to_string(tiling_.output_tile_p) +
            ") must be divisible by pool_height (" + std::to_string(params.pool_height) + ")");
    }

    if (tiling_.output_tile_q % params.pool_width != 0) {
        throw std::runtime_error(
            "Pooling alignment violation: output_tile_q (" + std::to_string(tiling_.output_tile_q) +
            ") must be divisible by pool_width (" + std::to_string(params.pool_width) + ")");
    }

    // Check if total output dimensions are properly aligned
    if (params.P % tiling_.output_tile_p != 0) {
        throw std::runtime_error(
            "Output dimension alignment violation: P (" + std::to_string(params.P) +
            ") must be divisible by output_tile_p (" + std::to_string(tiling_.output_tile_p) + ")");
    }

    if (params.Q % tiling_.output_tile_q != 0) {
        throw std::runtime_error(
            "Output dimension alignment violation: Q (" + std::to_string(params.Q) +
            ") must be divisible by output_tile_q (" + std::to_string(tiling_.output_tile_q) + ")");
    }
}

// ========================================================================
// Dependency Functions
// ========================================================================

std::vector<Dependency> TilingStrategy::get_load_dependencies(
    int load_idx, [[maybe_unused]] const CNNParams& params) const {
    std::vector<Dependency> deps;

    // ========================================================================
    // Backward Dependency: Double buffering - IBUF buffer reuse
    // ========================================================================
    // LOAD writes to IBUF buffer[load_idx % 2]
    // Before LOAD[n] can reuse buffer k, all IBUF_READs using data from
    // LOAD[n-2] (which also used buffer k) must complete.
    //
    // We need to find the LAST IBUF_READ that depends on LOAD[load_idx - 2]

    if (load_idx >= 2) {
        int prev_load_same_buffer = load_idx - 2;

        // Simple calculation: Each LOAD serves equal number of IBUF_READs
        // LOAD N waits for the last IBUF_READ served by LOAD (N-2)
        int total_ibuf_reads = tiling_.total_ibuf_reads;
        int total_loads = tiling_.total_loads;

        int ibuf_reads_per_load = total_ibuf_reads / total_loads;
        int last_ibuf_read_idx = (prev_load_same_buffer + 1) * ibuf_reads_per_load - 1;

        // Clamp to valid range
        if (last_ibuf_read_idx >= total_ibuf_reads) {
            last_ibuf_read_idx = total_ibuf_reads - 1;
        }

        deps.push_back(Dependency(OperationType::IBUF_READ, last_ibuf_read_idx));
    }

    return deps;
}

std::vector<Dependency>
TilingStrategy::get_ibuf_read_dependencies(int ibuf_read_idx,
                                           [[maybe_unused]] const CNNParams& params) const {
    std::vector<Dependency> deps;

    // ========================================================================
    // Forward Dependency: IBUF_READ → LOAD (simple ratio calculation)
    // ========================================================================
    // Simple calculation: Each LOAD serves equal number of IBUF_READs
    // IBUF_READ N depends on LOAD (N / ibuf_reads_per_load)

    int total_ibuf_reads = tiling_.total_ibuf_reads;
    int total_loads = tiling_.total_loads;
    int ibuf_reads_per_load = total_ibuf_reads / total_loads;

    int load_idx = ibuf_read_idx / ibuf_reads_per_load;

    // Clamp to valid range
    if (load_idx >= total_loads) {
        load_idx = total_loads - 1;
    }

    deps.push_back(Dependency(OperationType::LOAD, load_idx));

    // ========================================================================
    // Backward Dependency: Double buffering (unchanged)
    // ========================================================================
    if (ibuf_read_idx >= 2) {
        deps.push_back(Dependency(OperationType::CIM_COMPUTE, ibuf_read_idx - 2));
    }

    return deps;
}

std::vector<Dependency> TilingStrategy::get_compute_dependencies(int compute_idx,
                                                                 const CNNParams& params) const {
    std::vector<Dependency> deps;

    // Forward dependency: COMPUTE always has 1:1 mapping with IBUF_READ
    // (IBUF_READ handles the actual LOAD dependencies)
    deps.push_back(Dependency(OperationType::IBUF_READ, compute_idx));

    // Backward dependency: OBUF buffer reuse
    int pooling_group_idx = compute_idx / (params.pool_height * params.pool_width);
    if (pooling_group_idx >= 2) {
        int prev_obuf_write = pooling_group_idx - 2;
        deps.push_back(Dependency(OperationType::OBUF_WRITE, prev_obuf_write));
    }

    return deps;
}

std::vector<Dependency> TilingStrategy::get_obuf_write_dependencies(int obuf_write_idx,
                                                                    const CNNParams& params) const {
    std::vector<Dependency> deps;

    // ========================================================================
    // Forward Dependency: OBUF_WRITE → COMPUTE (unchanged - pooling based)
    // ========================================================================
    int pixels_per_pool = params.pool_height * params.pool_width;
    int last_compute_in_group = (obuf_write_idx + 1) * pixels_per_pool - 1;
    deps.push_back(Dependency(OperationType::CIM_COMPUTE, last_compute_in_group));

    // ========================================================================
    // Backward Dependency: OBUF_WRITE → STORE (simple calculation)
    // ========================================================================
    // Simple calculation: Each STORE handles one output tile
    // Each tile produces pooled_pixels_per_tile OBUF_WRITEs
    // OBUF_WRITE N depends on STORE (N / pooled_pixels_per_tile)

    int pooled_pixels_per_tile =
        (tiling_.output_tile_p / params.pool_height) * (tiling_.output_tile_q / params.pool_width);

    int corresponding_store_idx = obuf_write_idx / pooled_pixels_per_tile;

    // Wait for store from 2 operations back (double buffering)
    if (corresponding_store_idx >= 2) {
        int prev_store_idx = corresponding_store_idx - 2;
        deps.push_back(Dependency(OperationType::STORE, prev_store_idx));
    }

    return deps;
}

std::vector<Dependency> TilingStrategy::get_store_dependencies(int store_idx,
                                                               const CNNParams& params) const {
    std::vector<Dependency> deps;

    // ========================================================================
    // Forward Dependency: STORE → OBUF_WRITE (simple calculation)
    // ========================================================================
    // Simple calculation: Each STORE waits for all OBUF_WRITEs from its tile
    // Each tile produces pooled_pixels_per_tile OBUF_WRITEs
    // STORE N depends on the last OBUF_WRITE = N * pooled_pixels_per_tile + (pooled_pixels_per_tile - 1)

    int pooled_pixels_per_tile =
        (tiling_.output_tile_p / params.pool_height) * (tiling_.output_tile_q / params.pool_width);

    int obuf_write_idx = store_idx * pooled_pixels_per_tile + (pooled_pixels_per_tile - 1);

    deps.push_back(Dependency(OperationType::OBUF_WRITE, obuf_write_idx));

    return deps;
}

// ========================================================================
// Buffer Access Functions
// ========================================================================

std::pair<std::string, std::vector<std::pair<int, int>>>
TilingStrategy::get_external_read_access(int load_idx, const CNNParams& params) const {
    // Calculate which batch this load belongs to
    int loads_per_batch = tiling_.input_tile_count;
    int batch_idx = load_idx / loads_per_batch;
    int load_in_batch = load_idx % loads_per_batch;

    // Validate batch_idx is within bounds
    if (batch_idx >= params.batch_size) {
        std::cerr << "[ERROR] batch_idx=" << batch_idx << " >= batch_size=" << params.batch_size
                  << " (load_idx=" << load_idx << ", loads_per_batch=" << loads_per_batch << ")"
                  << std::endl;
        throw std::runtime_error("Batch index out of bounds in get_external_read_access");
    }

    int h_start, w_start, h_end, w_end;

    if (tiling_.case_type == 1) {
        // ========================================================================
        // Case 1 (Sub-tiling): Multiple input tiles → 1 output tile
        // ========================================================================
        // Calculate inputs_per_output ratio
        int inputs_per_output = tiling_.input_tile_count / tiling_.output_tile_count;

        // Find which output tile this input load belongs to
        int output_tile_idx = load_in_batch / inputs_per_output;
        int input_within_output = load_in_batch % inputs_per_output;

        // Output tile position in grid
        int output_tile_p = output_tile_idx / tiling_.num_output_tiles_q;
        int output_tile_q = output_tile_idx % tiling_.num_output_tiles_q;

        // Calculate sub-tile dimensions within output tile
        // For example, 2×2 output tile with 4 inputs → 2×2 sub-tiles (1×1 each)
        int sub_tiles_q = tiling_.output_tile_q / tiling_.input_tile_q;

        // Sub-tile position within output tile
        int sub_tile_p = input_within_output / sub_tiles_q;
        int sub_tile_q = input_within_output % sub_tiles_q;

        // Output coordinates of this sub-tile
        int output_p_start =
            output_tile_p * tiling_.output_tile_p + sub_tile_p * tiling_.input_tile_p;
        int output_q_start =
            output_tile_q * tiling_.output_tile_q + sub_tile_q * tiling_.input_tile_q;

        // Convert to input coordinates
        h_start = output_p_start * params.stride;
        w_start = output_q_start * params.stride;
        h_end = h_start + tiling_.input_tile_h;
        w_end = w_start + tiling_.input_tile_w;

    } else if (tiling_.case_type == 2) {
        // ========================================================================
        // Case 2 (Super-tiling): 1 input tile → Multiple output tiles
        // ========================================================================
        // Input tile position in the grid
        int input_tile_idx_p = load_in_batch / tiling_.num_input_tiles_q;
        int input_tile_idx_q = load_in_batch % tiling_.num_input_tiles_q;

        // OUTPUT coordinate this tile covers
        int output_p_start = input_tile_idx_p * tiling_.input_tile_p;
        int output_q_start = input_tile_idx_q * tiling_.input_tile_q;

        // Convert OUTPUT coordinates to INPUT coordinates (apply stride)
        h_start = output_p_start * params.stride;
        w_start = output_q_start * params.stride;

        // Add input tile size (actual INPUT coordinate size)
        h_end = h_start + tiling_.input_tile_h;
        w_end = w_start + tiling_.input_tile_w;

    } else {
        throw std::runtime_error("Invalid case_type: " + std::to_string(tiling_.case_type));
    }

    // Clamp to input bounds (important for boundary tiles)
    h_end = std::min(h_end, params.H);
    w_end = std::min(w_end, params.W);

    return {"input_tensor",
            {
                {batch_idx, batch_idx + 1},  // Batch dimension
                {0, params.C},               // All channels
                {h_start, h_end},            // Height window
                {w_start, w_end}             // Width window
            }};
}

std::pair<std::string, std::vector<std::pair<int, int>>>
TilingStrategy::get_external_write_access(int store_idx, const CNNParams& params) const {
    // TensorMemorySystem expects 4D coordinates: [batch, channel, height, width]

    // store_idx -> batch, output_tile (same for both Case 1 and Case 2)
    // output_tile_count is per-batch value, so use it directly
    int stores_per_batch = tiling_.output_tile_count;
    int batch_idx = store_idx / stores_per_batch;
    int tile_idx = store_idx % stores_per_batch;

    // Validate batch_idx is within bounds
    if (batch_idx >= params.batch_size) {
        std::cerr << "[ERROR] batch_idx=" << batch_idx << " >= batch_size=" << params.batch_size
                  << " (store_idx=" << store_idx << ", stores_per_batch=" << stores_per_batch << ")"
                  << std::endl;
        throw std::runtime_error("Batch index out of bounds in get_external_write_access");
    }

    // Calculate output tile position
    int tile_p = tile_idx / tiling_.num_output_tiles_q;
    int tile_q = tile_idx % tiling_.num_output_tiles_q;

    // Output tile size after pooling
    int pooled_tile_p = tiling_.output_tile_p / params.pool_height;
    int pooled_tile_q = tiling_.output_tile_q / params.pool_width;

    int start_p = tile_p * pooled_tile_p;
    int start_q = tile_q * pooled_tile_q;

    // Return 4D range: [batch, channel, height, width]
    return {"output_tensor",
            {
                {batch_idx, batch_idx + 1},          // Batch dimension
                {0, params.M},                       // All output channels
                {start_p, start_p + pooled_tile_p},  // Height window (pooled)
                {start_q, start_q + pooled_tile_q}   // Width window (pooled)
            }};
}

std::pair<std::string, std::vector<std::pair<int, int>>>
TilingStrategy::get_ibuf_write_access(int load_idx, const CNNParams& params) const {
    (void)load_idx;  // Buffer index handled by double buffering

    // Write input tile to IBUF - use tile coordinates (batch dimension fixed)
    return {"input_tensor",
            {
                {0, 1},                     // Batch dimension fixed for buffer
                {0, params.C},              // All channels
                {0, tiling_.input_tile_h},  // Tile height
                {0, tiling_.input_tile_w}   // Tile width
            }};
}

std::pair<std::string, std::vector<std::pair<int, int>>>
TilingStrategy::get_ibuf_read_access(int compute_idx, const CNNParams& params) const {
    // Calculate which output pixel within the tile this compute corresponds to
    int pixels_per_tile = tiling_.output_tile_p * tiling_.output_tile_q;
    int tile_idx = compute_idx / pixels_per_tile;
    int pixel_in_tile = compute_idx % pixels_per_tile;

    // Calculate position within output tile
    int pixel_p = pixel_in_tile / tiling_.output_tile_q;
    int pixel_q = pixel_in_tile % tiling_.output_tile_q;

    if (is_case1()) {
        // Case 1: Sub-tiling - multiple input tiles combine to produce one output tile
        // Each input tile covers (input_tile_p × input_tile_q) OUTPUT pixels
        // We need to:
        // 1. Determine which input sub-tile this pixel belongs to
        // 2. Calculate the pixel's position within that sub-tile
        // 3. Map that to the R×S receptive field in the input tile

        // Position within the sub-tile (in OUTPUT coordinate)
        int pixel_within_subtile_p = pixel_p % tiling_.input_tile_p;
        int pixel_within_subtile_q = pixel_q % tiling_.input_tile_q;

        // Calculate offset in INPUT coordinate (apply stride)
        int input_h_offset = pixel_within_subtile_p * params.stride;
        int input_w_offset = pixel_within_subtile_q * params.stride;

        // Read R×S receptive field starting from the offset
        int h_start = input_h_offset;
        int w_start = input_w_offset;
        int h_end = input_h_offset + params.R;
        int w_end = input_w_offset + params.S;

        // Clamp to tile boundaries (important for boundary tiles)
        h_end = std::min(h_end, tiling_.input_tile_h);
        w_end = std::min(w_end, tiling_.input_tile_w);

        return {"input_tensor",
                {
                    {0, 1},            // Batch dimension fixed for buffer
                    {0, params.C},     // All channels
                    {h_start, h_end},  // Height window within sub-tile
                    {w_start, w_end}   // Width window within sub-tile
                }};
    } else {
        // Case 2: Super-tiling - one input tile serves multiple output tiles

        // Step 1: Determine which output tile this compute belongs to
        int output_tile_p = tile_idx / tiling_.num_output_tiles_q;
        int output_tile_q = tile_idx % tiling_.num_output_tiles_q;

        // Step 2: Calculate the absolute output pixel position (in OUTPUT coordinate)
        int abs_output_p = output_tile_p * tiling_.output_tile_p + pixel_p;
        int abs_output_q = output_tile_q * tiling_.output_tile_q + pixel_q;

        // Step 3: Calculate the receptive field in INPUT coordinate
        int abs_input_h_start = abs_output_p * params.stride;
        int abs_input_w_start = abs_output_q * params.stride;
        int abs_input_h_end = abs_input_h_start + params.R;
        int abs_input_w_end = abs_input_w_start + params.S;

        // Step 4: Determine which input tile this output pixel uses
        // (needed to calculate offset within the IBUF-stored tile)
        int input_tile_for_this_output_p = abs_output_p / tiling_.input_tile_p;
        int input_tile_for_this_output_q = abs_output_q / tiling_.input_tile_q;

        // Step 5: Calculate the input tile's starting position in INPUT coordinate
        int input_tile_abs_h_start =
            input_tile_for_this_output_p * tiling_.input_tile_p * params.stride;
        int input_tile_abs_w_start =
            input_tile_for_this_output_q * tiling_.input_tile_q * params.stride;

        // Step 6: Calculate relative position within the IBUF tile
        int h_start = abs_input_h_start - input_tile_abs_h_start;
        int w_start = abs_input_w_start - input_tile_abs_w_start;
        int h_end = abs_input_h_end - input_tile_abs_h_start;
        int w_end = abs_input_w_end - input_tile_abs_w_start;

        // Step 7: Clamp to tile boundaries
        h_end = std::min(h_end, tiling_.input_tile_h);
        w_end = std::min(w_end, tiling_.input_tile_w);

        return {"input_tensor",
                {
                    {0, 1},            // Batch dimension fixed for buffer
                    {0, params.C},     // All channels
                    {h_start, h_end},  // Height window within tile
                    {w_start, w_end}   // Width window within tile
                }};
    }
}

std::pair<std::string, std::vector<std::pair<int, int>>>
TilingStrategy::get_obuf_write_access(int obuf_write_idx, const CNNParams& params) const {
    // OBUF writes happen per pooling group
    // Calculate which pooled pixel this write corresponds to
    int pooled_pixels_per_tile =
        (tiling_.output_tile_p / params.pool_height) * (tiling_.output_tile_q / params.pool_width);

    // Calculate pixel position within the tile
    int pixel_in_tile = obuf_write_idx % pooled_pixels_per_tile;

    // Calculate pooled pixel position within the tile
    int pooled_tile_q = tiling_.output_tile_q / params.pool_width;

    int pooled_p = pixel_in_tile / pooled_tile_q;
    int pooled_q = pixel_in_tile % pooled_tile_q;

    // Write single pooled pixel to OBUF
    return {"output_tensor",
            {
                {0, 1},                    // Batch dimension fixed for buffer
                {0, params.M},             // All output channels
                {pooled_p, pooled_p + 1},  // Single pooled pixel height
                {pooled_q, pooled_q + 1}   // Single pooled pixel width
            }};
}

std::pair<std::string, std::vector<std::pair<int, int>>>
TilingStrategy::get_obuf_read_access(int store_idx, const CNNParams& params) const {
    (void)store_idx;  // Buffer index handled by double buffering

    // Read output tile from OBUF - use tile coordinates (batch dimension fixed)
    int pooled_tile_p = tiling_.output_tile_p / params.pool_height;
    int pooled_tile_q = tiling_.output_tile_q / params.pool_width;

    return {"output_tensor",
            {
                {0, 1},              // Batch dimension fixed for buffer
                {0, params.M},       // All output channels
                {0, pooled_tile_p},  // Pooled tile height
                {0, pooled_tile_q}   // Pooled tile width
            }};
}

// ========================================================================
// Buffer Shape Functions
// ========================================================================

std::vector<int> TilingStrategy::get_ibuf_shape(const CNNParams& params) const {
    // IBUF stores input tile {batch=1, C, tile_h, tile_w}
    return {1, params.C, tiling_.input_tile_h, tiling_.input_tile_w};
}

std::vector<int> TilingStrategy::get_obuf_shape(const CNNParams& params) const {
    // OBUF stores pooled output tile {batch=1, M, pooled_tile_p, pooled_tile_q}
    int pooled_tile_p = tiling_.output_tile_p / params.pool_height;
    int pooled_tile_q = tiling_.output_tile_q / params.pool_width;
    return {1, params.M, pooled_tile_p, pooled_tile_q};
}

// ========================================================================
// Ratio-Based Dependency Calculation Helpers
// ========================================================================

/**
 * @brief Calculate operation count ratio for dependency inference
 * @param target_count Number of target operations
 * @param source_count Number of source operations
 * @return Ratio of target to source operations
 * @details
 * - ratio == 1.0: 1:1 mapping (each target has unique source)
 * - ratio > 1.0: Many:1 mapping (multiple targets share one source)
 * - ratio < 1.0: 1:Many mapping (one target waits for multiple sources)
 */
double TilingStrategy::get_operation_ratio(int target_count, int source_count) const {
    if (source_count == 0) {
        throw std::runtime_error("Division by zero in operation ratio calculation");
    }
    return static_cast<double>(target_count) / static_cast<double>(source_count);
}

/**
 * @brief Calculate dependency index from operation ratio
 * @param ratio Operation count ratio (target/source)
 * @param target_idx Index of the current target operation
 * @param sources_per_target Number of source operations per target (for ratio < 1.0)
 * @return Index of the source operation this target depends on
 * @details
 * Mapping logic:
 * - ratio == 1.0: Direct mapping → source[target_idx]
 * - ratio > 1.0: Many:1 mapping → source[target_idx / ratio]
 * - ratio < 1.0: 1:Many mapping → source[(target_idx + 1) × sources_per_target - 1] (last one)
 */
int TilingStrategy::calculate_dependency_from_ratio(double ratio, int target_idx,
                                                    int sources_per_target) const {
    const double EPSILON = 1e-6;  // For floating point comparison

    if (std::abs(ratio - 1.0) < EPSILON) {
        // 1:1 mapping - direct index
        return target_idx;
    } else if (ratio > 1.0) {
        // Many:1 mapping - multiple targets share one source
        // Find which source group this target belongs to
        return static_cast<int>(target_idx / ratio);
    } else {
        // 1:Many mapping - one target waits for multiple sources
        // Wait for the last source in the group
        return (target_idx + 1) * sources_per_target - 1;
    }
}
