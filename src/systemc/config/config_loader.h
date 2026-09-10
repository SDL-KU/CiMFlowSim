/**
 * @file config_loader.h
 * @brief Configuration loading system for CiMFlowSim CNN accelerator simulator
 *
 * This module provides a comprehensive JSON-based configuration loading system:
 * - Loads CNN layer parameters (dimensions, batch size, pooling, etc.)
 * - Loads hardware timing configurations (compute time, memory latencies)
 * - Loads buffer architecture specifications (banks, line sizes, ports)
 * - Validates all loaded parameters against simulator constraints
 * - Provides clear error messages for configuration issues
 *
 * Configuration Structure:
 * - cnn_layer: CNN layer parameters (H, W, C, R, S, M, stride, batch_size, pooling)
 * - hardware_timing: Timing parameters (compute_time_ns, buffer configs)
 * - buffer_architecture: Memory architecture (num_banks, bits_per_line)
 * - port_configuration: Memory port configuration (num_rw_ports, etc.)
 */

#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>

#include "constants.h"
#include "exceptions.h"
#include "json.hpp"
#include "types.h"  // For CNNParams and HWConfig definitions

using json = nlohmann::json;

namespace config {

/**
 * @class ConfigurationLoader
 * @brief Static utility class for loading and validating JSON configurations
 *
 * Provides static methods to load different parts of the configuration:
 * - File loading with error handling
 * - Field validation against required schema
 * - Parameter extraction with type checking
 * - Hardware configuration loading with defaults
 */
class ConfigurationLoader {
  public:
    /**
     * @brief Load and parse JSON configuration file
     * @param config_path Path to JSON configuration file
     * @return Parsed JSON object
     * @throws ConfigurationError if file cannot be loaded or parsed
     */
    static json load_config_file(const std::string& config_path);

    /**
     * @brief Load CNN layer parameters from configuration
     * @param config Parsed JSON configuration object
     * @return CNNParams structure with layer dimensions and parameters
     * @throws ConfigurationError if CNN parameters are invalid
     */
    static CNNParams load_cnn_params(const json& config);

    /**
     * @brief Load hardware timing configuration
     * @param config Parsed JSON configuration object
     * @return HWConfig structure with timing and architecture parameters
     * @throws ConfigurationError if hardware parameters are invalid
     */
    static HWConfig load_hw_config(const json& config);

    /**
     * @brief Load tiling configuration from configuration file
     * @param config Parsed JSON configuration object
     * @return TilingConfig structure with tiling parameters
     * @throws ConfigurationError if tiling config is invalid
     */
    static TilingConfig load_tiling_config(const json& config);

    /**
     * @brief Load CNN layer parameters from network configuration
     * @param network_config Network configuration JSON
     * @param layer_index Layer index to extract
     * @param layer_id Layer ID to match (optional, fallback to layer_index)
     * @return CNNParams structure with layer dimensions and parameters
     * @throws ConfigurationError if layer not found or parameters invalid
     */
    static CNNParams load_cnn_params_from_network(const json& network_config, int layer_index,
                                                  const std::string& layer_id = "");

    /**
     * @brief Load hardware configuration from hardware JSON
     * @param hardware_config Hardware configuration JSON
     * @return HWConfig structure with timing and architecture parameters
     * @throws ConfigurationError if hardware configuration invalid
     */
    static HWConfig load_hw_config_from_hardware(const json& hardware_config);

  private:
    /**
     * @brief Extract pooling parameters from layer configuration
     * @param layer Layer configuration object
     * @return Pair of (pool_height, pool_width)
     */
    static std::pair<int, int> extract_pooling_params(const json& layer);
};

}  // namespace config

#endif  // CONFIG_LOADER_H
