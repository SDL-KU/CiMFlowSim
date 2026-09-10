#include "config_loader.h"

#include <fstream>
#include <iostream>

#include "../pipeline_simulator.h"  // For CNNParams and HWConfig definitions

namespace config {

json ConfigurationLoader::load_config_file(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw FileNotFoundError(config_path);
    }

    json config;
    try {
        config_file >> config;
    } catch (const json::parse_error& e) {
        throw JSONParseError(config_path, e.what());
    }

    return config;
}

std::pair<int, int> ConfigurationLoader::extract_pooling_params(const json& layer) {
    int pool_h = config_defaults::DEFAULT_POOL_HEIGHT;
    int pool_w = config_defaults::DEFAULT_POOL_WIDTH;

    if (layer.contains("pooling")) {
        const auto& pooling = layer["pooling"];
        pool_h = pooling["pool_height"];
        pool_w = pooling["pool_width"];
    } else if (layer.contains("pool_height") && layer.contains("pool_width")) {
        pool_h = layer["pool_height"];
        pool_w = layer["pool_width"];
    }

    // Validate pooling parameters (used as divisors in tiling calculations)
    if (pool_h <= 0) {
        throw std::runtime_error("Invalid config: pool_height must be > 0");
    }
    if (pool_w <= 0) {
        throw std::runtime_error("Invalid config: pool_width must be > 0");
    }

    return {pool_h, pool_w};
}

CNNParams ConfigurationLoader::load_cnn_params(const json& config) {
    try {
        const auto& layer = config["cnn_layer"];

        // Extract pooling parameters
        auto [pool_h, pool_w] = extract_pooling_params(layer);

#if ENABLE_VERBOSE_LOGS
        std::cout << "[CONFIG_LOADER] Loading CNNParams from JSON:" << std::endl;
        std::cout << "  Input: H=" << layer["H"] << ", W=" << layer["W"] << ", C=" << layer["C"]
                  << std::endl;
        std::cout << "  Kernel: R=" << layer["R"] << ", S=" << layer["S"] << ", M=" << layer["M"]
                  << std::endl;
        std::cout << "  stride=" << layer["stride"] << ", batch_size=" << layer["batch_size"]
                  << std::endl;
        std::cout << "  pooling: " << pool_h << "×" << pool_w << std::endl;
        std::cout << "  bitwidth: input=" << layer["input_bitwidth"]
                  << ", output=" << layer["output_bitwidth"] << std::endl;
#endif

        // Use CNNParams constructor which automatically calculates P and Q
        CNNParams params(layer["H"], layer["W"], layer["C"], layer["R"], layer["S"], layer["M"],
                         layer["stride"], layer["batch_size"], layer["input_bitwidth"],
                         layer["output_bitwidth"], pool_h, pool_w);

        // Override with pre-calculated values from JSON (required by validation)
        // Python calculates these correctly accounting for padding
        params.P = layer["P"];
        params.Q = layer["Q"];
        params.P_pooled = layer["P_pooled"];
        params.Q_pooled = layer["Q_pooled"];

#if ENABLE_VERBOSE_LOGS
        std::cout << "  Using P=" << params.P << ", Q=" << params.Q << " from JSON" << std::endl;
        std::cout << "  Using P_pooled=" << params.P_pooled << ", Q_pooled=" << params.Q_pooled
                  << " from JSON" << std::endl;
#endif

        return params;
    } catch (const json::exception& e) {
        throw ConfigurationError("Invalid CNN layer configuration: " + std::string(e.what()));
    }
}

HWConfig ConfigurationLoader::load_hw_config(const json& config) {
    try {
        // Require unified format only: config["hardware"]["ibuf"]["timing"]
        if (!config.contains("hardware")) {
            throw std::runtime_error(
                "Hardware configuration must use unified format: config['hardware']['ibuf']\n"
                "Legacy flat format (config['ibuf']) is no longer supported.");
        }

        const auto& hardware = config["hardware"];
#if ENABLE_VERBOSE_LOGS
        std::cout << "[CONFIG_LOADER] Loading unified hardware format (hardware.*)" << std::endl;
#endif

        // Load buffer-specific timing configurations - all required
        if (!hardware.contains("ibuf") || !hardware["ibuf"].contains("timing")) {
            throw std::runtime_error("Missing required hardware.ibuf.timing configuration");
        }
        const auto& ibuf_timing = hardware["ibuf"]["timing"];
        MemoryTimingConfig ibuf_config(ibuf_timing["base_latency_cycles"],
                                       ibuf_timing["clk_frequency_mhz"]);
        if (ibuf_config.clk_frequency_mhz <= 0) {
            throw std::runtime_error("Invalid config: ibuf.clk_frequency_mhz must be > 0");
        }
        if (ibuf_config.base_latency_cycles < 0) {
            throw std::runtime_error("Invalid config: ibuf.base_latency_cycles must be >= 0");
        }

        if (!hardware.contains("obuf") || !hardware["obuf"].contains("timing")) {
            throw std::runtime_error("Missing required hardware.obuf.timing configuration");
        }
        const auto& obuf_timing = hardware["obuf"]["timing"];
        MemoryTimingConfig obuf_config(obuf_timing["base_latency_cycles"],
                                       obuf_timing["clk_frequency_mhz"]);
        if (obuf_config.clk_frequency_mhz <= 0) {
            throw std::runtime_error("Invalid config: obuf.clk_frequency_mhz must be > 0");
        }
        if (obuf_config.base_latency_cycles < 0) {
            throw std::runtime_error("Invalid config: obuf.base_latency_cycles must be >= 0");
        }

        if (!hardware.contains("external") || !hardware["external"].contains("timing")) {
            throw std::runtime_error("Missing required hardware.external.timing configuration");
        }
        const auto& external_timing = hardware["external"]["timing"];
        MemoryTimingConfig external_config(external_timing["base_latency_cycles"],
                                           external_timing["clk_frequency_mhz"]);
        if (external_config.clk_frequency_mhz <= 0) {
            throw std::runtime_error("Invalid config: external.clk_frequency_mhz must be > 0");
        }
        if (external_config.base_latency_cycles < 0) {
            throw std::runtime_error("Invalid config: external.base_latency_cycles must be >= 0");
        }

        // Load individual buffer configurations - all required
        if (!hardware.contains("ibuf") || !hardware["ibuf"].contains("architecture")) {
            throw std::runtime_error("Missing required hardware.ibuf.architecture configuration");
        }
        const auto& ibuf_arch = hardware["ibuf"]["architecture"];
        BankLineConfig ibuf_bank_config(ibuf_arch["num_banks"], ibuf_arch["bits_per_line"]);
        if (ibuf_bank_config.num_banks <= 0) {
            throw std::runtime_error("Invalid config: ibuf.num_banks must be > 0");
        }
        if (ibuf_bank_config.bits_per_line <= 0) {
            throw std::runtime_error("Invalid config: ibuf.bits_per_line must be > 0");
        }

        if (!hardware.contains("obuf") || !hardware["obuf"].contains("architecture")) {
            throw std::runtime_error("Missing required hardware.obuf.architecture configuration");
        }
        const auto& obuf_arch = hardware["obuf"]["architecture"];
        BankLineConfig obuf_bank_config(obuf_arch["num_banks"], obuf_arch["bits_per_line"]);
        if (obuf_bank_config.num_banks <= 0) {
            throw std::runtime_error("Invalid config: obuf.num_banks must be > 0");
        }
        if (obuf_bank_config.bits_per_line <= 0) {
            throw std::runtime_error("Invalid config: obuf.bits_per_line must be > 0");
        }

        if (!hardware.contains("external") || !hardware["external"].contains("architecture")) {
            throw std::runtime_error(
                "Missing required hardware.external.architecture configuration");
        }
        const auto& external_arch = hardware["external"]["architecture"];
        BankLineConfig external_bank_config(external_arch["num_banks"],
                                            external_arch["bits_per_line"]);
        if (external_bank_config.num_banks <= 0) {
            throw std::runtime_error("Invalid config: external.num_banks must be > 0");
        }
        if (external_bank_config.bits_per_line <= 0) {
            throw std::runtime_error("Invalid config: external.bits_per_line must be > 0");
        }

        // Load CIM compute time
        if (!hardware.contains("cim") || !hardware["cim"].contains("compute_time_ns")) {
            throw std::runtime_error("Missing required hardware.cim.compute_time_ns configuration");
        }
        double compute_time_ns = hardware["cim"]["compute_time_ns"];
        if (compute_time_ns <= 0) {
            throw std::runtime_error("Invalid config: cim.compute_time_ns must be > 0");
        }

        return HWConfig(compute_time_ns, ibuf_config, obuf_config, external_config,
                        ibuf_bank_config, obuf_bank_config, external_bank_config);
    } catch (const json::exception& e) {
        throw ConfigurationError("Invalid hardware configuration: " + std::string(e.what()));
    }
}

TilingConfig ConfigurationLoader::load_tiling_config(const json& config) {
    try {
        if (!config.contains("tiling_config")) {
            throw ConfigurationError("Tiling configuration missing for tiling strategy");
        }

        const json& tiling = config["tiling_config"];

        // Validate required fields (base configuration)
        std::vector<std::string> required_fields = {
            "output_tile_p",      "output_tile_q",      "input_tile_h",
            "input_tile_w",       "input_tile_p",       "input_tile_q",  // Input→Output mapping
            "num_output_tiles_p", "num_output_tiles_q",                  // Output tile counts
            "num_input_tiles_p",  "num_input_tiles_q",                   // Input tile counts
            "output_tile_count",  "input_tile_count",                    // Total counts
            "case_type"                                                  // Tiling case type
        };

        for (const auto& field : required_fields) {
            if (!tiling.contains(field)) {
                throw ConfigurationError("Missing required tiling field: " + field);
            }
        }

        // Extract tiling parameters
#if ENABLE_VERBOSE_LOGS
        std::cout << "[CONFIG_LOADER] Loading TilingConfig from JSON:" << std::endl;
        std::cout << "  output_tile: " << tiling["output_tile_p"] << "×" << tiling["output_tile_q"]
                  << std::endl;
        std::cout << "  input_tile: " << tiling["input_tile_h"] << "×" << tiling["input_tile_w"]
                  << " → " << tiling["input_tile_p"] << "×" << tiling["input_tile_q"]
                  << " (output space)" << std::endl;
        std::cout << "  num_output_tiles: " << tiling["num_output_tiles_p"] << "×"
                  << tiling["num_output_tiles_q"] << std::endl;
        std::cout << "  num_input_tiles: " << tiling["num_input_tiles_p"] << "×"
                  << tiling["num_input_tiles_q"] << std::endl;
        std::cout << "  output_tile_count: " << tiling["output_tile_count"] << std::endl;
        std::cout << "  input_tile_count: " << tiling["input_tile_count"] << std::endl;
        std::cout << "  case_type: " << tiling["case_type"] << std::endl;
#endif

        // Read total operation counts (REQUIRED - Phase 3)
        // These must be pre-calculated in Python to avoid batch_size confusion
        std::vector<std::string> required_total_fields = {"total_loads", "total_ibuf_reads",
                                                          "total_cim_computes", "total_obuf_writes",
                                                          "total_stores"};

        for (const auto& field : required_total_fields) {
            if (!tiling.contains(field)) {
                throw ConfigurationError(
                    "Missing required Phase 3 field: " + field +
                    "\nStrategy JSON must include pre-calculated total operation counts.");
            }
        }

        int t_loads = tiling["total_loads"];
        int t_ibuf_reads = tiling["total_ibuf_reads"];
        int t_cim_computes = tiling["total_cim_computes"];
        int t_obuf_writes = tiling["total_obuf_writes"];
        int t_stores = tiling["total_stores"];

        // Validate values used as divisors to prevent division by zero
        int output_tile_count = tiling["output_tile_count"];
        int input_tile_count = tiling["input_tile_count"];
        int num_output_tiles_q = tiling["num_output_tiles_q"];

        if (output_tile_count <= 0) {
            throw std::runtime_error("Invalid config: output_tile_count must be > 0");
        }
        if (input_tile_count <= 0) {
            throw std::runtime_error("Invalid config: input_tile_count must be > 0");
        }
        if (num_output_tiles_q <= 0) {
            throw std::runtime_error("Invalid config: num_output_tiles_q must be > 0");
        }
        if (t_loads <= 0) {
            throw std::runtime_error("Invalid config: total_loads must be > 0");
        }
        if (t_stores <= 0) {
            throw std::runtime_error("Invalid config: total_stores must be > 0");
        }

#if ENABLE_VERBOSE_LOGS
        std::cout << "  [Phase 3] Total operation counts:" << std::endl;
        std::cout << "    Loads: " << t_loads << std::endl;
        std::cout << "    IBUF Reads: " << t_ibuf_reads << std::endl;
        std::cout << "    CIM Computes: " << t_cim_computes << std::endl;
        std::cout << "    OBUF Writes: " << t_obuf_writes << std::endl;
        std::cout << "    Stores: " << t_stores << std::endl;
#endif

        return TilingConfig(
            tiling["output_tile_p"], tiling["output_tile_q"], tiling["input_tile_h"],
            tiling["input_tile_w"], tiling["input_tile_p"], tiling["input_tile_q"],
            tiling["num_output_tiles_p"], tiling["num_output_tiles_q"], tiling["num_input_tiles_p"],
            tiling["num_input_tiles_q"], tiling["output_tile_count"], tiling["input_tile_count"],
            tiling["case_type"], t_loads, t_ibuf_reads, t_cim_computes, t_obuf_writes, t_stores);
    } catch (const json::exception& e) {
        throw ConfigurationError("Invalid tiling configuration: " + std::string(e.what()));
    }
}

CNNParams ConfigurationLoader::load_cnn_params_from_network(const json& network_config,
                                                            int layer_index,
                                                            const std::string& layer_id) {
    try {
        if (!network_config.contains("layers") || !network_config["layers"].is_array()) {
            throw ConfigurationError("Network configuration missing 'layers' array");
        }

        const json* layer_params = nullptr;

        // Try to find by layer_id first
        if (!layer_id.empty()) {
            for (const auto& layer : network_config["layers"]) {
                if (layer.contains("name") && layer["name"] == layer_id) {
                    layer_params = &layer["params"];
#if ENABLE_VERBOSE_LOGS
                    std::cout << "[CONFIG_LOADER] Found layer by ID: " << layer_id << std::endl;
#endif
                    break;
                }
            }
        }

        // Fallback to layer_index
        if (layer_params == nullptr) {
            if (layer_index >= 0 &&
                static_cast<size_t>(layer_index) < network_config["layers"].size()) {
                layer_params = &network_config["layers"][layer_index]["params"];
#if ENABLE_VERBOSE_LOGS
                std::cout << "[CONFIG_LOADER] Using layer at index: " << layer_index << std::endl;
#endif
            } else {
                throw ConfigurationError("Layer index " + std::to_string(layer_index) +
                                         " out of range");
            }
        }

        // Create merged config for load_cnn_params
        json merged_config;
        merged_config["cnn_layer"] = *layer_params;

        // Add network-level batch_size if not in layer params
        if (!merged_config["cnn_layer"].contains("batch_size") &&
            network_config.contains("batch_size")) {
            merged_config["cnn_layer"]["batch_size"] = network_config["batch_size"];
        }

        return load_cnn_params(merged_config);
    } catch (const json::exception& e) {
        throw ConfigurationError("Invalid network configuration: " + std::string(e.what()));
    }
}

HWConfig ConfigurationLoader::load_hw_config_from_hardware(const json& hardware_config) {
    try {
        // Require hierarchical format: hardware_config["hardware"]["ibuf"]
        if (!hardware_config.contains("hardware")) {
            throw ConfigurationError(
                "Hardware configuration must use hierarchical format.\n"
                "Expected: hardware_config['hardware']['ibuf'/'obuf'/'external'/'cim']\n"
                "Legacy flat format no longer supported.");
        }

        // Create merged config for load_hw_config
        json merged_config;
        merged_config["hardware"] = hardware_config["hardware"];

        return load_hw_config(merged_config);
    } catch (const json::exception& e) {
        throw ConfigurationError("Invalid hardware configuration: " + std::string(e.what()));
    }
}

}  // namespace config
