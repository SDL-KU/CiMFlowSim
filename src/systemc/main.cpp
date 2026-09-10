/**
 * @file main.cpp
 * @brief Main entry point for CiMFlowSim CNN accelerator simulator
 *
 * This file contains the main simulation controller that orchestrates:
 * - Command line parsing and configuration loading
 * - Parameter validation for CNN layers and hardware
 * - SystemC simulation execution with selected strategy
 * - Output generation (Gantt charts, statistics)
 */

#include <systemc.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "config/config_loader.h"
#include "output/gantt_generator.h"
#include "pipeline_simulator.h"

using namespace config;
using namespace output;

/**
 * @class StreamRedirectGuard
 * @brief RAII guard for stdout/stderr redirection to log file
 *
 * Automatically restores original stream buffers when destroyed,
 * ensuring proper cleanup even when exceptions occur.
 */
class StreamRedirectGuard {
  public:
    StreamRedirectGuard() : original_cout_(std::cout.rdbuf()), original_cerr_(std::cerr.rdbuf()) {}

    ~StreamRedirectGuard() { restore(); }

    // Non-copyable, non-movable
    StreamRedirectGuard(const StreamRedirectGuard&) = delete;
    StreamRedirectGuard& operator=(const StreamRedirectGuard&) = delete;

    /**
     * @brief Redirect stdout/stderr to a log file
     * @param log_path Path to the log file
     * @return true if successful, false if file could not be opened
     */
    bool redirect_to_file(const std::string& log_path) {
        log_file_ = std::make_unique<std::ofstream>(log_path);
        if (!log_file_->is_open()) {
            log_file_.reset();
            return false;
        }
        std::cout.rdbuf(log_file_->rdbuf());
        std::cerr.rdbuf(log_file_->rdbuf());
        return true;
    }

    /**
     * @brief Restore original stdout/stderr buffers
     */
    void restore() {
        if (original_cout_) {
            std::cout.rdbuf(original_cout_);
            original_cout_ = nullptr;
        }
        if (original_cerr_) {
            std::cerr.rdbuf(original_cerr_);
            original_cerr_ = nullptr;
        }
        if (log_file_) {
            log_file_->close();
            log_file_.reset();
        }
    }

  private:
    std::streambuf* original_cout_;
    std::streambuf* original_cerr_;
    std::unique_ptr<std::ofstream> log_file_;
};

/**
 * @class SimulationController
 * @brief Main controller class for CNN accelerator simulation
 *
 * Manages the complete simulation workflow:
 * 1. Configuration Loading: Parse JSON config with CNN/hardware parameters
 * 2. Parameter Validation: Check mathematical correctness and strategy compatibility
 * 3. SystemC Simulation: Execute pipeline simulation with selected strategy
 * 4. Output Generation: Create analysis files and performance reports
 */
class SimulationController {
  public:
    SimulationController()
        : output_dir("."),
          save_simulation_log(false),
          save_gantt_data(false),
          save_execution_trace(false),
          save_dependency_graph(false) {}

    /**
     * @brief Main entry point for simulation execution
     * @param argc Command line argument count
     * @param argv Command line arguments
     * @return Exit code (0 = success, non-zero = error)
     */
    int run(int argc, char* argv[]) {
        // RAII guard for stdout/stderr redirection - automatically restores on scope exit
        StreamRedirectGuard stream_guard;

        try {
            // Parse command line and load configuration files (strategy, network, hardware)
            auto [strategy_path, network_path, hardware_path] = parse_command_line(argc, argv);

            // Only redirect to log file if --save-simulation-log was specified
            if (save_simulation_log) {
                std::string log_path =
                    output_dir + "/" + std::string(validation_constants::SIMULATION_LOG_FILENAME);
                if (!stream_guard.redirect_to_file(log_path)) {
                    std::cerr << "Warning: Could not open log file: " << log_path << std::endl;
                }
            }

            // Load 3 separate configuration files
            INFO_LOG("Loading strategy: " << strategy_path);
            INFO_LOG("Loading network: " << network_path);
            INFO_LOG("Loading hardware: " << hardware_path);

            json strategy_config = ConfigurationLoader::load_config_file(strategy_path);
            json network_config = ConfigurationLoader::load_config_file(network_path);
            json hardware_config = ConfigurationLoader::load_config_file(hardware_path);

            // Extract layer_index and layer_id from strategy
            int layer_index = strategy_config.value("layer_index", 0);
            std::string layer_id = strategy_config.value("layer_id", "");

            // Extract simulation parameters from separate configs
            CNNParams params = ConfigurationLoader::load_cnn_params_from_network(
                network_config, layer_index, layer_id);
            HWConfig hw_config = ConfigurationLoader::load_hw_config_from_hardware(hardware_config);

            // Load Independent Tiling configuration (JSON-based approach)
            TilingConfig tiling = ConfigurationLoader::load_tiling_config(strategy_config);
            log_tiling_configuration(params, hw_config, tiling);

            // Execute SystemC simulation with Independent Tiling
            return run_tiling_simulation(params, hw_config, tiling);
            // stream_guard destructor automatically restores stdout/stderr

        } catch (const config::ConfigurationError& e) {
            std::cerr << e.what() << std::endl;
            stream_guard.restore();  // Restore before printing to console
            std::cerr << e.what() << std::endl;
            return 1;
        } catch (const config::ValidationError& e) {
            std::cerr << e.what() << std::endl;
            stream_guard.restore();
            std::cerr << e.what() << std::endl;
            return 1;
        } catch (const std::exception& e) {
            std::cerr << "Unexpected error: " << e.what() << std::endl;
            stream_guard.restore();
            std::cerr << "Unexpected error: " << e.what() << std::endl;
            return 1;
        }
    }

  private:
    std::string output_dir;

    // Granular log control flags
    bool save_simulation_log;     // simulation_log.txt (~100MB)
    bool save_gantt_data;          // gantt_data.txt (~44MB)
    bool save_execution_trace;     // execution_trace.log (~38MB)
    bool save_dependency_graph;    // dependency_graph.txt (~31MB)

    std::tuple<std::string, std::string, std::string> parse_command_line(int argc, char* argv[]) {
        if (argc < 4) {
            print_usage(argv[0]);
            throw ConfigurationError("Invalid command line arguments");
        }

        std::string strategy_path = std::string(argv[1]);
        std::string network_path = std::string(argv[2]);
        std::string hardware_path = std::string(argv[3]);

        // Initialize default values (all logs disabled by default)
        save_simulation_log = false;
        save_gantt_data = false;
        save_execution_trace = false;
        save_dependency_graph = false;

        // Parse optional arguments
        for (int i = 4; i < argc; i++) {
            std::string arg = std::string(argv[i]);
            if (arg == "-o" || arg == "--output-dir") {
                if (i + 1 < argc) {
                    output_dir = std::string(argv[i + 1]);
                    i++;  // Skip next argument as it's the directory path
                } else {
                    throw ConfigurationError("Output directory option requires a path");
                }
            } else if (arg == "--save-simulation-log") {
                save_simulation_log = true;
            } else if (arg == "--save-gantt-data") {
                save_gantt_data = true;
            } else if (arg == "--save-execution-trace") {
                save_execution_trace = true;
            } else if (arg == "--save-dependency-graph") {
                save_dependency_graph = true;
            }
        }

        // Create output directory if it doesn't exist
        if (output_dir != ".") {
            std::filesystem::create_directories(output_dir);
        }

        return {strategy_path, network_path, hardware_path};
    }

    void print_usage(const char* program_name) {
        std::cerr << "Usage: " << program_name
                  << " <strategy.json> <network.json> <hardware.json> [OPTIONS]"
                  << std::endl;
        std::cerr << "\nRequired arguments:" << std::endl;
        std::cerr << "  strategy.json   Strategy configuration (tiling parameters, operation counts)" << std::endl;
        std::cerr << "  network.json    Network configuration (CNN layers, dimensions)" << std::endl;
        std::cerr << "  hardware.json   Hardware configuration (compute units, memory, timing)" << std::endl;
        std::cerr << "\nOptions:" << std::endl;
        std::cerr << "  -o, --output-dir <dir>     Output directory (default: current directory)" << std::endl;
        std::cerr << "  --save-simulation-log      Save simulation_log.txt (~100MB, stdout/stderr)" << std::endl;
        std::cerr << "  --save-gantt-data          Save gantt_data.txt (~44MB, timeline data)" << std::endl;
        std::cerr << "  --save-execution-trace     Save execution_trace.log (~38MB, operation details)" << std::endl;
        std::cerr << "  --save-dependency-graph    Save dependency_graph.txt (~31MB, dependencies)" << std::endl;
        std::cerr << "\nNote: Without log flags, only lightweight JSON files are saved (~3KB)" << std::endl;
    }

    void log_tiling_configuration([[maybe_unused]] const CNNParams& params,
                                  [[maybe_unused]] const HWConfig& hw_config,
                                  [[maybe_unused]] const TilingConfig& tiling) {
        INFO_LOG("=== SystemC Pipeline Simulator ===");
        INFO_LOG("Strategy: Independent Tiling");
        INFO_LOG("Configuration: Compute="
                 << hw_config.compute_time_ns << "ns, "
                 << "IBUF=" << hw_config.ibuf_config.base_latency_ns() << "ns@"
                 << hw_config.ibuf_config.clk_frequency_mhz << "MHz, "
                 << "OBUF=" << hw_config.obuf_config.base_latency_ns() << "ns@"
                 << hw_config.obuf_config.clk_frequency_mhz << "MHz, "
                 << "External=" << hw_config.external_config.base_latency_ns() << "ns@"
                 << hw_config.external_config.clk_frequency_mhz << "MHz");
        INFO_LOG("CNN Parameters: H=" << params.H << ", W=" << params.W << ", C=" << params.C
                                      << ", R=" << params.R << ", S=" << params.S
                                      << ", M=" << params.M);
        INFO_LOG("Pooling Parameters: pool_height=" << params.pool_height
                                                    << ", pool_width=" << params.pool_width);
        INFO_LOG("Output dimensions: P=" << params.P << ", Q=" << params.Q << " → P_pooled="
                                         << params.P_pooled << ", Q_pooled=" << params.Q_pooled);
        INFO_LOG("");
        INFO_LOG("Tiling Configuration:");
        INFO_LOG("  Output tile: " << tiling.output_tile_p << "×" << tiling.output_tile_q);
        INFO_LOG("  Input tile: " << tiling.input_tile_h << "×" << tiling.input_tile_w);
        INFO_LOG("  Number of tiles: " << tiling.num_output_tiles_p << "×"
                                       << tiling.num_output_tiles_q << " = "
                                       << tiling.output_tile_count);
        INFO_LOG("");

        INFO_LOG("Tensor shapes:");
        INFO_LOG("  input_tensor: [" << params.batch_size << ", " << params.C << ", " << params.H
                                     << ", " << params.W << "]");
        INFO_LOG("  output_tensor: [" << params.batch_size << ", " << params.M << ", "
                                      << params.P_pooled << ", " << params.Q_pooled
                                      << "] (after pooling)");
        INFO_LOG("");
    }

    /**
     * @brief Execute the SystemC simulation with Independent Tiling
     * @param params CNN layer parameters (dimensions, batch size, etc.)
     * @param hw_config Hardware configuration (timing, buffers, etc.)
     * @param tiling Independent Tiling configuration
     * @return Exit code (0 = success)
     */
    int run_tiling_simulation(const CNNParams& params, const HWConfig& hw_config,
                              const TilingConfig& tiling) {
        // Create simulator instance with Independent Tiling strategy
        // Pass save_execution_trace flag to control execution trace file
        PipelineSimulator simulator("PipelineSim", params, hw_config, tiling, output_dir, save_execution_trace);

        // Start SystemC simulation
        sc_start();

        // Generate Gantt chart data if requested
        if (save_gantt_data) {
            std::string gantt_data_path =
                output_dir + "/" + std::string(validation_constants::GANTT_DATA_FILENAME);
            auto timeline = simulator.get_timeline();
            GanttGenerator::create_gantt_data(timeline, gantt_data_path);
        }

        // Write dependency graph if requested
        if (save_dependency_graph) {
            std::string dependency_path =
                output_dir + "/" + std::string(validation_constants::DEPENDENCY_GRAPH_FILENAME);
            simulator.write_dependencies_to_file(dependency_path);
        }

        // Print performance statistics (always needed for JSON output)
        simulator.print_statistics();

        return 0;
    }
};

int sc_main(int argc, char* argv[]) {
    SimulationController controller;
    return controller.run(argc, argv);
}
