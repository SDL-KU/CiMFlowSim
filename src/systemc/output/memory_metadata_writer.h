/**
 * @file memory_metadata_writer.h
 * @brief Lightweight memory metadata output for visualization
 *
 * Generates a small JSON file (~5-10KB) containing memory layout information
 * needed for visualization, eliminating the need for 3MB+ simulation logs.
 *
 * Output: memory_metadata.json
 * Contents:
 *   - Tensor information (shape, lines, groups_per_line)
 *   - Memory access patterns (max_lines_per_bank histograms)
 *   - Hardware configuration (bank counts)
 */

#ifndef MEMORY_METADATA_WRITER_H
#define MEMORY_METADATA_WRITER_H

#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace output {

/**
 * @struct TensorMetadata
 * @brief Metadata for a single tensor's memory layout
 */
struct TensorMetadata {
    std::string name;
    std::vector<int> shape;
    int total_lines;
    int groups_per_line;  // Can be negative for multi-line groups
    int line_bits;
    int element_bits;

    std::string to_json() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "      \"name\": \"" << name << "\",\n";
        ss << "      \"shape\": [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << shape[i];
        }
        ss << "],\n";
        ss << "      \"total_lines\": " << total_lines << ",\n";
        ss << "      \"groups_per_line\": " << groups_per_line << ",\n";
        ss << "      \"line_bits\": " << line_bits << ",\n";
        ss << "      \"element_bits\": " << element_bits << "\n";
        ss << "    }";
        return ss.str();
    }
};

/**
 * @struct AccessPatternMetadata
 * @brief Metadata for memory access patterns
 */
struct AccessPatternMetadata {
    std::string memory_name;  // "external", "ibuf", "obuf"
    std::string operation;    // "read", "write"
    std::map<int, int> max_lines_per_bank_histogram;  // value -> count

    void add_access(int max_lines_per_bank) {
        max_lines_per_bank_histogram[max_lines_per_bank]++;
    }

    std::string to_json() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "      \"memory\": \"" << memory_name << "\",\n";
        ss << "      \"operation\": \"" << operation << "\",\n";
        ss << "      \"total_accesses\": " << get_total_accesses() << ",\n";
        ss << "      \"max_lines_per_bank_histogram\": {";
        bool first = true;
        for (const auto& [value, count] : max_lines_per_bank_histogram) {
            if (!first) ss << ", ";
            ss << "\"" << value << "\": " << count;
            first = false;
        }
        ss << "}\n";
        ss << "    }";
        return ss.str();
    }

    int get_total_accesses() const {
        int total = 0;
        for (const auto& [_, count] : max_lines_per_bank_histogram) {
            total += count;
        }
        return total;
    }
};

/**
 * @class MemoryMetadataWriter
 * @brief Collects and writes memory metadata for visualization
 */
class MemoryMetadataWriter {
  public:
    MemoryMetadataWriter() = default;

    /**
     * @brief Add tensor metadata
     */
    void add_tensor(const std::string& memory_type, const TensorMetadata& tensor) {
        tensors_[memory_type] = tensor;
    }

    /**
     * @brief Record a memory access for pattern analysis
     */
    void record_access(const std::string& memory_name, const std::string& operation,
                       int max_lines_per_bank) {
        std::string key = memory_name + "_" + operation;
        if (access_patterns_.find(key) == access_patterns_.end()) {
            access_patterns_[key] = AccessPatternMetadata{memory_name, operation, {}};
        }
        access_patterns_[key].add_access(max_lines_per_bank);
    }

    /**
     * @brief Set hardware configuration
     */
    void set_hardware_config(int external_banks, int external_bits_per_line,
                             int ibuf_banks, int ibuf_bits_per_line,
                             int obuf_banks, int obuf_bits_per_line) {
        external_banks_ = external_banks;
        external_bits_per_line_ = external_bits_per_line;
        ibuf_banks_ = ibuf_banks;
        ibuf_bits_per_line_ = ibuf_bits_per_line;
        obuf_banks_ = obuf_banks;
        obuf_bits_per_line_ = obuf_bits_per_line;
    }

    /**
     * @brief Write all metadata to JSON file
     */
    void write_to_file(const std::string& output_path) const {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not create memory_metadata.json" << std::endl;
            return;
        }

        file << "{\n";

        // Hardware configuration
        file << "  \"hardware\": {\n";
        file << "    \"external\": {\"num_banks\": " << external_banks_
             << ", \"bits_per_line\": " << external_bits_per_line_ << "},\n";
        file << "    \"ibuf\": {\"num_banks\": " << ibuf_banks_
             << ", \"bits_per_line\": " << ibuf_bits_per_line_ << "},\n";
        file << "    \"obuf\": {\"num_banks\": " << obuf_banks_
             << ", \"bits_per_line\": " << obuf_bits_per_line_ << "}\n";
        file << "  },\n";

        // Tensors
        file << "  \"tensors\": {\n";
        bool first_tensor = true;
        for (const auto& [key, tensor] : tensors_) {
            if (!first_tensor) file << ",\n";
            file << "    \"" << key << "\": " << tensor.to_json();
            first_tensor = false;
        }
        file << "\n  },\n";

        // Access patterns
        file << "  \"access_patterns\": [\n";
        bool first_pattern = true;
        for (const auto& [key, pattern] : access_patterns_) {
            if (!first_pattern) file << ",\n";
            file << "    " << pattern.to_json();
            first_pattern = false;
        }
        file << "\n  ]\n";

        file << "}\n";
        file.close();

        std::cout << "[OK] Memory metadata written to: " << output_path << std::endl;
    }

  private:
    std::map<std::string, TensorMetadata> tensors_;
    std::map<std::string, AccessPatternMetadata> access_patterns_;

    int external_banks_ = 8;
    int external_bits_per_line_ = 32;
    int ibuf_banks_ = 4;
    int ibuf_bits_per_line_ = 128;
    int obuf_banks_ = 4;
    int obuf_bits_per_line_ = 128;
};

}  // namespace output

#endif  // MEMORY_METADATA_WRITER_H
