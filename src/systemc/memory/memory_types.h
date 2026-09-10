#ifndef MEMORY_TYPES_H
#define MEMORY_TYPES_H

#include <systemc.h>

#include <map>
#include <vector>

// ============================================================================
// Memory port configuration
// ============================================================================
enum class PortType {
    READ_ONLY,   // Port can only read
    WRITE_ONLY,  // Port can only write
    READ_WRITE   // Port can both read and write
};

// Generalized port configuration
struct PortConfiguration {
    int num_read_ports;   // Number of read-only ports
    int num_write_ports;  // Number of write-only ports
    int num_rw_ports;     // Number of read/write ports

    PortConfiguration(int r = 0, int w = 0, int rw = 1)
        : num_read_ports(r), num_write_ports(w), num_rw_ports(rw) {}

    int total_ports() const { return num_read_ports + num_write_ports + num_rw_ports; }

    // Factory method for common configuration
    static PortConfiguration single_port() { return PortConfiguration(0, 0, 1); }
};

// ============================================================================
// Memory timing configuration
// ============================================================================
struct MemoryTimingConfig {
    // SRAM buffer timing (clock cycle-based)
    int base_latency_cycles;   // Base latency in clock cycles
    double clk_frequency_mhz;  // Memory clock frequency in MHz

    MemoryTimingConfig(int base_cycles, double freq_mhz)
        : base_latency_cycles(base_cycles), clk_frequency_mhz(freq_mhz) {}

    // Helper function to calculate ns per clock cycle
    double ns_per_clock_cycle() const {
        return 1000.0 / clk_frequency_mhz;  // Convert MHz to ns per cycle
    }

    // Calculate base latency in nanoseconds
    double base_latency_ns() const { return base_latency_cycles * ns_per_clock_cycle(); }
};

// ============================================================================
// Bank/Line configuration
// ============================================================================
struct BankLineConfig {
    int num_banks;      // Number of memory banks
    int bits_per_line;  // Bits per line

    BankLineConfig(int banks, int bits) : num_banks(banks), bits_per_line(bits) {}

    // Note: No line limits - dynamic allocation tracked via usage stats
};

// ============================================================================
// Memory access statistics
// ============================================================================
struct MemoryAccessPattern {
    int total_reads = 0;
    int total_writes = 0;
    int total_read_lines = 0;   // Total lines accessed across all banks (for energy)
    int total_write_lines = 0;  // Total lines accessed across all banks (for energy)
    double total_read_time_ns = 0.0;
    double total_write_time_ns = 0.0;
    double avg_read_latency_ns = 0.0;
    double avg_write_latency_ns = 0.0;
    int bank_conflicts = 0;

    void update_read(double bytes, double latency_ns) {
        (void)bytes;  // Parameter reserved for future use
        total_reads++;
        total_read_time_ns += latency_ns;
        avg_read_latency_ns = total_read_time_ns / total_reads;
    }

    void update_write(double bytes, double latency_ns) {
        (void)bytes;  // Parameter reserved for future use
        total_writes++;
        total_write_time_ns += latency_ns;
        avg_write_latency_ns = total_write_time_ns / total_writes;
    }

    void record_bank_conflict() { bank_conflicts++; }
};

#endif  // MEMORY_TYPES_H