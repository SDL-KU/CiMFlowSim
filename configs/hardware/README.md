# Hardware Configuration

Hardware configurations define CIM accelerator parameters for simulation.

## Directory Structure

```
hardware/
├── active/     # Configs used by ./efsim benchmark --all
└── archived/   # Unused configs (not included in benchmarks)
```

## Configuration Format

```json
{
  "description": "Hardware description",
  "hardware": {
    "cim": {
      "compute_time_ns": 18.3,
      "macro_config": {
        "input_size": 512,
        "output_size": 128,
        "macro_area_um2": 51650.0
      }
    },
    "ibuf": {
      "timing": {
        "clk_frequency_mhz": 200.0,
        "base_latency_cycles": 1
      },
      "architecture": {
        "num_banks": 1,
        "bits_per_line": 64
      },
      "ports": {
        "num_read_ports": 0,
        "num_write_ports": 0,
        "num_rw_ports": 1
      }
    },
    "obuf": {
      "timing": { ... },
      "architecture": { ... },
      "ports": { ... }
    },
    "external": {
      "timing": {
        "clk_frequency_mhz": 1600.0,
        "base_latency_cycles": 192
      },
      "architecture": {
        "num_banks": 1,
        "bits_per_line": 32
      },
      "ports": { ... }
    },
    "technology": {
      "node": "22nm",
      "sram_bit_area_um2": 0.5
    }
  },
  "energy": {
    "computation": {
      "mac_energy": 0.0691,
      "pooling_energy": 0
    },
    "memory": {
      "sram_read_energy_per_bit": 0.1625,
      "sram_write_energy_per_bit": 0.1875,
      "dram_read_energy_per_bit": 10.0,
      "dram_write_energy_per_bit": 10.0
    },
    "communication": {
      "on_chip_wire_energy": 0.003
    },
    "static": {
      "static_power_mw": 0.006
    }
  }
}
```

## Key Parameters

### CIM (Compute-in-Memory)

| Parameter | Description | Unit |
|-----------|-------------|------|
| `compute_time_ns` | Single CIM operation latency | ns |
| `macro_config.input_size` | CIM macro input dimension | - |
| `macro_config.output_size` | CIM macro output dimension | - |
| `macro_config.macro_area_um2` | CIM macro area | um^2 |

### Buffer Timing

| Parameter | Description | Unit |
|-----------|-------------|------|
| `clk_frequency_mhz` | Buffer clock frequency | MHz |
| `base_latency_cycles` | Access latency in cycles | cycles |

Actual latency: `base_latency_cycles / clk_frequency_mhz * 1000` ns

### Buffer Architecture

| Parameter | Description |
|-----------|-------------|
| `num_banks` | Number of memory banks |
| `bits_per_line` | Bits per cache line |
| `num_rw_ports` | Number of read/write ports |

### Energy Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `mac_energy` | Energy per MAC operation | pJ |
| `sram_*_energy_per_bit` | SRAM access energy | pJ/bit |
| `dram_*_energy_per_bit` | DRAM access energy | pJ/bit |

## Usage

```bash
# Use specific hardware config
./efsim benchmark --hardware isscc_2021_22nm_reram --networks vgg11_4layers

# All active configs
./efsim benchmark --all
```
