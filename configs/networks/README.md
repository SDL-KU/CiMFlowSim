# Network Configuration

Network configurations define CNN layer parameters for simulation.

## Directory Structure

```
networks/
├── active/     # Configs used by ./efsim benchmark --all
└── archived/   # Unused configs (not included in benchmarks)
```

## Configuration Format

```json
{
  "network_name": "LeNet-5",
  "description": "LeNet-5 CNN for MNIST digit recognition",
  "batch_size": 64,
  "layers": [
    {
      "name": "conv1",
      "type": "conv2d",
      "params": {
        "H": 32,
        "W": 32,
        "C": 1,
        "R": 5,
        "S": 5,
        "M": 6,
        "stride": 1,
        "input_bitwidth": 4,
        "output_bitwidth": 4,
        "P": 28,
        "Q": 28,
        "P_pooled": 14,
        "Q_pooled": 14,
        "pool_height": 2,
        "pool_width": 2
      }
    }
  ]
}
```

## Network-level Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `network_name` | Network identifier | "LeNet-5" |
| `description` | Human-readable description | "LeNet-5 CNN..." |
| `batch_size` | Global batch size | 64 |

## Layer Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `H`, `W` | Input feature map height/width | 32, 32 |
| `C` | Input channels | 1 |
| `R`, `S` | Kernel height/width | 5, 5 |
| `M` | Output channels (filters) | 6 |
| `stride` | Convolution stride | 1 |
| `input_bitwidth` | Input data precision | 4 |
| `output_bitwidth` | Output data precision | 4 |
| `P`, `Q` | Output feature map height/width | 28, 28 |
| `P_pooled`, `Q_pooled` | Pooled output size | 14, 14 |
| `pool_height`, `pool_width` | Pooling kernel size | 2, 2 |

## Output Size Calculation

```
P = (H - R) / stride + 1
Q = (W - S) / stride + 1
```

With pooling:
```
P_pooled = P / pool_height
Q_pooled = Q / pool_width
```

## Usage

```bash
# Use specific network config
./efsim benchmark --hardware isscc_2021_22nm_reram --networks vgg11_4layers

# All active configs
./efsim benchmark --all
```

## Supported Layer Types

- `conv2d`: Standard 2D convolution with optional pooling
