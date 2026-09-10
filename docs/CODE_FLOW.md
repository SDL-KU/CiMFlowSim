# Code Flow Guide

엔트리포인트부터 실제 실행까지의 코드 흐름 추적.

## 1. CLI 엔트리포인트

```
./efsim <command> [args]
    │
    └─→ efsim (Python script)
            │
            ├─→ benchmark     → tools/dse_benchmark.py
            ├─→ simulate      → tools/simulate.py
            ├─→ plot          → tools/plot.py
            ├─→ sweep         → tools/sweep.py
            ├─→ sweep-analyze → tools/sweep_analysis.py
            ├─→ plot-sweep    → efsim::plot_sweep_command() (inline)
            ├─→ score         → tools/score.py
            └─→ pattern       → tools/pattern.py
```

---

## 2. Benchmark 워크플로우 (가장 일반적)

```bash
./efsim benchmark --hardware isscc_2021_22nm_reram --networks vgg11_4layers
```

### 2.1 전체 흐름

```
tools/dse_benchmark.py::main()
    │
    ├─[1] 워크스페이스 생성
    │     └─→ WorkspaceManager.create_workspace()
    │             └─→ workspaces/{name}/ 디렉토리 생성
    │             └─→ hardware_config.json, network_config.json 복사
    │
    ├─[2] 전략 생성
    │     └─→ _generate_strategies()
    │             └─→ IndependentTilingGenerator.generate_all_strategies()
    │                     └─→ strategies/L{layer}_S{id}_out{P}x{Q}_in{P}x{Q}.json
    │
    ├─[3] 시뮬레이션 실행
    │     └─→ _run_simulation()
    │             └─→ subprocess: tools/simulate.py
    │
    └─[4] 플롯 생성
          └─→ _generate_plots()
                  └─→ subprocess: tools/plot.py --progressive
```

### 2.2 전략 생성 상세

```
IndependentTilingGenerator (src/python/core/tiling_generator.py)
    │
    ├─→ __init__(layer_params, hardware_config)
    │       └─→ CNNLayerParams 파싱
    │       └─→ 하드웨어 제약 로드
    │
    └─→ generate_all_strategies()
            │
            ├─→ _generate_case1_strategies()
            │       └─→ Case 1 (Sub-tiling): 작은 입력 → 큰 출력
            │
            ├─→ _generate_case2_strategies()
            │       └─→ Case 2 (Super-tiling): 큰 입력 → 작은 출력들
            │
            └─→ _create_strategy_descriptor(output_tile, input_tile)
                    │
                    ├─→ StrategyDescriptor 생성
                    │       └─→ num_output_tiles, num_input_tiles 계산
                    │       └─→ sub-tiling / super-tiling 케이스 결정
                    │
                    └─→ JSON 파일 저장
                            └─→ strategies/L{layer}_S{id}_out{P}x{Q}_in{P}x{Q}.json
```

---

## 3. Simulate 워크플로우

```bash
./efsim simulate my_workspace [--resume]
```

### 3.1 전체 흐름

```
tools/simulate.py::main()
    │
    ├─→ _load_pending_strategies()
    │       └─→ strategies/ 디렉토리 스캔
    │       └─→ DB에서 완료된 전략 제외 (--resume 시)
    │
    ├─→ ray.init(address=RAY_HEAD_ADDRESS)
    │       └─→ 클러스터 연결
    │
    └─→ _run_parallel_simulations()
            │
            ├─→ Ray Task 생성 (SPREAD 스케줄링)
            │       └─→ @ray.remote _simulate_single_strategy()
            │
            └─→ 결과 수집 및 DB 저장
                    └─→ PerformanceDatabase.insert_strategy_result()
```

### 3.2 단일 전략 시뮬레이션 (Ray Worker)

```
_simulate_single_strategy() [Ray Worker에서 실행]
    │
    ├─[1] SystemC 실행
    │     └─→ SystemCRunner.run_simulation(strategy_json)
    │             │
    │             ├─→ subprocess: ./src/systemc/pipeline_sim
    │             │       └─→ stdin: strategy JSON
    │             │       └─→ stdout: simulation 결과
    │             │
    │             └─→ 출력 파일 생성
    │                     └─→ simulations/L{layer}_S{id}/simulation_statistics.json
    │                     └─→ simulations/L{layer}_S{id}/memory_metadata.json
    │
    ├─[2] 결과 파싱
    │     └─→ SystemCParser.parse_statistics()
    │             └─→ latency_ns, operation counts 추출
    │
    ├─[3] 에너지 계산
    │     └─→ EnergyCalculator.calculate()
    │             │
    │             ├─→ MAC 에너지 = mac_count × mac_energy
    │             ├─→ SRAM 에너지 = sram_reads × sram_read_energy + ...
    │             └─→ DRAM 에너지 = dram_reads × dram_read_energy + ...
    │
    ├─[4] 면적 계산
    │     └─→ AreaCalculator.calculate()
    │             │
    │             ├─→ ibuf_area = input_tile_size × sram_area_per_bit
    │             ├─→ obuf_area = output_tile_size × sram_area_per_bit
    │             └─→ cim_area = cim_rows × cim_cols × cim_area_per_cell
    │
    └─[5] 결과 반환
          └─→ ray_results/L{layer}_S{id}.json
                  └─→ {latency_ns, energy_nj, area_mm2, ...}
```

---

## 4. Plot 워크플로우

```bash
./efsim plot my_workspace [--progressive]
```

### 4.1 전체 흐름

```
tools/plot.py::main()
    │
    └─→ generate_all_plots() [src/python/visualization/plotting/generate_all_plots.py]
            │
            ├─[1] 네트워크 레벨 Pareto
            │     └─→ _generate_network_pareto()
            │
            ├─[2] 비교 플롯
            │     └─→ _generate_comparison_plots()
            │
            ├─[3] Progressive 레이어 (--progressive)
            │     └─→ _generate_progressive_plots()
            │
            └─[4] 패턴 분석
                  └─→ _generate_pattern_analysis()
```

### 4.2 네트워크 Pareto 생성 상세

```
_generate_network_pareto()
    │
    ├─[1] DB에서 전략 로딩
    │     └─→ pareto_utils.load_all_layer_strategies(db_path)
    │             │
    │             └─→ SQL Query:
    │                     SELECT strategy_id, latency_ns, energy_nj,
    │                            ibuf_area_mm2, obuf_area_mm2, cim_area_mm2,
    │                            output_tile_p, output_tile_q,
    │                            input_tile_p, input_tile_q
    │                     FROM strategy_results
    │                     WHERE layer_index = ?
    │
    ├─[2] Monte Carlo 샘플링
    │     └─→ pareto_sampling.compute_network_pareto()
    │             │
    │             ├─→ _broadcast_db_to_nodes()
    │             │       └─→ strategies.db를 tar.gz로 압축
    │             │       └─→ 모든 Ray 노드에 배포
    │             │
    │             ├─→ Ray Tasks 생성
    │             │       └─→ @ray.remote _ray_sample_worker()
    │             │               │
    │             │               ├─→ 로컬 DB에서 전략 로드
    │             │               ├─→ 랜덤 조합 샘플링 (20M samples)
    │             │               ├─→ 네트워크 메트릭 계산
    │             │               │       └─→ _worker_compute_network_metrics()
    │             │               ├─→ 로컬 Pareto front 계산
    │             │               │       └─→ _worker_local_pareto_front()
    │             │               └─→ 결과 저장
    │             │                       └─→ /mnt/workers/{hostname}/pareto_results/
    │             │
    │             └─→ 결과 병합
    │                     └─→ 전역 Pareto front 계산
    │
    └─[3] 플롯 생성
          └─→ pareto_grid_plot.plot_pareto_grid()
                  │
                  ├─→ 3x3 그리드 생성 (7 objectives + legend)
                  │       └─→ SUBPLOT_POSITIONS_7 = [0, 3, 4, 5, 6, 7, 8]
                  │
                  ├─→ 각 subplot에 scatter + Pareto line
                  │       └─→ SCATTER_BACKGROUND, SCATTER_PARETO 스타일 적용
                  │
                  └─→ 저장
                          └─→ plots/network_pareto.png
```

### 4.3 네트워크 메트릭 계산

```python
# pareto_sampling.py::_worker_compute_network_metrics()

def compute(strategy_combination: Dict[layer_index, strategy]) -> Dict:
    # 레이턴시: 모든 레이어 합산
    total_latency = sum(s["latency_ns"] for s in strategies)

    # 에너지: 모든 레이어 합산
    total_energy = sum(s["energy_nj"] for s in strategies)

    # 버퍼 면적: 레이어 간 최대값 (버퍼 재사용)
    buffer_area = max(ibuf_area) + max(obuf_area)

    # CIM 면적: 두 가지 방식
    sum_cim = sum(cim_area)   # 모든 레이어 CIM 합
    peak_cim = max(cim_area)  # 최대 CIM (재사용 시)

    # 총 면적
    sum_area = buffer_area + sum_cim
    peak_area = buffer_area + peak_cim

    # EAP (Energy-Area Product)
    buffer_eap = buffer_area × total_energy
    sum_eap = sum_area × total_energy
    peak_eap = peak_area × total_energy

    return {latency, energy, buffer_area, sum_area, peak_area,
            buffer_eap, sum_eap, peak_eap, combination}
```

---

## 5. 비교 플롯 워크플로우

### 5.1 All vs Coupled vs Legacy 비교

```
plot_network_pareto_comparison.py::generate_comparison_plots()
    │
    ├─[1] 세 가지 전략 세트 로딩
    │     │
    │     ├─→ All: load_all_layer_strategies()
    │     │       └─→ 모든 전략 (input ≠ output 타일 포함)
    │     │
    │     ├─→ Coupled: load_coupled_strategies()
    │     │       └─→ WHERE input_tile_p = output_tile_p
    │     │               AND input_tile_q = output_tile_q
    │     │
    │     └─→ Legacy: load_legacy_strategies()
    │             └─→ INNER JOIN legacy_mappings
    │                     └─→ LS1-LS5 전략만
    │
    ├─[2] 각각 Pareto front 계산
    │     └─→ compute_network_pareto() × 3
    │
    └─[3] 비교 플롯 생성
          │
          ├─→ network_pareto_3way_all_coupled_legacy.png
          │       └─→ 4x3 그리드 (7 Pareto + HV bar chart)
          │
          └─→ network_pareto_ls_markers.png
                  └─→ LS1-LS5 마커 표시
```

---

## 6. SystemC 시뮬레이터 내부

```
src/systemc/pipeline_sim (C++ 실행 파일)
    │
    ├─→ main.cpp::main()
    │       │
    │       ├─→ JSON 입력 파싱
    │       │       └─→ strategy config 로드
    │       │
    │       └─→ PipelineSimulator 생성 및 실행
    │
    └─→ PipelineSimulator (pipeline_simulator.h)
            │
            ├─→ sc_main()
            │       └─→ SystemC 시뮬레이션 시작
            │
            ├─→ 파이프라인 스테이지 실행
            │       ├─→ Input Buffer (IBUF)
            │       ├─→ CIM Array
            │       ├─→ Output Buffer (OBUF)
            │       └─→ External Memory
            │
            └─→ 결과 출력
                    ├─→ simulation_statistics.json
                    │       └─→ {latency_ns, mac_count, sram_reads, ...}
                    │
                    └─→ memory_metadata.json
                            └─→ {memory_access_pattern, ...}
```

---

## 7. 핵심 데이터 구조

### 7.1 Strategy JSON

```json
// strategies/L0_S0_out4x4_in8x8.json
{
  "layer_index": 0,
  "strategy_id": "S0",
  "output_tile": {"p": 4, "q": 4},
  "input_tile": {"p": 8, "q": 8},
  "num_output_tiles": 16,
  "num_input_tiles": 4,
  "tiling_case": "sub_tiling",
  "layer_params": {
    "H": 32, "W": 32, "C": 3,
    "R": 3, "S": 3, "M": 64
  }
}
```

### 7.2 Simulation Statistics JSON

```json
// simulations/L0_S0/simulation_statistics.json
{
  "latency_ns": 12345.67,
  "total_cycles": 2058,
  "mac_operations": 1048576,
  "sram_reads": 524288,
  "sram_writes": 262144,
  "dram_reads": 32768,
  "dram_writes": 16384
}
```

### 7.3 DB strategy_results Row

```sql
(id, layer_index, strategy_id,
 latency_ns, energy_nj,
 area_mm2, ibuf_area_mm2, obuf_area_mm2, cim_area_mm2,
 output_tile_p, output_tile_q, input_tile_p, input_tile_q,
 created_at)
```

---

## 8. 호출 관계 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                        efsim CLI                                │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ dse_benchmark│     │  simulate   │     │    plot     │
   └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │TilingGenerator│   │SystemCRunner│     │pareto_sampling│
   └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ strategies/ │     │ simulations/│     │ pareto_math │
   │   *.json    │     │   L*_S*/    │     │  OBJECTIVES │
   └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ strategies.db   │
                    │ (SQLite)        │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ plots/*.png     │
                    │ plots/*.csv     │
                    └─────────────────┘
```

---

## 9. 디버깅 팁

### 특정 단계 로그 확인

```bash
# 전략 생성 로그
PYTHONPATH=src/python python3 -c "
from core.tiling_generator import IndependentTilingGenerator
# ...
"

# 시뮬레이션 단일 실행
./src/systemc/pipeline_sim < strategies/L0_S0.json

# Pareto 계산만
PYTHONPATH=src/python python3 -m visualization.plotting.pareto_sampling
```

### Ray 워커 로그 위치

```
/mnt/workers/{hostname}/{workspace}/
├── strategies.db          # 브로드캐스트된 DB
└── pareto_results/
    └── pareto_task_*.json # 워커 결과
```
