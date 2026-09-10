# CiMFlowSim Architecture Guide

프로젝트 유지보수 및 수정을 위한 아키텍처 문서.

## 1. 시스템 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                           efsim CLI                                 │
│                     (./efsim <command>)                             │
└─────────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   benchmark   │      │    simulate     │      │      plot       │
│ (전체 워크플로우) │      │  (Ray 병렬 실행)  │      │   (시각화 생성)   │
└───────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ TilingGenerator│      │  SystemCRunner  │      │ Pareto Sampling │
│ (전략 생성)     │      │  (C++ 실행)      │      │ (Monte Carlo)   │
└───────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PerformanceDatabase (SQLite)                    │
│                        strategies.db                                │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. 핵심 모듈 의존성

### 수정 시 영향 범위

| 모듈 수정 | 영향받는 모듈 |
|-----------|--------------|
| `tiling_generator.py` | simulate, 전략 JSON 형식 |
| `performance_database.py` | plot, score, pattern (DB 스키마 변경 시 전체) |
| `pareto_math.py:OBJECTIVES` | 모든 Pareto 플롯, CSV 출력 |
| `plot_constants.py` | 모든 시각화 모듈 |
| `systemc_runner.py` | simulate (Ray 워커) |

### 모듈 역할

```
src/python/
├── core/
│   ├── tiling_generator.py     # 전략 생성 (입력: network config → 출력: strategies/*.json)
│   ├── tiling.py               # 데이터 구조 (CNNLayerParams, TilingConfig)
│   ├── performance_database.py # DB 스키마 및 CRUD (strategy_results, legacy_mappings 테이블)
│   ├── systemc_runner.py       # C++ 실행 + 결과 파싱 + 에너지/면적 계산
│   ├── energy_calculator.py    # 에너지 계산 (simulation_statistics.json → energy_nj)
│   ├── area_calculator.py      # 면적 계산 (타일 크기 → area_mm2)
│   ├── strategy_scorer.py      # ./efsim score 구현
│   └── pattern_analyzer.py     # ./efsim pattern 구현
│
└── visualization/plotting/
    ├── pareto_math.py          # OBJECTIVES 정의, Pareto 계산 알고리즘
    ├── pareto_sampling.py      # Ray 분산 Monte Carlo 샘플링
    ├── pareto_utils.py         # DB에서 전략 로딩 (load_all_layer_strategies 등)
    ├── pareto_grid_plot.py     # 3x3 그리드 플롯 생성
    ├── pareto_common.py        # 공통 유틸리티 (경로, 마커 등)
    ├── plot_constants.py       # 모든 플롯 상수 (크기, 색상, 폰트)
    └── plot_network_pareto_comparison.py  # 비교 플롯 (All vs Coupled vs Legacy)
```

## 3. 데이터 플로우 상세

### 3.1 전략 생성 → 시뮬레이션 → DB

```
1. TilingGenerator.generate_all_strategies()
   │
   ├─→ strategies/L0_S0_out2x2_in5x5.json
   ├─→ strategies/L0_S1_out2x2_in10x10.json
   └─→ ...

2. Ray Worker (각 노드에서 병렬 실행)
   │
   ├─→ SystemCRunner.run_simulation(strategy_json)
   │       └─→ simulations/L0_S0/simulation_statistics.json
   │       └─→ simulations/L0_S0/memory_metadata.json
   │
   ├─→ EnergyCalculator.calculate(simulation_statistics)
   │       └─→ energy_nj
   │
   └─→ AreaCalculator.calculate(tile_sizes)
           └─→ area_mm2, ibuf_area, obuf_area, cim_area

3. Main Process (HEAD 노드)
   │
   └─→ PerformanceDatabase.insert_strategy_result(...)
           └─→ strategies.db (strategy_results 테이블)
```

### 3.2 Pareto 플롯 생성

```
1. pareto_utils.load_all_layer_strategies(db_path)
   │
   └─→ Dict[layer_index, List[strategy_dict]]

2. pareto_sampling.run_distributed_pareto_sampling()
   │
   ├─→ HEAD: DB를 tar.gz로 압축 → 모든 노드에 브로드캐스트
   ├─→ WORKER: 로컬에서 Monte Carlo 샘플링 (20M samples)
   ├─→ WORKER: 로컬 Pareto front 계산
   ├─→ WORKER: 결과를 /mnt/workers/{hostname}/ 에 저장
   └─→ HEAD: 모든 노드 결과 수집 → 전역 Pareto front 병합

3. pareto_grid_plot.plot_pareto_grid()
   │
   └─→ plots/network_pareto.png
```

## 4. DB 스키마

### strategy_results (핵심 테이블)

```sql
CREATE TABLE strategy_results (
    id INTEGER PRIMARY KEY,
    layer_index INTEGER,
    strategy_id TEXT,

    -- 성능 메트릭
    latency_ns REAL,
    energy_nj REAL,

    -- 면적 메트릭
    area_mm2 REAL,
    ibuf_area_mm2 REAL,
    obuf_area_mm2 REAL,
    cim_area_mm2 REAL,

    -- 타일 크기
    output_tile_p INTEGER,
    output_tile_q INTEGER,
    input_tile_p INTEGER,
    input_tile_q INTEGER,

    -- 기타
    created_at TIMESTAMP
);
```

### legacy_mappings (레거시 전략 매핑)

```sql
CREATE TABLE legacy_mappings (
    layer_index INTEGER,
    strategy_id TEXT,
    legacy_number INTEGER,  -- 1-5 (LS1-LS5)
    legacy_name TEXT,
    PRIMARY KEY (layer_index, legacy_number)
);
```

## 5. 7가지 Pareto Objectives

`pareto_math.py`에 정의됨. **순서 변경 시 모든 플롯 영향**.

```python
OBJECTIVES = [
    ("latency_ns", "energy_nj", "Latency (ns)", "Energy (nJ)"),
    ("latency_ns", "buffer_area_mm2", "Latency (ns)", "Buffer Area (mm²)"),
    ("latency_ns", "sum_area_mm2", "Latency (ns)", "Sum Area (mm²)"),
    ("latency_ns", "peak_area_mm2", "Latency (ns)", "Peak Area (mm²)"),
    ("latency_ns", "buffer_eap", "Latency (ns)", "Buffer EAP"),
    ("latency_ns", "sum_eap", "Latency (ns)", "Sum EAP"),
    ("latency_ns", "peak_eap", "Latency (ns)", "Peak EAP"),
]
```

**Area 계산 방식:**
- `buffer_area` = max(ibuf_area) + max(obuf_area) (레이어 간)
- `sum_area` = buffer_area + sum(cim_area)
- `peak_area` = buffer_area + max(cim_area)
- `EAP` = Area × Energy (Energy-Area Product)

## 6. Ray 분산 처리

### 워커 데이터 경로

```
/mnt/workers/{hostname}/{workspace_name}/
├── strategies.db              # HEAD에서 브로드캐스트된 DB
├── analytical_strategies.json # 분석 모델용 (선택)
└── pareto_results/
    └── pareto_task_{seed}.json
```

- 워커 노드: 로컬 디스크 (빠른 I/O)
- HEAD 노드: NFS 마운트 (결과 수집용)

### 태스크 스케줄링

```python
@ray.remote(num_cpus=1, scheduling_strategy="SPREAD")
def worker_function(...):
    ...
```

모든 태스크는 `SPREAD` 전략으로 노드 간 균등 분배.

## 7. 주요 수정 시나리오

### 7.1 새로운 Objective 추가

1. `pareto_math.py`: `OBJECTIVES` 리스트에 추가
2. `pareto_sampling.py`: `WORKER_OBJECTIVES` 동기화
3. `plot_constants.py`: `SUBPLOT_POSITIONS_7` → `SUBPLOT_POSITIONS_8` (필요시)
4. `pareto_grid_plot.py`: 레이아웃 조정

### 7.2 DB 스키마 변경

1. `performance_database.py`: 테이블/컬럼 추가
2. `pareto_utils.py`: 쿼리 수정
3. **기존 워크스페이스 마이그레이션 필요**

### 7.3 새로운 플롯 타입 추가

1. `visualization/plotting/plot_*.py` 생성
2. `generate_all_plots.py`에 등록
3. `tools/plot.py`에서 호출

### 7.4 하드웨어 파라미터 추가

1. `configs/hardware/*.json` 형식 수정
2. `energy_calculator.py` 또는 `area_calculator.py` 수정
3. `systemc_runner.py`에서 C++로 전달

## 8. 트러블슈팅

### 일반적인 오류

| 오류 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: orjson` | Ray 환경 미활성화 | `source /opt/rayenv/bin/activate` |
| `sqlite3.Row has no attribute 'get'` | `.get()` 대신 `[]` 사용 필요 | `row["column"]` 으로 변경 |
| `Ray tasks stuck` | 클러스터 상태 이상 | `ray status` 확인, 필요시 재시작 |
| `NFS timeout` | 네트워크 문제 | `/mnt/workers/` 마운트 확인 |
| `SystemC simulation failed` | C++ 빌드 문제 | `cd src/systemc && make clean && make` |
| `No strategies found` | 전략 생성 안됨 | `./efsim benchmark` 먼저 실행 |
| `Database locked` | 동시 접근 충돌 | 다른 프로세스 종료 후 재시도 |

### 시뮬레이션 실패 복구

```bash
# 실패한 전략 재시도
./efsim simulate WORKSPACE --resume --retry-failed

# 특정 레이어만 확인
./efsim plot WORKSPACE --layers 0,1

# 시뮬레이션 상태 확인
./efsim info WORKSPACE
```

### Ray 클러스터 재시작

```bash
# 전체 클러스터 (Ansible)
cd ~/ansible-ray-cluster
ansible all -m shell -a "source /opt/rayenv/bin/activate && ray stop --force"
ansible all -m systemd -a "name=ray state=started" --become
```

### 메모리 부족

```bash
# CPU 수 제한으로 메모리 사용 감소
./efsim benchmark --all --max-cpus 90

# Monte Carlo 샘플 수 감소
./efsim plot WORKSPACE --num-samples 1000000
```

## 9. FAQ

### Q: 시뮬레이션이 중단되면 어떻게 하나요?
```bash
./efsim simulate WORKSPACE --resume
```
이미 완료된 전략은 건너뛰고 나머지만 실행합니다.

### Q: 특정 레이어만 분석하고 싶습니다
```bash
./efsim plot WORKSPACE --layers 0,2,4
./efsim score WORKSPACE --layer 0
```

### Q: Monte Carlo 샘플 수는 얼마가 적당한가요?
- 기본값: 20,000,000 (정밀도 높음, 느림)
- 빠른 테스트: 1,000,000
- 초고정밀: 100,000,000

### Q: Pareto front가 비어있습니다
1. 시뮬레이션 완료 확인: `./efsim info WORKSPACE`
2. DB 확인: `sqlite3 workspaces/WORKSPACE/strategies.db "SELECT COUNT(*) FROM strategy_results"`

### Q: 새 하드웨어 설정을 추가하려면?
1. `configs/hardware/active/` 에 JSON 파일 생성
2. 기존 파일 복사 후 수정
3. `./efsim benchmark --hardware NEW_NAME` 으로 테스트

### Q: Ray 대시보드는 어디서 확인하나요?
http://<head-node-ip>:8265

### Q: Sweep 실행 중 중단되면?
`--resume` 플래그로 이어서 실행 가능:
```bash
# Sweep 실행 재개
./efsim sweep --config configs/sweep/my_config.json --resume

# Sweep 플롯 재생성 재개
./efsim plot-sweep workspaces/sweep_xxx --resume
```

진행상황은 `sweep_progress.json`, `plot_regen_progress.json`에 자동 저장됨.

## 10. 테스트

```bash
# 전체 벤치마크 테스트
./efsim benchmark --all

# 특정 워크스페이스 플롯만 재생성
./efsim plot WORKSPACE_NAME

# 결과 검증
ls workspaces/WORKSPACE_NAME/plots/*.png | wc -l
```

## 11. 파일 위치 빠른 참조

| 기능 | 파일 |
|------|------|
| CLI 진입점 | `efsim` |
| 전략 생성 | `src/python/core/tiling_generator.py` |
| Ray 시뮬레이션 | `tools/simulate.py` |
| Pareto 계산 | `src/python/visualization/plotting/pareto_math.py` |
| 분산 샘플링 | `src/python/visualization/plotting/pareto_sampling.py` |
| DB 로딩 | `src/python/visualization/plotting/pareto_utils.py` |
| 플롯 상수 | `src/python/visualization/plotting/plot_constants.py` |
| 그리드 플롯 | `src/python/visualization/plotting/pareto_grid_plot.py` |
| 비교 플롯 | `src/python/visualization/plotting/plot_network_pareto_comparison.py` |
| Sweep 실행 | `tools/sweep.py`, `src/python/core/sweep_executor.py` |
| Sweep 분석 | `tools/sweep_analysis.py` |
| Sweep 플롯 재생성 | `efsim` (plot_sweep_command 함수) |
