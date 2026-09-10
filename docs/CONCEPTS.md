# Core Concepts

프로젝트 이해에 필수적인 도메인 지식.

## 1. 좌표계 (Coordinate Systems)

CNN 레이어에서 세 가지 좌표계 사용:

```
Input (H×W×C)  →  Convolution  →  Output (P×Q×M)  →  Pooling  →  Pooled (P'×Q'×M)
```

### 1.1 입력 좌표계 (H×W)

```
H: Input height (e.g., 224)
W: Input width (e.g., 224)
C: Input channels (e.g., 3)
```

### 1.2 출력 좌표계 (P×Q)

Convolution 후 출력 크기:

```
P = (H - R) / stride + 1
Q = (W - S) / stride + 1

예: H=224, R=3, stride=1 → P=222
```

### 1.3 풀링 후 좌표계 (P_pooled×Q_pooled)

Pooling 후 출력 크기:

```
P_pooled = P / pool_height
Q_pooled = Q / pool_width

예: P=222, pool_height=2 → P_pooled=111
```

### 1.4 입력 타일 크기 계산

출력 타일 (d_p×d_q)를 생성하려면 필요한 입력 크기:

```
input_h = R + (d_p - 1) × stride
input_w = S + (d_q - 1) × stride

예: d_p=4, R=3, stride=1 → input_h=6
```

---

## 2. Independent Tiling

### 2.1 개념

입력/출력 타일 크기를 **독립적으로** 설정하는 전략.

기존 방식은 입력 타일 = 출력 타일 (coupled) 이었으나,
Independent Tiling은 이를 분리하여 더 넓은 설계 공간 탐색 가능.

### 2.2 Case 1: Sub-tiling (여러 작은 입력 → 1 출력)

```
┌─────────────────────────────┐
│      Output Tile (4×4)      │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐   │
│  │i1 │ │i2 │ │i3 │ │i4 │   │  ← 4개의 입력 타일
│  └───┘ └───┘ └───┘ └───┘   │
└─────────────────────────────┘
```

- `input_tile < output_tile`
- 하나의 출력 타일을 생성하기 위해 여러 작은 입력 타일 사용
- 작은 버퍼로 큰 출력 생성 가능
- 메모리 제약 환경에 적합

### 2.3 Case 2: Super-tiling (1 큰 입력 → 여러 출력)

```
┌─────────────────────────────┐
│     Large Input Tile        │
│  ┌─────────────────────┐    │
│  │                     │    │
│  │   →  o1  o2  o3     │    │  ← 1 입력 → 3 출력
│  │                     │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
```

- `input_tile >= output_tile`
- 하나의 큰 입력 타일로 여러 출력 타일 생성
- 입력 데이터 재사용 극대화
- 메모리 여유 있는 환경에 적합

### 2.4 코드에서의 표현

```python
# tiling_generator.py
CASE_TYPE_SUB_TILING = 1   # input_tile < output_tile
CASE_TYPE_SUPER_TILING = 2  # input_tile >= output_tile
```

---

## 3. Legacy Strategies (LS1-LS5)

5가지 참조용 기준 전략. 비교 분석에 사용.

### 3.1 LS1: Output Stationary (Image-wise)

```
IBUF: H × W × C       (전체 입력 이미지)
OBUF: P' × Q' × M     (전체 출력 이미지)
```

- 가중치 재사용 최대화
- 메모리 풍부한 환경
- 가장 큰 버퍼 필요

### 3.2 LS2: Weight Stationary (Window-based)

```
IBUF: R × S × C       (단일 컨볼루션 윈도우)
OBUF: P' × Q' × M     (배치 출력)
```

- 메모리와 연산 균형
- 중간 크기 버퍼

### 3.3 LS3: Input Stationary (Per-pixel)

```
IBUF: R × S × C       (단일 수용영역)
OBUF: M               (단일 출력 픽셀)
```

- 최소 버퍼 사용
- 메모리 제약 환경에 적합
- 가장 많은 외부 메모리 접근

### 3.4 LS4: Pipeline Tiling (2×2 tile)

```
IBUF: (R + (pool-1)×stride) × (S + (pool-1)×stride) × C
OBUF: M               (풀링 후 단일 픽셀)
```

- 풀링과 연계된 타일링
- 중간 메모리 사용

### 3.5 LS5: Maximum Tiling (Line-wise)

```
IBUF: (R + (pool-1)×stride) × W × C   (풀링 그룹 라인)
OBUF: Q' × M                          (전체 출력 라인)
```

- 라인 단위 처리
- 높은 데이터 재사용

### 3.6 플롯에서의 표시

```python
# plot_constants.py
LEGACY_MARKERS = {
    1: ("o", "red",    ..., "LS1"),   # Output Stationary
    2: ("o", "blue",   ..., "LS2"),   # Weight Stationary
    3: ("o", "green",  ..., "LS3"),   # Input Stationary
    4: ("o", "orange", ..., "LS4"),   # Pipeline Tiling
    5: ("o", "purple", ..., "LS5"),   # Maximum Tiling
}
```

---

## 4. Coupled Strategies

**입력 타일 == 출력 타일**인 전략.

```sql
WHERE input_tile_p = output_tile_p
  AND input_tile_q = output_tile_q
```

### 4.1 특징

- 기존 방식과 호환
- 설계 공간이 더 작음 (제약 조건)
- 플롯에서 사각형 마커로 표시

### 4.2 Independent Tiling과의 비교

```
All Strategies (Independent)
└── Coupled Strategies (input == output)
    └── Legacy Strategies (LS1-LS5 특정 패턴)
```

---

## 5. Analytical Model

SystemC 시뮬레이션 없이 빠른 성능 추정.

### 5.1 레이턴시 모델

```
t_exec = max{t_comm, t_comp}
```

#### t_comm (메모리 스테이지, 타일링 의존)

```
t_ext_sum = n_in × t_ext_r + n_out × t_ext_w
t_comm = max{t_ext_sum, n_in × t_ibuf_w, n_out × t_obuf_r}
```

#### t_comp (연산 스테이지, 타일링 무관)

```
t_comp = max{P×Q×t_ibuf_r, P×Q×t_mac, P'×Q'×t_obuf_w}
```

### 5.2 6가지 Bottleneck

| Bottleneck | 설명 |
|------------|------|
| `ext` | 외부 메모리 (DRAM) 대역폭 |
| `ibuf_w` | 입력 버퍼 쓰기 |
| `obuf_r` | 출력 버퍼 읽기 |
| `ibuf_r` | 입력 버퍼 읽기 |
| `imc` | CIM 연산 (In-Memory Computing) |
| `obuf_w` | 출력 버퍼 쓰기 |

### 5.3 사용 위치

```python
# analysis/analytical_model.py
model = AnalyticalModel.from_config(hw_config)
result = model.estimate(layer_config, tiling_config)
# result: AnalyticalResult(latency_ns, energy_nj, area_mm2, bottleneck)
```

---

## 6. 7가지 Pareto Objectives

### 6.1 기본 메트릭

| 메트릭 | 단위 | 설명 |
|--------|------|------|
| `latency_ns` | ns | 총 실행 시간 (모든 레이어 합) |
| `energy_nj` | nJ | 총 에너지 (모든 레이어 합) |

### 6.2 면적 메트릭

| 메트릭 | 계산 | 설명 |
|--------|------|------|
| `buffer_area_mm2` | max(ibuf) + max(obuf) | 버퍼 면적 (재사용 가정) |
| `sum_area_mm2` | buffer + sum(cim) | 총 면적 (CIM 합) |
| `peak_area_mm2` | buffer + max(cim) | 피크 면적 (CIM 최대) |

### 6.3 EAP (Energy-Area Product)

| 메트릭 | 계산 | 의미 |
|--------|------|------|
| `buffer_eap` | buffer_area × energy | 버퍼 효율성 |
| `sum_eap` | sum_area × energy | 총 효율성 |
| `peak_eap` | peak_area × energy | 피크 효율성 |

### 6.4 왜 7개인가?

- X축: 항상 `latency_ns` (고정)
- Y축: 나머지 7개 메트릭

→ 7개의 2D Pareto front

---

## 7. Network-level Metrics

레이어별 메트릭을 네트워크 전체로 집계하는 방식.

### 7.1 합산 메트릭

```python
total_latency = sum(layer.latency_ns for layer in layers)
total_energy = sum(layer.energy_nj for layer in layers)
```

### 7.2 버퍼 면적 (최대값)

```python
# 버퍼는 레이어 간 재사용 가정
buffer_area = max(layer.ibuf_area for layer in layers) + \
              max(layer.obuf_area for layer in layers)
```

### 7.3 CIM 면적 (두 가지 방식)

```python
# sum: 각 레이어에 별도 CIM
sum_cim = sum(layer.cim_area for layer in layers)

# peak: CIM 재사용 (최대값만 필요)
peak_cim = max(layer.cim_area for layer in layers)
```

### 7.4 총 면적

```python
sum_area = buffer_area + sum_cim    # 보수적
peak_area = buffer_area + peak_cim  # 낙관적
```

---

## 8. Monte Carlo Sampling

네트워크 레벨 Pareto를 찾기 위한 전략 조합 샘플링.

### 8.1 문제

- N개 레이어, 각각 M개 전략
- 총 조합: M^N (지수적 증가)
- 예: 4 레이어 × 1000 전략 = 10^12 조합

### 8.2 해결

- Monte Carlo 샘플링 (기본 20M samples)
- Ray 분산 처리
- 로컬 Pareto → 전역 Pareto 병합

### 8.3 설정

```python
# plot_constants.py
DEFAULT_NUM_SAMPLES = 20_000_000      # 20M samples
DEFAULT_SAMPLES_PER_TASK = 20_000     # 태스크당 샘플
DEFAULT_MAX_SCATTER_POINTS = 50_000   # 시각화용 최대 점
```

---

## 9. Hypervolume Indicator

Pareto front 품질 측정 지표.

### 9.1 정의

Reference point와 Pareto front 사이의 면적/부피.

```
         ┌─────────────────────┐ Reference Point (max×1.1)
         │░░░░░░░░░░░░░░░░░░░░░│
         │░░░░░░░░░░░░░░░░░░░░░│
         │░░░░*─────*░░░░░░░░░░│ ← Pareto Front
         │░░░░░░░░░░│░░░░░░░░░░│
         │░░░░░░░░░░*──────*░░░│
         │░░░░░░░░░░░░░░░░░│░░░│
         └─────────────────────┘
              ↑ Hypervolume (음영 영역)
```

### 9.2 해석

- 높을수록 더 좋은 Pareto front
- 비교 분석에 사용 (All vs Coupled vs Legacy)

---

## 10. 용어 정리

| 용어 | 의미 |
|------|------|
| Strategy | 특정 타일 크기 조합 (하나의 설계 점) |
| Tiling | 출력을 타일로 분할하여 처리하는 방식 |
| CIM | Compute-in-Memory (메모리 내 연산) |
| IBUF | Input Buffer (입력 버퍼) |
| OBUF | Output Buffer (출력 버퍼) |
| MAC | Multiply-Accumulate (곱셈-누적 연산) |
| EAP | Energy-Area Product |
| DSE | Design Space Exploration (설계 공간 탐색) |
