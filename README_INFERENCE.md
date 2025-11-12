# PIM Simulator - Event-Driven Inference System

Processing-In-Memory (PIM) 시뮬레이터의 Event-driven Inference 시스템 문서

## 개요

ONNX 그래프 기반으로 dependency를 고려한 자동 스케줄링을 제공하며, 여러 eFlash Array의 병렬 실행과 메모리 계층 구조를 시뮬레이션합니다.

## 주요 기능

### 1. **Event-Driven 시뮬레이션**
- 우선순위 큐 기반 이벤트 처리
- 시간 순서대로 연산 실행
- 정확한 레이턴시 추적

### 2. **Dependency-Aware 스케줄링**
- 데이터 의존성 자동 분석
- Ready queue 관리
- 병렬 실행 가능한 노드 자동 감지

### 3. **병렬 실행**
- 여러 eFlash Array 동시 실행
- Array별 독립적인 busy 시간 관리
- Dependency 없으면 동시 스케줄링

### 4. **메모리 계층 구조**
- **내부 SRAM**: 같은 Array 내 데이터 전달
- **공유 SRAM**: Array 간 데이터 전달, residual connection
- 자동 위치 결정 (consumer 노드 분석)

### 5. **데이터 전송 시뮬레이션**
- 대역폭 기반 전송 시간 계산
- 외부 SRAM 접근 시간 고려
- Activation 크기 자동 계산

### 6. **Pipeline 지원**
- 여러 입력 연속 처리
- 처리량(throughput) 계산
- 평균 레이턴시 측정

## 아키텍처

```
┌─────────────────────────────────────────────────┐
│           PIMSimulator                          │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ eFlashArray 0│  │ eFlashArray 1│            │
│  │  ┌────────┐  │  │  ┌────────┐  │            │
│  │  │ Area 0 │  │  │  │ Area 0 │  │            │
│  │  │ Area 1 │  │  │  │ Area 1 │  │            │
│  │  └────────┘  │  │  └────────┘  │            │
│  │  [Int SRAM]  │  │  [Int SRAM]  │            │
│  └──────────────┘  └──────────────┘            │
│           [Shared SRAM]                         │
└─────────────────────────────────────────────────┘
                    ▲
                    │
         ┌──────────┴──────────┐
         │  InferenceScheduler  │
         │  • Event Queue       │
         │  • Dependency Check  │
         │  • Memory Manager    │
         └─────────────────────┘
                    ▲
                    │
         ┌──────────┴──────────┐
         │   ComputeGraph      │
         │  • Nodes            │
         │  • Dependencies     │
         │  • Execution Order  │
         └─────────────────────┘
```

## 주요 클래스

### Event & EventType
```python
class EventType(Enum):
    COMPUTE_START = "compute_start"
    COMPUTE_DONE = "compute_done"
    TRANSFER_START = "transfer_start"
    TRANSFER_DONE = "transfer_done"
    INFERENCE_START = "inference_start"
    INFERENCE_DONE = "inference_done"

class Event:
    - time_ns: 이벤트 발생 시간
    - event_type: 이벤트 타입
    - node_id: 관련 노드 ID
    - data: 추가 데이터
```

### ActivationBuffer & ActivationManager
```python
class ActivationBuffer:
    - buffer_id: 버퍼 식별자
    - shape: Activation shape
    - size_bytes: 크기 (8bit/element)
    - location: 저장 위치

class ActivationManager:
    - create_buffer(): 버퍼 생성
    - allocate_to_sram(): SRAM 할당
    - get_node_output_buffer(): 노드 출력 조회
```

### ComputeNode & ComputeGraph
```python
class ComputeNode:
    - node_id: 노드 ID
    - node_type: "conv", "add", etc.
    - array_id, area_id: weight 위치
    - input_nodes: dependency 리스트
    - input_shape, output_shape
    - is_ready, is_running, is_done

class ComputeGraph:
    - add_node(): 노드 추가
    - get_ready_nodes(): 실행 가능한 노드
    - get_source_nodes(): 소스 노드
```

### InferenceScheduler
```python
class InferenceScheduler:
    - pim_simulator: PIM 시뮬레이터
    - compute_graph: 연산 그래프
    - activation_manager: Activation 관리
    - event_queue: 우선순위 큐
    - array_busy_until: Array 사용 상태
    
    - run_inference(): Inference 실행
    - _schedule_compute(): 연산 스케줄링
    - _handle_compute_done(): 완료 이벤트 처리
    - print_timeline(): 타임라인 출력
```

### InferenceContext & PipelineManager
```python
class InferenceContext:
    - context_id: 컨텍스트 ID
    - scheduler: 스케줄러
    - input_batch_size, input_shape
    - execute(): Inference 실행
    - get_latency(): 레이턴시 반환

class PipelineManager:
    - add_inference(): Inference 추가
    - run_sequential(): 순차 실행
    - get_total_throughput(): 처리량
    - get_average_latency(): 평균 레이턴시
```

## 사용 예제

### 1. 기본 사용

```python
from src.pim_simulator import PIMSimulator
from src.compute_node import ComputeNode, ComputeGraph
from src.scheduler import InferenceScheduler
from src.inference_context import InferenceContext

# 1. PIM 생성
pim = PIMSimulator(
    num_arrays=2,
    area_execution_time_ns=100.0,
    array_sram_size_bytes=2 * 1024 * 1024,
    shared_sram_size_bytes=20 * 1024 * 1024
)

# 2. 그래프 생성
graph = ComputeGraph()

conv1 = ComputeNode(
    node_id="conv1",
    node_type="conv",
    array_id=0,
    area_id=0,
    input_nodes=[],  # 소스 노드
    output_shape=(64, 112, 112)
)
graph.add_node(conv1)

conv2 = ComputeNode(
    node_id="conv2",
    node_type="conv",
    array_id=0,
    area_id=1,
    input_nodes=["conv1"],  # conv1 출력 필요
    output_shape=(128, 56, 56)
)
graph.add_node(conv2)

# 3. Weight 배치
pim.place_weight_on_array(0, 0, "conv1_w", (64, 512))
pim.place_weight_on_array(0, 1, "conv2_w", (128, 1024))

# 4. 스케줄러 생성
scheduler = InferenceScheduler(
    pim_simulator=pim,
    compute_graph=graph,
    shared_sram_bandwidth_gbps=10.0
)

# 5. Inference 실행
context = InferenceContext(
    context_id="inference_0",
    scheduler=scheduler,
    input_batch_size=1,
    input_shape=(224, 224, 3)
)

result = context.execute()

print(f"레이턴시: {result['total_time_ns']/1e6:.4f} ms")
scheduler.print_timeline()
```

### 2. Residual Connection 예제

```python
# Residual: conv1 -> conv2a
#                 -> conv2b -> add(conv2a, conv2b)

conv1 = ComputeNode("conv1", "conv", 0, 0, [], (64, 112, 112))
conv2a = ComputeNode("conv2a", "conv", 0, 1, ["conv1"], (128, 56, 56))
conv2b = ComputeNode("conv2b", "conv", 1, 0, ["conv1"], (128, 56, 56))
add = ComputeNode("add", "add", 0, 2, ["conv2a", "conv2b"], (128, 56, 56))

# conv2a와 conv2b는 병렬 실행 가능!
# add의 출력은 자동으로 shared SRAM에 저장 (여러 consumer 가능)
```

### 3. Pipeline 실행

```python
pipeline = PipelineManager(scheduler)

# 여러 입력 추가
for i in range(10):
    pipeline.add_inference(f"inference_{i}", 1, (224, 224, 3))

# 순차 실행
results = pipeline.run_sequential()

stats = pipeline.get_stats()
print(f"평균 레이턴시: {stats['average_latency_ns']/1e6:.4f} ms")
print(f"처리량: {stats['throughput_per_sec']:.2f} inf/s")
```

## 실행 흐름

### 1. 초기화
```
1. PIM 시뮬레이터 생성
2. Weight를 eFlash Array의 Area에 배치
3. ComputeGraph 생성 (노드와 dependency)
4. InferenceScheduler 생성
```

### 2. Inference 실행
```
1. 소스 노드 탐색 (dependency 없는 노드)
2. 소스 노드들 스케줄링
3. Event loop:
   - Event queue에서 다음 이벤트 꺼내기
   - COMPUTE_DONE 처리:
     * 노드 완료 표시
     * Output activation 생성
     * SRAM 할당 (내부 or 공유)
     * 다음 ready 노드들 스케줄링
4. 모든 노드 완료 시 종료
```

### 3. 스케줄링 로직
```python
def _schedule_compute(node):
    # 1. Array 사용 가능 시간 확인
    available_time = max(current_time, array_busy_until[array_id])
    
    # 2. 입력 데이터 전송 시간 계산
    for input in node.inputs:
        if input.location == "shared_sram":
            transfer_time += data_size / bandwidth
    
    # 3. 연산 시작/완료 이벤트 생성
    compute_start = available_time + transfer_time
    compute_done = compute_start + area_execution_time
    
    # 4. Array busy 업데이트
    array_busy_until[array_id] = compute_done
```

### 4. Activation 위치 결정
```python
def determine_location(producer, consumers):
    if all consumers in same array as producer:
        return f"array_{array_id}_sram"  # 내부 SRAM
    else:
        return "shared_sram"  # 외부 SRAM
```

## 시간 모델

### 연산 시간
```
compute_time = area_execution_time_ns (고정값, 예: 100ns)
```

### 전송 시간
```
transfer_time = data_size_bytes / bandwidth_bytes_per_ns

예시:
- 데이터 크기: 1MB = 1,048,576 bytes
- 대역폭: 10 GB/s = 10 bytes/ns
- 전송 시간: 1,048,576 / 10 = 104,857.6 ns ≈ 0.105 ms
```

### 총 레이턴시
```
total_latency = max(array_done_times for all arrays)
```

## 주요 특징

### ✅ 자동 병렬화
- Dependency 없는 노드는 자동으로 병렬 실행
- 서로 다른 Array에서 동시 실행 가능

### ✅ 메모리 최적화
- 내부 SRAM 우선 사용 (같은 Array 내)
- 필요 시에만 공유 SRAM 사용

### ✅ 정확한 시간 모델링
- 연산 시간 + 전송 시간
- Array별 busy 시간 추적

### ✅ 확장 가능
- ONNX 파서 추가 가능
- 새로운 연산 타입 추가 쉬움
- Pipeline 스케줄링 확장 가능

## 실행 예제

```bash
# 기본 예제
python example_inference.py

# 출력:
# - 그래프 구조
# - Weight 배치
# - 타임라인
# - 실행 순서
# - 메모리 사용률
# - Pipeline 통계
```

## 예상 출력

```
[실행 순서]
  1. conv1     :     0.00ns ~   100.00ns [Array0]
  2. conv2a    : 80381.60ns ~ 80481.60ns [Array0]
  3. conv2b    : 80381.60ns ~ 80481.60ns [Array1]  ← 병렬!
  4. conv3     : 80481.60ns ~ 80581.60ns [Array1]
  5. add1      : 120722.40ns ~ 120822.40ns [Array0]
  6. conv4     : 120822.40ns ~ 120922.40ns [Array0]

총 실행 시간: 120.92 µs (0.121 ms)
```

## 향후 확장

- [ ] ONNX 파일 직접 파싱
- [ ] 진짜 Pipeline (입력1이 conv2일 때 입력2가 conv1 시작)
- [ ] Weight tile 분할 전략
- [ ] Dynamic 메모리 할당/해제
- [ ] 에너지 소비 모델
- [ ] NPU 유닛 추가
- [ ] 다양한 연산 타입 (Pool, BatchNorm, etc.)

## 파일 구조

```
PIM_simulator/
├── src/
│   ├── event.py              # Event, EventType
│   ├── activation.py         # ActivationBuffer, ActivationManager
│   ├── compute_node.py       # ComputeNode, ComputeGraph
│   ├── scheduler.py          # InferenceScheduler
│   ├── inference_context.py # InferenceContext, PipelineManager
│   └── ... (기존 파일들)
├── example_inference.py      # Event-driven 예제
└── README_INFERENCE.md       # 이 문서
```

## 문의 및 기여

이 시뮬레이터는 PIM 아키텍처 연구를 위한 도구입니다.
