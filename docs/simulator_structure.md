# 1.1 PIM Simulator 개발

## 1.1.1 Simulator 구조 및 특징

### 1.1.1.1 개요
- **구성 요소 집약형 PIMSimulator**: eFlash Array 묶음, 선택적 NPU, 공용 SRAM을 한 컨테이너로 묶어 초기 자원 크기·성능(TOPS, 배열/에어리어 실행 시간)을 설정하고 전체 실행 통계를 수집한다.【F:src/hardware/pim_simulator.py†L7-L116】
- **타일·버퍼 단위의 경량 메타모델**: WeightTile과 SRAMBuffer가 데이터 자체 없이 shape·위치·크기만 추적하여 배치/전송/용량 검증에 집중한다.【F:src/hardware/weight_tile.py†L4-L35】【F:src/hardware/sram.py†L4-L105】
- **그래프-스케줄 결합형 워크플로**: ComputeGraph가 노드 의존성과 배치 메타데이터를 담고, InferenceScheduler가 이벤트 드리븐 방식으로 전송·연산·메모리 수명을 추적해 실행 타임라인과 리소스 사용률을 산출한다.【F:src/graph/compute_node.py†L4-L159】【F:src/scheduler/scheduler.py†L11-L205】【F:src/scheduler/scheduler.py†L924-L999】

### 1.1.1.2 전체 구조
- **하드웨어 계층**
  - eFlashArray: 8개 Area와 전용 SRAM(기본 1MB)을 보유하며, Area 실행 시간(us)과 배치된 타일 수, 총 실행 시간 통계를 유지한다.【F:src/hardware/eflash_array.py†L6-L71】
  - Area: 최대 128×1280 행렬 공간을 row/리덕션 축으로 패킹하며, 중첩 없이 배치 가능한지 검증한 뒤 WeightTile을 생성·추적한다.【F:src/hardware/area.py†L5-L118】
  - NPU: TOPS를 기반으로 연산 시간을 계산하고 전용 SRAM(기본 2MB)과 busy 타임을 관리한다.【F:src/hardware/npu.py†L5-L65】
  - SRAMBuffer: 이름별로 크기와 사용량을 기록하고 할당/해제/사용률·가용량을 집계한다.【F:src/hardware/sram.py†L4-L105】
- **그래프 계층**
  - ComputeNode/ComputeGraph: 노드별 장치 타입(eflash/NPU), 배치 위치(array/area), 의존성, shape, 타일 리스트를 관리하며 실행 가능 노드 선별과 수행 순서 기록을 지원한다.【F:src/graph/compute_node.py†L4-L159】
  - GraphPreprocessor: 다중 타일 Conv를 tile→reduce→concat 서브그래프로 확장하고 consumer 연결을 재배선하여 스케줄링 단순화를 돕는다.【F:src/graph/graph_preprocessor.py†L17-L159】
  - GraphValidator: Conv 노드의 weight shape/배치/패킹을 검증하고 오류·경고를 수집한다.【F:src/graph/graph_validator.py†L9-L185】
- **스케줄러 계층**
  - InferenceScheduler: 공유 SRAM 대역폭 기반 전송 시간, array/NPU busy 타임, activation 위치/복제/해제 이벤트를 관리하며 TRANSFER/COMPUTE 이벤트를 우선순위 큐로 실행한다.【F:src/scheduler/scheduler.py†L11-L205】【F:src/scheduler/scheduler.py†L406-L759】【F:src/scheduler/scheduler.py†L924-L999】
  - TimelineFormatter 기반 출력: 실행·메모리 타임라인과 연산/전송 중첩 요약을 제공해 병렬성·대역폭 활용도를 분석할 수 있다.【F:src/scheduler/scheduler.py†L1001-L1023】

### 1.1.1.3 Profiling 목표
- **배치 및 용량 검증**: eFlash Array/Area별 최대 출력·리덕션 한계(128×1280)와 SRAM 가용량을 만족하는지 검사하고, 공유/로컬 SRAM 사용률·잔여 용량을 수치화한다.【F:src/hardware/area.py†L5-L118】【F:src/hardware/sram.py†L18-L105】【F:src/hardware/pim_simulator.py†L97-L116】
- **전송·연산 시간 분리 측정**: 공유 버스 대역폭으로 전송 시간을 산출하고, NPU TOPS 또는 Area 실행 시간 기반으로 연산 시간을 분리 집계해 total_compute_time_us/total_transfer_time_us로 보고한다.【F:src/scheduler/scheduler.py†L139-L205】【F:src/scheduler/scheduler.py†L924-L999】
- **타임라인 기반 병렬성 분석**: 이벤트 시퀀스와 메모리 이벤트를 수집해 완료 노드 수, 해제 횟수/바이트, 타임라인·오버랩 요약을 출력함으로써 전송/연산 중첩과 버퍼 수명 최적화를 평가한다.【F:src/scheduler/scheduler.py†L924-L1023】
