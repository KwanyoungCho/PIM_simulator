# PIM Simulator

Processing-In-Memory (PIM) 시뮬레이터 - eFlash Array 기반

## 구조

```
PIMSimulator
├── eFlashArray (여러 개)
│   ├── Area (8개)
│   │   └── WeightTile (여러 개)
│   └── SRAM (내부 버퍼)
└── Shared SRAM (공유 메모리)
```

## 주요 구성 요소

### 1. **WeightTile**
- Area 내 배치된 개별 weight
- 실제 데이터 저장 없이 메타정보만 관리
- Shape: (output_dim, reduction_dim)
- Position: Area 내 행 범위

### 2. **Area**
- 최대 크기: 128 × 1280
- 여러 WeightTile 배치 가능
- 한 cycle에 1개 Area만 실행 가능
- Reduction 축: 1280 (한번 실행 시 전체 완료)

### 3. **SRAMBuffer**
- 데이터 이름과 크기만 관리
- 할당/해제 기능
- 사용률 추적

### 4. **eFlashArray**
- 8개의 Area 포함
- 내부 SRAM 버퍼
- Area 실행 시간 모델링

### 5. **PIMSimulator** (최상위)
- 여러 eFlashArray 관리
- 공유 SRAM
- 전체 시스템 통계

## 사용 예제

```python
from src.pim_simulator import PIMSimulator

# 시뮬레이터 생성
pim = PIMSimulator(
    num_arrays=4,                          # 4개의 eFlash Array
    area_execution_time_ns=100.0,          # Area 실행 시간: 100ns
    array_sram_size_bytes=1024 * 1024,     # 각 Array SRAM: 1MB
    shared_sram_size_bytes=10 * 1024 * 1024 # 공유 SRAM: 10MB
)

# Weight 배치
success, array_id, area_id = pim.auto_place_weight(
    weight_id="conv1",
    shape=(64, 512),  # 64 output, 512 reduction
    metadata={'layer': 'conv1', 'type': 'weight'}
)

# 공유 SRAM 할당
shared_sram = pim.get_shared_sram()
shared_sram.allocate("input_data", 512 * 1024)  # 512KB

# Area 실행
result = pim.execute_area(array_id=0, area_id=0)
print(f"실행 시간: {result['execution_time_ns']}ns")

# 통계 확인
stats = pim.get_total_stats()
print(f"전체 Tiles: {stats['total_tiles']}")
print(f"총 실행 시간: {stats['total_execution_time_ns']}ns")
```

## 실행

```bash
# 예제 실행
python example.py
```

## 파일 구조

```
PIM_simulator/
├── src/
│   ├── __init__.py
│   ├── weight_tile.py      # WeightTile 클래스
│   ├── area.py             # Area 클래스
│   ├── sram.py             # SRAMBuffer 클래스
│   ├── eflash_array.py     # eFlashArray 클래스
│   └── pim_simulator.py    # PIMSimulator 클래스 (최상위)
├── example.py              # 사용 예제
└── README.md
```

## 주요 특징

- ✅ 실제 weight 데이터 저장 없음 (시뮬레이션만)
- ✅ Shape 정보만으로 메모리/연산 모델링
- ✅ 실행 시간 추적 (nanoseconds)
- ✅ SRAM 사용률 관리
- ✅ 모듈별 파일 분리
- ✅ 계층적 구조 (PIM → Array → Area → Tile)

## 설정 가능 파라미터

### PIMSimulator
- `num_arrays`: eFlash Array 개수
- `area_execution_time_ns`: Area 실행 시간
- `array_sram_size_bytes`: 각 Array 내부 SRAM 크기
- `shared_sram_size_bytes`: 공유 SRAM 크기

### Area 제약
- `MAX_OUTPUT_DIM`: 128
- `MAX_REDUCTION_DIM`: 1280

## 향후 확장 가능 기능

- NPU 모듈 추가
- 데이터 전송 시간 모델링
- 에너지 소비 모델링
- 병렬 실행 스케줄링
- 실제 모델 (ResNet, Transformer 등) weight 배치 최적화
