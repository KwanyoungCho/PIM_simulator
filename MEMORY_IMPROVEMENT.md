# PIM Simulator 메모리 관리 개선

## 개선 사항 요약

### 1. Multi-Location 저장
**문제:** 이전에는 activation을 하나의 위치에만 저장
```python
# Before
if all_same_array:
    location = "array_0_sram"  # 또는
else:
    location = "shared_sram"  # 둘 중 하나만

# 문제: conv1(Array0) → conv2(Array0), conv3(Array1)
# conv1_output이 shared_sram에만 저장
# conv2가 같은 array인데도 shared_sram에서 읽음 (불필요한 전송!)
```

**해결:** 여러 위치에 중복 저장
```python
# After
locations = _determine_activation_locations(producer, consumers)
# → ['shared_sram', 'array_0_sram']

# conv1_output을 두 곳에 저장:
# - shared_sram: conv3(Array1)가 사용
# - array_0_sram: conv2(Array0)가 사용 (전송 시간 0!)
```

### 2. Consumer별 최적 위치 선택
```python
def _get_optimal_location(buffer_id, consumer_array_id):
    # 1순위: 같은 array 내부 SRAM
    internal = f"array_{consumer_array_id}_sram"
    if internal in storage_locations:
        return internal  # 전송 시간 0!
    
    # 2순위: Shared SRAM
    return "shared_sram"  # 전송 시간 발생
```

### 3. Reference Counting으로 자동 해제
**문제:** 이전에는 activation이 한번도 해제되지 않음
```python
# Before
conv1 완료 → conv1_output 생성 (802 KB)
conv2 완료 → conv2_output 생성 (1600 KB)
conv3 완료 → conv3_output 생성 (784 KB)
...
Inference 끝 → SRAM 여전히 3.2 MB 차지! ❌
```

**해결:** 사용 완료되면 자동 해제
```python
# After
activation_lifetime = {
    'consumers': {'conv2', 'conv3'},
    'used_by': set(),
    'ref_count': 2
}

# conv2 완료 → ref_count: 1
# conv3 완료 → ref_count: 0 → 자동 해제! ✓
```

## 실제 실행 로그

### Multi-Location 할당
```
[Allocation] conv1_output → locations: ['shared_sram', 'array_0_sram']
  ✓ Allocated to shared_sram (784.0 KB)
  ✓ Allocated to array_0_sram (784.0 KB)
```

### 최적 위치에서 읽기
```
[Transfer] conv2a reads conv1_output from array_0_sram (0 ns)      ← Array0, 전송시간 없음!
[Transfer] conv2b reads conv1_output from shared_sram (80281.60 ns) ← Array1, 전송시간 발생
```

### Reference Counting 해제
```
[Dealloc] conv1_output (ref_count=0, freeing 784.0 KB)
  ✓ Freed from shared_sram
  ✓ Freed from array_0_sram
```

## 성능 비교

### Before (개선 전)
```
전송 시간: 200,704 ns
메모리 해제: 0개, 0 KB
최종 SRAM 사용: 1519 KB (계속 차지)
```

### After (개선 후)
```
전송 시간: 120,422 ns  ← 40% 감소!
메모리 해제: 5개, 2744 KB ✓
최종 SRAM 사용: 196 KB (conv4_output만)
```

## 전체 시나리오 예시

### 그래프
```
conv1(Array0) → conv2a(Array0)
             → conv2b(Array1) → conv3(Array1)
                             ↘
                              add1(Array0) → conv4(Array0)
                           ↗
                      conv2a
```

### 실행 흐름

#### 1. conv1 완료
```
[Allocation] conv1_output → ['shared_sram', 'array_0_sram']
  Consumers: {conv2a, conv2b}
  Ref count: 2
```

#### 2. conv2a 실행
```
[Transfer] conv2a reads conv1_output from array_0_sram (0 ns)  ← 같은 array!
  → ref_count: 1 (아직 conv2b가 안 씀)
```

#### 3. conv2b 실행
```
[Transfer] conv2b reads conv1_output from shared_sram (80281.60 ns)  ← 다른 array
  → ref_count: 0
[Dealloc] conv1_output (모든 위치에서 해제)
```

#### 4. add1 실행 (residual)
```
[Transfer] add1 reads conv2a_output from array_0_sram (0 ns)     ← 같은 array
[Transfer] add1 reads conv3_output from shared_sram (40140.80 ns) ← 다른 array
```

## 코드 변경 사항

### scheduler.py

1. **Lifetime 관리 추가**
```python
self.activation_lifetimes: Dict[str, Dict] = {
    'buffer': ActivationBuffer,
    'consumers': Set[str],
    'used_by': Set[str],
    'ref_count': int,
    'storage_locations': Dict[str, SRAMBuffer]
}
```

2. **Multi-location 결정**
```python
def _determine_activation_locations(producer, consumers) -> List[str]:
    locations = []
    
    # 다른 array 사용? → shared_sram
    if has_different_array:
        locations.append("shared_sram")
    
    # 같은 array 사용? → 내부 SRAM도 추가
    if producer_array in consumer_arrays:
        locations.append(f"array_{producer_array}_sram")
    
    return locations
```

3. **최적 위치 선택**
```python
def _get_optimal_location(buffer_id, consumer_array_id) -> str:
    # 같은 array 내부 SRAM 우선
    internal = f"array_{consumer_array_id}_sram"
    if internal in storage_locations:
        return internal
    return "shared_sram"
```

4. **Reference counting**
```python
def _handle_compute_done(event):
    # 입력 activation 사용 완료
    for input_node_id in node.input_nodes:
        lifetime['used_by'].add(node_id)
        lifetime['ref_count'] -= 1
        
        if lifetime['ref_count'] == 0:
            _deallocate_activation(buffer_id)  # 자동 해제!
```

5. **자동 해제**
```python
def _deallocate_activation(buffer_id):
    # 모든 저장 위치에서 해제
    for location, sram in storage_locations.items():
        sram.deallocate(buffer_id)
    
    del activation_lifetimes[buffer_id]
```

## 실행 결과 분석

### 메모리 절약
```
Before: 7개 activation 모두 유지 = 3087 KB
After:  최종 1개만 유지 = 196 KB
절약:   2891 KB (93.6% 감소)
```

### 전송 시간 절약
```
Before: 모든 읽기가 shared_sram에서 = 200,704 ns
After:  같은 array는 내부 SRAM = 120,422 ns
절약:   80,282 ns (40% 감소)
```

### Lifecycle 관리
```
conv1_output: 생성(0ns) → 사용(conv2a, conv2b) → 해제(200ns)
conv2a_output: 생성(200ns) → 사용(add1) → 해제(120822ns)
conv2b_output: 생성(80481ns) → 사용(conv3) → 해제(80581ns)
conv3_output: 생성(80581ns) → 사용(add1) → 해제(120822ns)
add1_output: 생성(120822ns) → 사용(conv4) → 해제(120922ns)
conv4_output: 생성(120922ns) → [유지] (출력)
```

## 실행 방법

```bash
cd /home/chokwans99/PIM_simulator
python example_inference.py
```

### 주요 출력
- **[Allocation]**: Activation이 어디에 저장되는지
- **[Transfer]**: 어디서 읽고 전송 시간은 얼마인지
- **[Dealloc]**: 언제 해제되는지

## 향후 개선 방향

1. **Dynamic placement**: Weight를 runtime에 재배치
2. **Prefetching**: 다음 필요한 activation 미리 로드
3. **Compression**: Activation 압축하여 전송 시간 감소
4. **Energy model**: 전송 에너지 소비량 계산
5. **Multi-batch**: 여러 배치를 겹쳐서 실행

## 참고 파일

- `src/scheduler.py`: 개선된 스케줄러
- `example_inference.py`: 실행 예제
- `src/memory_manager.py`: Reference counting 모듈 (참고용)
- `src/scheduler_improved.py`: 개선 예제 (참고용)
