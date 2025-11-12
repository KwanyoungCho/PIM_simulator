from typing import Dict, List, Optional, Tuple, Set
import heapq
from .event import Event, EventType
from .compute_node import ComputeGraph, ComputeNode
from .activation import ActivationManager, ActivationBuffer
from .pim_simulator import PIMSimulator


class InferenceScheduler:
    """Event-driven inference 스케줄러"""
    
    def __init__(self,
                 pim_simulator: PIMSimulator,
                 compute_graph: ComputeGraph,
                 shared_sram_bandwidth_gbps: float = 10.0):  # 10 GB/s
        """
        Args:
            pim_simulator: PIM 시뮬레이터
            compute_graph: 연산 그래프
            shared_sram_bandwidth_gbps: 공유 SRAM 대역폭 (GB/s)
        """
        self.pim = pim_simulator
        self.graph = compute_graph
        self.activation_manager = ActivationManager()
        
        # 대역폭 변환: GB/s -> bytes/ns
        self.sram_bandwidth_bytes_per_ns = shared_sram_bandwidth_gbps * 1e9 / 1e9  # = GB/s
        
        # 이벤트 큐 (우선순위 큐: 시간 순)
        self.event_queue: List[Event] = []
        
        # Array 사용 상태 추적
        self.array_busy_until: Dict[int, float] = {i: 0.0 for i in range(pim_simulator.num_arrays)}
        
        # 실행 상태
        self.current_time_ns = 0.0
        self.completed_nodes: set = set()
        self.running_nodes: Dict[str, Tuple[int, int]] = {}  # {node_id: (array_id, area_id)}
        
        # 개선: Activation lifetime 관리 (reference counting)
        self.activation_lifetimes: Dict[str, Dict] = {}  # {buffer_id: lifetime_info}
        
        # 통계
        self.total_compute_time_ns = 0.0
        self.total_transfer_time_ns = 0.0
        self.timeline: List[Event] = []
        self.deallocated_count = 0
        self.deallocated_bytes = 0
    
    def _add_event(self, event: Event):
        """이벤트 큐에 추가"""
        heapq.heappush(self.event_queue, event)
        self.timeline.append(event)
    
    def _calculate_transfer_time(self, data_size_bytes: int) -> float:
        """
        데이터 전송 시간 계산
        
        Args:
            data_size_bytes: 전송할 데이터 크기 (bytes)
            
        Returns:
            전송 시간 (ns)
        """
        return data_size_bytes / self.sram_bandwidth_bytes_per_ns
    
    def _determine_activation_locations(self,
                                        producer_node: ComputeNode,
                                        consumer_nodes: List[ComputeNode]) -> List[str]:
        """
        개선: Activation을 저장할 위치들 결정 (multi-location 지원)
        
        같은 array의 consumer와 다른 array의 consumer를 모두 고려하여
        최적 위치에 중복 저장
        
        Args:
            producer_node: 생성하는 노드
            consumer_nodes: 사용할 노드들
            
        Returns:
            저장 위치 리스트 ["shared_sram", "array_0_sram", ...]
        """
        if not consumer_nodes:
            return ["shared_sram"]
        
        producer_array = producer_node.array_id
        consumer_arrays = {node.array_id for node in consumer_nodes}
        
        locations = []
        
        # Case 1: 다른 array에서 사용? → shared_sram 필수
        has_different_array = any(arr != producer_array for arr in consumer_arrays)
        if has_different_array or len(consumer_nodes) > 1:
            locations.append("shared_sram")
        
        # Case 2: 같은 array에서도 사용? → 내부 SRAM에도 저장 (중복)
        if producer_array in consumer_arrays:
            internal = f"array_{producer_array}_sram"
            if internal not in locations:
                locations.append(internal)
        
        # Case 3: 같은 array만 사용, consumer 1개 → 내부 SRAM만
        if len(consumer_arrays) == 1 and producer_array in consumer_arrays and len(consumer_nodes) == 1:
            return [f"array_{producer_array}_sram"]
        
        # Fallback
        if not locations:
            locations.append("shared_sram")
        
        return locations
    
    def _get_consumer_nodes(self, producer_node_id: str) -> List[ComputeNode]:
        """특정 노드의 출력을 사용하는 consumer 노드들 반환"""
        consumers = []
        for node in self.graph.get_all_nodes():
            if producer_node_id in node.input_nodes:
                consumers.append(node)
        return consumers
    
    def _get_consumers_for_location(self, consumers: List[ComputeNode], location: str, producer_array_id: int) -> Set[str]:
        """
        특정 위치를 사용할 consumer들 반환
        
        Args:
            consumers: 모든 consumer 노드들
            location: 저장 위치
            producer_array_id: Producer의 array ID
            
        Returns:
            해당 위치를 사용할 consumer ID set
        """
        location_consumers = set()
        
        if location == "shared_sram":
            # Shared SRAM: 다른 array의 consumer들만
            for c in consumers:
                if c.array_id != producer_array_id:
                    location_consumers.add(c.node_id)
        else:
            # 내부 SRAM: 같은 array의 consumer들만
            arr_id = int(location.split("_")[1])
            for c in consumers:
                if c.array_id == arr_id:
                    location_consumers.add(c.node_id)
        
        return location_consumers
    
    def _get_optimal_location(self, buffer_id: str, consumer_array_id: int) -> str:
        """
        Consumer에게 최적 위치 반환
        
        Args:
            buffer_id: 버퍼 ID
            consumer_array_id: Consumer array ID
            
        Returns:
            최적 위치 이름
        """
        if buffer_id not in self.activation_lifetimes:
            return "shared_sram"
        
        lifetime = self.activation_lifetimes[buffer_id]
        storage_locations = lifetime['storage_locations']
        
        # 같은 array 내부 SRAM이 있으면 우선
        internal = f"array_{consumer_array_id}_sram"
        if internal in storage_locations and storage_locations[internal]['ref_count'] > 0:
            return internal
        
        # 없으면 shared SRAM
        return "shared_sram"
    
    def _deallocate_activation(self, buffer_id: str, location: str):
        """
        특정 위치에서만 activation 해제
        
        Args:
            buffer_id: 버퍼 ID
            location: 해제할 위치
        """
        if buffer_id not in self.activation_lifetimes:
            return
        
        lifetime = self.activation_lifetimes[buffer_id]
        
        if location not in lifetime['storage_locations']:
            return
        
        loc_info = lifetime['storage_locations'][location]
        freed_bytes = lifetime['buffer'].size_bytes
        
        print(f"  [Dealloc] {buffer_id} from {location} (location ref_count=0, freeing {freed_bytes/1024:.1f} KB)")
        
        # 해당 위치에서만 해제
        if loc_info['sram'].deallocate(buffer_id):
            print(f"    ✓ Freed from {location}")
        
        # 위치 정보 제거
        del lifetime['storage_locations'][location]
        
        # 통계
        self.deallocated_count += 1
        self.deallocated_bytes += freed_bytes
    
    def _schedule_compute(self, node: ComputeNode, start_time_ns: float):
        """
        연산 스케줄링 (Transfer → Compute 분리)
        
        Args:
            node: 실행할 노드
            start_time_ns: 스케줄링 시작 시간
        """
        array_id = node.array_id
        area_id = node.area_id
        
        # Transfer는 dependency만 만족하면 시작 (Array busy 무관)
        transfer_start_time = start_time_ns
        
        # 입력 데이터 전송 시간 계산
        transfer_time_ns = 0.0
        transfer_details = []  # 전송 상세 정보 저장
        
        for input_node_id in node.input_nodes:
            buffer_id = self.activation_manager.node_outputs.get(input_node_id)
            if buffer_id and buffer_id in self.activation_lifetimes:
                lifetime = self.activation_lifetimes[buffer_id]
                
                # 개선: 최적 위치 결정 (내부 SRAM 우선)
                optimal_location = self._get_optimal_location(buffer_id, array_id)
                
                if optimal_location == f"array_{array_id}_sram":
                    # 같은 array 내부 SRAM → 전송 시간 없음
                    transfer_details.append({
                        'buffer_id': buffer_id,
                        'location': optimal_location,
                        'size_bytes': lifetime['buffer'].size_bytes,
                        'time_ns': 0.0
                    })
                else:
                    # Shared SRAM에서 읽기 → 전송 시간 발생
                    xfer_time = self._calculate_transfer_time(lifetime['buffer'].size_bytes)
                    transfer_time_ns += xfer_time
                    transfer_details.append({
                        'buffer_id': buffer_id,
                        'location': optimal_location,
                        'size_bytes': lifetime['buffer'].size_bytes,
                        'time_ns': xfer_time
                    })
        
        if transfer_time_ns > 0:
            # Transfer 시간이 있으면 Transfer 이벤트 생성
            
            # 1. TRANSFER_START 이벤트 (Array busy 무관, dependency만 체크)
            transfer_start_event = Event(
                time_ns=transfer_start_time,
                event_type=EventType.TRANSFER_START,
                node_id=node.node_id,
                data={
                    'array_id': array_id,
                    'area_id': area_id,
                    'transfer_time_ns': transfer_time_ns,
                    'details': transfer_details
                }
            )
            self._add_event(transfer_start_event)
            
            # 2. TRANSFER_DONE 이벤트 (Compute 시작 트리거)
            transfer_done_time = transfer_start_time + transfer_time_ns
            transfer_done_event = Event(
                time_ns=transfer_done_time,
                event_type=EventType.TRANSFER_DONE,
                node_id=node.node_id,
                data={
                    'array_id': array_id,
                    'area_id': area_id,
                    'transfer_time_ns': transfer_time_ns,
                    'details': transfer_details
                }
            )
            self._add_event(transfer_done_event)
            
            # 통계: 전송 시간 누적
            self.total_transfer_time_ns += transfer_time_ns
        else:
            # Transfer 시간이 0이면 (내부 SRAM만 사용) 바로 Compute 시작
            
            # 전송 상세 로깅 (내부 SRAM 사용 표시)
            for detail in transfer_details:
                print(f"  [Transfer] {node.node_id} reads {detail['buffer_id']} from {detail['location']} ({detail['time_ns']:.2f} ns)")
            
            # Compute 시작 시간: Array available time 체크
            array_available_time = self.array_busy_until.get(array_id, 0.0)
            compute_start_time = max(transfer_start_time, array_available_time)
            
            compute_start_event = Event(
                time_ns=compute_start_time,
                event_type=EventType.COMPUTE_START,
                node_id=node.node_id,
                data={'array_id': array_id, 'area_id': area_id}
            )
            self._add_event(compute_start_event)
            
            # 연산 실행 시간
            array = self.pim.get_array(array_id)
            compute_time_ns = array.area_execution_time_ns
            
            # 연산 완료 시간
            compute_done_time = compute_start_time + compute_time_ns
            
            # 연산 완료 이벤트
            compute_done_event = Event(
                time_ns=compute_done_time,
                event_type=EventType.COMPUTE_DONE,
                node_id=node.node_id,
                data={'array_id': array_id, 'area_id': area_id}
            )
            self._add_event(compute_done_event)
            
            # Array busy 시간 업데이트
            self.array_busy_until[array_id] = compute_done_time
            
            # 통계 업데이트
            self.total_compute_time_ns += compute_time_ns
        
        # 노드를 running 상태로 표시 (Transfer 시작 시점 = dependency 완료 시점)
        node.mark_running(transfer_start_time)
        self.running_nodes[node.node_id] = (array_id, area_id)
    
    def _handle_transfer_start(self, event: Event):
        """
        전송 시작 이벤트 처리
        
        Args:
            event: TRANSFER_START 이벤트
        """
        node_id = event.node_id
        node = self.graph.get_node(node_id)
        
        if not node:
            return
        
        # 전송 시작 시간 기록
        node.transfer_start_time = event.time_ns
        
        # 전송 상세 정보 출력
        for detail in event.data['details']:
            print(f"  [Transfer] {node_id} reads {detail['buffer_id']} from {detail['location']} ({detail['time_ns']:.2f} ns)")
    
    def _handle_transfer_done(self, event: Event):
        """
        전송 완료 이벤트 처리
        
        Args:
            event: TRANSFER_DONE 이벤트
        """
        node_id = event.node_id
        node = self.graph.get_node(node_id)
        
        if not node:
            return
        
        # 전송 완료 시간 기록
        node.transfer_done_time = event.time_ns
        
        # 전송 시간 계산
        transfer_time_ns = event.data['transfer_time_ns']
        
        # 전송 완료 후 연산 시작
        array_id = event.data['array_id']
        area_id = event.data['area_id']
        
        # Array가 사용 가능한 시간
        array_available_time = self.array_busy_until.get(array_id, 0.0)
        actual_start_time = max(event.time_ns, array_available_time)
        
        # 연산 시작 이벤트
        compute_start_event = Event(
            time_ns=actual_start_time,
            event_type=EventType.COMPUTE_START,
            node_id=node_id,
            data={'array_id': array_id, 'area_id': area_id}
        )
        self._add_event(compute_start_event)
        
        # 연산 실행 시간
        array = self.pim.get_array(array_id)
        compute_time_ns = array.area_execution_time_ns
        
        # 연산 완료 시간
        compute_done_time = actual_start_time + compute_time_ns
        
        # 연산 완료 이벤트
        compute_done_event = Event(
            time_ns=compute_done_time,
            event_type=EventType.COMPUTE_DONE,
            node_id=node_id,
            data={'array_id': array_id, 'area_id': area_id}
        )
        self._add_event(compute_done_event)
        
        # Array busy 시간 업데이트
        self.array_busy_until[array_id] = compute_done_time
        
        # 노드 상태 업데이트
        node.mark_running(actual_start_time)
        self.running_nodes[node.node_id] = (array_id, area_id)
        
        # 통계 업데이트
        self.total_compute_time_ns += compute_time_ns
    
    def _handle_compute_start(self, event: Event):
        """
        연산 시작 이벤트 처리
        
        Args:
            event: COMPUTE_START 이벤트
        """
        node_id = event.node_id
        node = self.graph.get_node(node_id)
        
        if not node:
            return
        
        # 연산 시작 시간 기록
        node.compute_start_time = event.time_ns
        
        # 로깅 (선택적)
        # print(f"  [Compute] {node_id} starts at {event.time_ns:.2f} ns")
    
    def _handle_compute_done(self, event: Event):
        """
        연산 완료 이벤트 처리
        
        Args:
            event: COMPUTE_DONE 이벤트
        """
        node_id = event.node_id
        node = self.graph.get_node(node_id)
        
        if not node:
            return
        
        # 노드 완료 처리
        node.mark_done(event.time_ns)
        self.completed_nodes.add(node_id)
        self.graph.record_execution(node_id)  # 실행 순서 기록
        if node_id in self.running_nodes:
            del self.running_nodes[node_id]
        
        # 개선: 입력 activation 위치별 reference counting
        for input_node_id in node.input_nodes:
            buffer_id = self.activation_manager.node_outputs.get(input_node_id)
            if buffer_id and buffer_id in self.activation_lifetimes:
                lifetime = self.activation_lifetimes[buffer_id]
                
                # 전역 사용 완료 표시
                if node_id in lifetime['consumers']:
                    lifetime['used_by'].add(node_id)
                    lifetime['ref_count'] -= 1
                    
                    # 위치별 reference counting
                    for location, loc_info in list(lifetime['storage_locations'].items()):
                        if node_id in loc_info['consumers']:
                            # 이 위치를 사용한 consumer 완료!
                            loc_info['ref_count'] -= 1
                            
                            # 이 위치의 모든 consumer 완료? → 이 위치만 해제!
                            if loc_info['ref_count'] == 0:
                                self._deallocate_activation(buffer_id, location)
                    
                    # 모든 위치 해제 완료? → lifetime 제거
                    if not lifetime['storage_locations']:
                        del self.activation_lifetimes[buffer_id]
        
        # 출력 activation 생성
        if node.output_shape:
            buffer_id = f"{node_id}_output"
            
            # 개선: Consumer 노드들 확인하여 저장 위치들 결정 (multi-location)
            consumers = self._get_consumer_nodes(node_id)
            locations = self._determine_activation_locations(node, consumers)
            
            # Activation 버퍼 생성 (메타데이터만, 위치 정보는 storage_locations에서 관리)
            activation_buffer = ActivationBuffer(
                buffer_id=buffer_id,
                shape=node.output_shape,
                bytes_per_element=1
            )
            self.activation_manager.register_node_output(node_id, buffer_id)
            
            # Lifetime 정보 저장
            self.activation_lifetimes[buffer_id] = {
                'buffer': activation_buffer,
                'consumers': {c.node_id for c in consumers},
                'used_by': set(),
                'ref_count': len(consumers),
                'storage_locations': {}  # {location: {'sram': obj, 'consumers': set(), 'ref_count': int}}
            }
            
            # 개선: 여러 위치에 할당 + 위치별 consumer 추적
            print(f"  [Allocation] {buffer_id} → locations: {locations}")
            for location in locations:
                if location == "shared_sram":
                    sram = self.pim.get_shared_sram()
                else:
                    # "array_X_sram"
                    arr_id = int(location.split("_")[1])
                    sram = self.pim.get_array(arr_id).get_sram()
                
                if sram.allocate(buffer_id, activation_buffer.size_bytes):
                    # 이 위치를 사용할 consumer들 찾기
                    location_consumers = self._get_consumers_for_location(consumers, location, node.array_id)
                    
                    self.activation_lifetimes[buffer_id]['storage_locations'][location] = {
                        'sram': sram,
                        'consumers': location_consumers,
                        'ref_count': len(location_consumers)
                    }
                    print(f"    ✓ Allocated to {location} ({activation_buffer.size_bytes/1024:.1f} KB) for {location_consumers}")
        
        # 다음 실행 가능한 노드들 스케줄링
        self._schedule_ready_nodes(event.time_ns)
    
    def _schedule_ready_nodes(self, current_time_ns: float):
        """
        현재 시점에서 실행 가능한 노드들 스케줄링
        
        Args:
            current_time_ns: 현재 시간
        """
        ready_nodes = self.graph.get_ready_nodes(self.completed_nodes)
        
        for node in ready_nodes:
            if node.node_id not in self.running_nodes:
                self._schedule_compute(node, current_time_ns)
    
    def run_inference(self, input_batch_size: int, input_shape: Tuple[int, ...]) -> Dict:
        """
        Inference 실행
        
        Args:
            input_batch_size: 배치 크기
            input_shape: 입력 shape
            
        Returns:
            실행 결과 및 통계
        """
        # 초기화
        self.graph.reset_all()
        self.activation_manager.clear()
        self.event_queue.clear()
        self.array_busy_until = {i: 0.0 for i in range(self.pim.num_arrays)}
        self.completed_nodes.clear()
        self.running_nodes.clear()
        self.current_time_ns = 0.0
        self.total_compute_time_ns = 0.0
        self.total_transfer_time_ns = 0.0
        self.timeline.clear()
        
        # 개선: Activation lifetime 초기화
        self.activation_lifetimes.clear()
        self.deallocated_count = 0
        self.deallocated_bytes = 0
        
        # SRAM 초기화 (이전 inference의 activation 제거)
        self.pim.get_shared_sram().clear()
        for i in range(self.pim.num_arrays):
            self.pim.get_array(i).get_sram().clear()
        
        # 시작 이벤트
        start_event = Event(
            time_ns=0.0,
            event_type=EventType.INFERENCE_START,
            data={'batch_size': input_batch_size, 'input_shape': input_shape}
        )
        self._add_event(start_event)
        
        # 입력 activation 생성 (메타데이터만)
        input_buffer = self.activation_manager.create_buffer(
            buffer_id="input",
            shape=(input_batch_size, *input_shape),
            bytes_per_element=1
        )
        # 실제 할당은 shared SRAM에
        shared_sram = self.pim.get_shared_sram()
        self.activation_manager.allocate_to_sram("input", shared_sram, "shared_sram")
        
        # 소스 노드들(dependency 없는 노드) 스케줄링
        source_nodes = self.graph.get_source_nodes()
        for node in source_nodes:
            self._schedule_compute(node, 0.0)
        
        # 이벤트 처리 루프
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time_ns = event.time_ns
            
            if event.event_type == EventType.TRANSFER_START:
                self._handle_transfer_start(event)
            elif event.event_type == EventType.TRANSFER_DONE:
                self._handle_transfer_done(event)
            elif event.event_type == EventType.COMPUTE_START:
                self._handle_compute_start(event)
            elif event.event_type == EventType.COMPUTE_DONE:
                self._handle_compute_done(event)
            # 다른 이벤트 타입도 처리 가능
        
        # 종료 이벤트
        end_event = Event(
            time_ns=self.current_time_ns,
            event_type=EventType.INFERENCE_DONE,
            data={'total_nodes': len(self.graph.nodes)}
        )
        self.timeline.append(end_event)
        
        # 결과 반환
        return {
            'total_time_ns': self.current_time_ns,
            'total_compute_time_ns': self.total_compute_time_ns,
            'total_transfer_time_ns': self.total_transfer_time_ns,
            'completed_nodes': len(self.completed_nodes),
            'total_nodes': len(self.graph.nodes),
            'deallocated_count': self.deallocated_count,
            'deallocated_bytes': self.deallocated_bytes,
            'timeline': self.timeline.copy(),
            'graph_stats': self.graph.get_stats(),
            'activation_stats': self.activation_manager.get_stats()
        }
    
    def get_timeline(self) -> List[Event]:
        """실행 타임라인 반환"""
        return self.timeline.copy()
    
    def print_timeline(self):
        """타임라인 출력"""
        print("\n" + "=" * 80)
        print("INFERENCE TIMELINE")
        print("=" * 80)
        for event in self.timeline:
            print(f"[{event.time_ns:10.2f}ns] {event.event_type.value:20s} {event.node_id or ''}")
        print("=" * 80)
    
    def __repr__(self):
        return (f"InferenceScheduler(nodes={len(self.graph.nodes)}, "
                f"current_time={self.current_time_ns}ns)")
