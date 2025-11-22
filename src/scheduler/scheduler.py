from typing import Dict, List, Optional, Tuple, Set
import heapq
from .event import Event, EventType
from ..graph import ComputeGraph, ComputeNode
from .activation import ActivationManager, ActivationBuffer
from ..hardware import PIMSimulator
from .scheduler_utils import TimelineFormatter
from .scheduler_components import ActivationPlanner


class InferenceScheduler:
    """Event-driven inference 스케줄러"""
    
    def __init__(self,
                 pim_simulator: PIMSimulator,
                 compute_graph: ComputeGraph,
                 shared_sram_bandwidth_kb_per_us: float = 3.2):  # 3.2 KB/us
        """
        Args:
            pim_simulator: PIM 시뮬레이터
            compute_graph: 연산 그래프
            shared_sram_bandwidth_kb_per_us: 공유 SRAM 대역폭 (KB/us)
        """
        self.pim = pim_simulator
        self.graph = compute_graph
        # 대역폭: KB/us -> bytes/us
        self.sram_bandwidth_bytes_per_us = shared_sram_bandwidth_kb_per_us * 1024
        self.transfer_epsilon = 1e-6  # shared bus ordering 보정
        self.ctx = None
        self.activation_planner = ActivationPlanner(self)

    def _bind_context(self, context):
        self.ctx = context

    def _release_context(self):
        self.ctx = None

    @property
    def activation_manager(self):
        return self.ctx.activation_manager

    @property
    def event_queue(self):
        return self.ctx.event_queue

    @property
    def array_busy_until(self):
        return self.ctx.array_busy_until

    @property
    def npu_busy_until(self):
        return self.ctx.npu_busy_until

    @property
    def shared_sram_bus_busy_until(self):
        return self.ctx.shared_sram_bus_busy_until

    @shared_sram_bus_busy_until.setter
    def shared_sram_bus_busy_until(self, value):
        self.ctx.shared_sram_bus_busy_until = value

    @property
    def current_time_us(self):
        return self.ctx.current_time_us

    @current_time_us.setter
    def current_time_us(self, value):
        self.ctx.current_time_us = value

    @property
    def completed_nodes(self):
        return self.ctx.completed_nodes

    @property
    def running_nodes(self):
        return self.ctx.running_nodes

    @property
    def running_npu_nodes(self):
        return self.ctx.running_npu_nodes

    @property
    def scheduled_nodes(self):
        return self.ctx.scheduled_nodes

    @property
    def activation_lifetimes(self):
        return self.ctx.activation_lifetimes

    @property
    def total_compute_time_us(self):
        return self.ctx.total_compute_time_us

    @total_compute_time_us.setter
    def total_compute_time_us(self, value):
        self.ctx.total_compute_time_us = value

    @property
    def total_transfer_time_us(self):
        return self.ctx.total_transfer_time_us

    @total_transfer_time_us.setter
    def total_transfer_time_us(self, value):
        self.ctx.total_transfer_time_us = value

    @property
    def timeline(self):
        return self.ctx.timeline

    @property
    def deallocated_count(self):
        return self.ctx.deallocated_count

    @deallocated_count.setter
    def deallocated_count(self, value):
        self.ctx.deallocated_count = value

    @property
    def deallocated_bytes(self):
        return self.ctx.deallocated_bytes

    @deallocated_bytes.setter
    def deallocated_bytes(self, value):
        self.ctx.deallocated_bytes = value

    @property
    def memory_events(self):
        return self.ctx.memory_events

    @property
    def transfer_time_by_direction(self):
        return self.ctx.transfer_time_by_direction
    
    def _add_event(self, event: Event):
        """이벤트 큐에 추가"""
        heapq.heappush(self.ctx.event_queue, event)
        self.ctx.timeline.append(event)
    
    def _calculate_transfer_time(self, data_size_bytes: int) -> float:
        """
        데이터 전송 시간 계산
        
        Args:
            data_size_bytes: 전송할 데이터 크기 (bytes)
            
        Returns:
            전송 시간 (us)
        """
        return data_size_bytes / self.sram_bandwidth_bytes_per_us
    
    def _calculate_flops(self, node: ComputeNode) -> int:
        """
        노드의 FLOP (Floating Point Operations) 계산
        
        Args:
            node: ComputeNode
            
        Returns:
            총 연산 수 (INT8 operations)
        """
        node_type = node.node_type
        
        if node_type == "fc" or node_type == "matmul":
            # Fully Connected / Matrix Multiplication
            # Input: (batch_size, input_features) or (input_features,)
            # Output: (batch_size, output_features) or (output_features,)
            # Weight: (input_features, output_features)
            # FLOP = 2 × batch_size × input_features × output_features
            
            input_features = node.input_shape[-1]  # 마지막 차원이 feature 수
            output_features = node.output_shape[-1]
            
            # Batch size (없으면 1)
            if len(node.input_shape) > 1:
                batch_size = node.input_shape[0]
            else:
                batch_size = 1
            
            flops = 2 * batch_size * input_features * output_features
            return flops
        
        elif node_type == "conv":
            # Convolution
            # Output: (C_out, H_out, W_out)
            # Input: (C_in, H_in, W_in)
            # Kernel: (C_out, C_in, K_h, K_w)
            # FLOP = 2 × C_out × H_out × W_out × C_in × K_h × K_w
            
            C_out, H_out, W_out = node.output_shape
            C_in, H_in, W_in = node.input_shape
            K_h = node.metadata.get('kernel_size', 3)
            K_w = K_h  # Assume square kernel
            
            flops = 2 * C_out * H_out * W_out * C_in * K_h * K_w
            return flops
        
        elif node_type in {"add", "concat", "maxpool", "upsample", "input"}:
            # Element-wise operations and zero-cost nodes
            return 0.1
        
        else:
            # 기타 연산
            return 0

    def _get_array_compute_time(self, node: ComputeNode, array_id: int) -> float:
        """
        eFlash array에서 노드를 실행할 때 걸리는 시간 계산.
        경량 연산(concat/reduce/add/maxpool/upsample/input)은 0으로 처리한다.
        """
        zero_cost_ops = {"concat", "reduce", "add", "maxpool", "upsample", "input"}
        if node.node_type in zero_cost_ops:
            return 0.01
        array = self.pim.get_array(array_id)
        return array.area_execution_time_us

    def _get_node_tag(self, node: ComputeNode) -> Optional[str]:
        """전처리로 생성된 노드 식별 태그 반환"""
        if not node.metadata.get("generated_by_preprocessor"):
            return None
        tag_map = {
            "tile": "[TILE]",
            "reduce": "[REDUCE]",
            "concat": "[P-CONCAT]",
        }
        pre_type = node.metadata.get("preprocessed_type")
        return tag_map.get(pre_type, "[P-NODE]")
    
    def _determine_activation_locations(self,
                                        producer_node: ComputeNode,
                                        consumer_nodes: List[ComputeNode]) -> List[str]:
        return self.activation_planner.determine_locations(producer_node, consumer_nodes)
    
    def _get_consumer_nodes(self, producer_node_id: str) -> List[ComputeNode]:
        """특정 노드의 출력을 사용하는 consumer 노드들 반환"""
        return self.activation_planner.get_consumer_nodes(producer_node_id)

    def _get_consumers_for_location(self,
                                    consumers: List[ComputeNode],
                                    location: str,
                                    producer_array_id: Optional[int]) -> Set[str]:
        """
        특정 위치를 사용할 consumer들 반환
        
        Args:
            consumers: 모든 consumer 노드들
            location: 저장 위치
            producer_array_id: Producer의 array ID
            
        Returns:
            해당 위치를 사용할 consumer ID set
        """
        return self.activation_planner.consumers_for_location(consumers, location, producer_array_id)
    
    def _get_optimal_location(self, buffer_id: str, consumer_array_id: int) -> str:
        """
        Consumer에게 최적 위치 반환
        
        Args:
            buffer_id: 버퍼 ID
            consumer_array_id: Consumer array ID
            
        Returns:
            최적 위치 이름
        """
        return self.activation_planner.optimal_location(buffer_id, consumer_array_id)
    
    def _deallocate_activation(self, buffer_id: str, location: str):
        """
        특정 위치에서만 activation 해제
        
        Args:
            buffer_id: 버퍼 ID
            location: 해제할 위치
        """
        self.activation_planner.deallocate_activation(buffer_id, location)
    
    def _record_array_copy(self, buffer_id: str, array_id: int, event_time_us: float):
        """
        Shared SRAM에서 array로 데이터를 전송한 후, 해당 array SRAM에도 buffer를 기록한다.
        """
        self.activation_planner.record_array_copy(buffer_id, array_id, event_time_us)

    def _schedule_shared_write(self,
                               node: ComputeNode,
                               buffer_id: str,
                               activation_buffer: ActivationBuffer,
                               event_time_us: float,
                               node_tag: Optional[str] = None):
        self.activation_planner.schedule_shared_write(node, buffer_id, activation_buffer, event_time_us, node_tag)

    def _complete_shared_write(self, node: ComputeNode, event: Event):
        self.activation_planner.complete_shared_write(node, event)

    def _finalize_node_completion(self,
                                  node: ComputeNode,
                                  completion_time_us: float,
                                  schedule_ready: bool):
        self.activation_planner.finalize_node_completion(node, completion_time_us, schedule_ready)

    def _schedule_npu_compute(self, node: ComputeNode, start_time_us: float):
        """
        NPU 연산 스케줄링 (TOPS 기반)
        
        Args:
            node: 실행할 NPU 노드
            start_time_us: 스케줄링 시작 시간 (us)
        """
        npu_id = node.npu_id
        node_tag = self._get_node_tag(node)
        npu = self.pim.get_npu(npu_id)
        
        # Transfer 시작 시간
        transfer_start_time = start_time_us
        
        # 입력 데이터 전송 시간 계산
        transfer_time_us = 0.0
        transfer_details = []
        
        for input_node_id in node.input_nodes:
            buffer_id = self.activation_manager.node_outputs.get(input_node_id)
            if buffer_id and buffer_id in self.activation_lifetimes:
                lifetime = self.ctx.activation_lifetimes[buffer_id]
                
                # NPU SRAM에 있으면 전송 없음
                if f"npu_{npu_id}_sram" in lifetime['storage_locations']:
                    transfer_details.append({
                        'buffer_id': buffer_id,
                        'location': f"npu_{npu_id}_sram",
                        'size_bytes': lifetime['buffer'].size_bytes,
                        'time_us': 0.0
                    })
                else:
                    # Shared SRAM에서 전송
                    xfer_time = self._calculate_transfer_time(lifetime['buffer'].size_bytes)
                    transfer_time_us += xfer_time
                    transfer_details.append({
                        'buffer_id': buffer_id,
                        'location': 'shared_sram',
                        'size_bytes': lifetime['buffer'].size_bytes,
                        'time_us': xfer_time
                    })
        
        if transfer_time_us > 0:
            # Transfer 이벤트 생성 (Shared SRAM에서 읽기)
            transfer_start_time = max(start_time_us, self.shared_sram_bus_busy_until + self.transfer_epsilon)
            
            start_data = {
                'npu_id': npu_id,
                'transfer_time_us': transfer_time_us,
                'details': transfer_details,
                'uses_shared_sram': True,
                'transfer_phase': 'shared_read',
                'transfer_direction': 'read'
            }
            if node_tag:
                start_data['node_tag'] = node_tag
            transfer_start_event = Event(
                time_us=transfer_start_time,
                event_type=EventType.TRANSFER_START,
                node_id=node.node_id,
                data=start_data
            )
            self._add_event(transfer_start_event)
            
            # Bus busy 예약
            self.shared_sram_bus_busy_until = transfer_start_time + transfer_time_us
        else:
            # Transfer 없으면 바로 Compute
            pass
            
            # NPU available time 체크
            npu_available_time = self.npu_busy_until.get(npu_id, 0.0)
            compute_start_time = max(transfer_start_time, npu_available_time)
            
            # FLOP 계산 및 실행 시간
            flops = self._calculate_flops(node)
            compute_time_us = npu.calculate_execution_time(flops)
            
            compute_start_data = {'npu_id': npu_id, 'flops': flops, 'compute_time_us': compute_time_us}
            if node_tag:
                compute_start_data['node_tag'] = node_tag
            compute_start_event = Event(
                time_us=compute_start_time,
                event_type=EventType.COMPUTE_START,
                node_id=node.node_id,
                data=compute_start_data
            )
            self._add_event(compute_start_event)
            
            compute_done_time = compute_start_time + compute_time_us
            compute_done_data = {'npu_id': npu_id}
            if node_tag:
                compute_done_data['node_tag'] = node_tag
            compute_done_event = Event(
                time_us=compute_done_time,
                event_type=EventType.COMPUTE_DONE,
                node_id=node.node_id,
                data=compute_done_data
            )
            self._add_event(compute_done_event)
            
            # NPU busy until 업데이트
            self.npu_busy_until[npu_id] = compute_done_time
            self.total_compute_time_us += compute_time_us
    
    def _schedule_compute(self, node: ComputeNode, start_time_us: float):
        """
        연산 스케줄링 (Transfer → Compute 분리)
        
        Args:
            node: 실행할 노드
            start_time_us: 스케줄링 시작 시간 (us)
        """
        # 이미 스케줄링된 노드는 중복 방지
        if node.node_id in self.scheduled_nodes:
            return
        
        node_tag = self._get_node_tag(node)

        # Device type에 따라 분기
        if node.device_type == "npu":
            self.scheduled_nodes.add(node.node_id)
            self._schedule_npu_compute(node, start_time_us)
            return
        
        # eFlash Array 연산
        array_id = node.array_id
        area_id = node.area_id
        
        # 입력 데이터 전송 시간 계산
        transfer_time_us = 0.0
        transfer_details = []  # 전송 상세 정보 저장
        uses_shared_sram = False  # Shared SRAM 사용 여부
        
        # Shared SRAM에서 읽어야 하는데 아직 준비 안된 경우 체크
        for input_node_id in node.input_nodes:
            buffer_id = self.activation_manager.node_outputs.get(input_node_id)
            if buffer_id and buffer_id in self.activation_lifetimes:
                lifetime = self.activation_lifetimes[buffer_id]
                optimal_location = self._get_optimal_location(buffer_id, array_id)
                
                if optimal_location == "shared_sram":
                    loc_info = lifetime['storage_locations'].get('shared_sram', {})
                    if not loc_info.get('is_ready', False):
                        # Shared SRAM write가 아직 완료되지 않음 → 스케줄링 불가
                        return
        
        # 여기까지 왔으면 스케줄링 가능
        self.scheduled_nodes.add(node.node_id)
        
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
                        'time_us': 0.0
                    })
                else:
                    # Shared SRAM에서 읽기 → 전송 시간 발생
                    xfer_time = self._calculate_transfer_time(lifetime['buffer'].size_bytes)
                    transfer_time_us += xfer_time
                    uses_shared_sram = True  # Shared SRAM 사용 표시
                    transfer_details.append({
                        'buffer_id': buffer_id,
                        'location': optimal_location,
                        'size_bytes': lifetime['buffer'].size_bytes,
                        'time_us': xfer_time
                    })
        
        if transfer_time_us > 0:
            # Transfer 시간이 있으면 Transfer 이벤트 생성 (Shared SRAM에서 읽기)
            
            # Transfer 시작 시간: dependency + Shared SRAM bus busy 고려
            transfer_start_time = max(start_time_us, self.shared_sram_bus_busy_until + self.transfer_epsilon)
            
            transfer_start_data = {
                'array_id': array_id,
                'area_id': area_id,
                'transfer_time_us': transfer_time_us,
                'details': transfer_details,
                'uses_shared_sram': uses_shared_sram,
                'transfer_phase': 'shared_read',
                'transfer_direction': 'read'
            }
            if node_tag:
                transfer_start_data['node_tag'] = node_tag
            transfer_start_event = Event(
                time_us=transfer_start_time,
                event_type=EventType.TRANSFER_START,
                node_id=node.node_id,
                data=transfer_start_data
            )
            self._add_event(transfer_start_event)
            
            # Bus busy 예약 (transfer_time_us > 0이므로 항상 Shared SRAM 사용)
            self.shared_sram_bus_busy_until = transfer_start_time + transfer_time_us
        else:
            # Transfer 시간이 0이면 (내부 SRAM만 사용) 바로 Compute 시작
            
            # Compute 시작 시간: dependency + Array available time 체크
            array_available_time = self.array_busy_until.get(array_id, 0.0)
            compute_start_time = max(start_time_us, array_available_time)
            
            compute_start_data = {'array_id': array_id, 'area_id': area_id}
            if node_tag:
                compute_start_data['node_tag'] = node_tag
            compute_start_event = Event(
                time_us=compute_start_time,
                event_type=EventType.COMPUTE_START,
                node_id=node.node_id,
                data=compute_start_data
            )
            self._add_event(compute_start_event)
            
            # 연산 시간
            compute_time_us = self._get_array_compute_time(node, array_id)
            compute_done_time = compute_start_time + compute_time_us
            
            # 연산 완료 이벤트
            compute_done_data = {'array_id': array_id, 'area_id': area_id}
            if node_tag:
                compute_done_data['node_tag'] = node_tag
            compute_done_event = Event(
                time_us=compute_done_time,
                event_type=EventType.COMPUTE_DONE,
                node_id=node.node_id,
                data=compute_done_data
            )
            self._add_event(compute_done_event)
            
            # Array busy 시간 업데이트
            self.array_busy_until[array_id] = compute_done_time
            
            # 통계 업데이트
            self.total_compute_time_us += compute_time_us
            
            # 노드를 running 상태로 표시 (실제 Compute 시작 시간 사용)
            node.mark_running(compute_start_time)
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
            print(f"[Error] Node {node_id} not found")
            return
        
        # shared_read만 노드의 시작 (혹은 compute_start)
        if event.data.get('transfer_phase') != 'shared_write':
            node.mark_running(event.time_us)
        
        # running_nodes에 추가 (중복 스케줄링 방지)
        array_id = event.data.get('array_id')
        area_id = event.data.get('area_id')
        if array_id is not None and area_id is not None:
            self.running_nodes[node_id] = (array_id, area_id)
        
        # 실제 전송 시간 계산 및 detail 업데이트
        transfer_details = event.data.get('details', [])
        transfer_phase = event.data.get('transfer_phase', 'shared_read')
        array_id = event.data.get('array_id')
        
        actual_transfer_time = 0.0
        updated_details = []
        
        # shared_read만 optimal location 재확인 필요
        need_recheck = (transfer_phase == 'shared_read' and array_id is not None)
        
        for detail in transfer_details:
            detail_time = detail.get('time_us', 0.0)
            
            # Optimal location 재확인 (shared_read + eFlash만)
            if need_recheck:
                buffer_id = detail.get('buffer_id')
                if buffer_id:
                    optimal = self._get_optimal_location(buffer_id, array_id)
                    if optimal == f"array_{array_id}_sram" and detail.get('location') != optimal:
                        # 위치 변경: shared → 내부 SRAM으로 이동됨
                        detail = {'buffer_id': buffer_id, 'location': optimal, 
                                  'size_bytes': detail.get('size_bytes', 0), 'time_us': 0.0}
                        detail_time = 0.0
            
            actual_transfer_time += detail_time
            updated_details.append(detail)
        
        done_time = event.time_us + actual_transfer_time
        
        # TRANSFER_DONE 데이터 구성
        done_data = {
            'transfer_time_us': actual_transfer_time,
            'details': updated_details,
            'transfer_phase': transfer_phase,
            'transfer_direction': event.data.get('transfer_direction', 'read')
        }
        
        # Device별 추가 정보
        if node.device_type == "npu":
            done_data['npu_id'] = event.data.get('npu_id')
        else:
            done_data['array_id'] = array_id
            done_data['area_id'] = event.data.get('area_id')
            if event.data.get('buffer_id'):
                done_data['buffer_id'] = event.data['buffer_id']
                done_data['location'] = event.data.get('location')
        
        # Shared SRAM bus busy는 스케줄링 시점에 이미 예약됨
        # 실행 시점에 다시 업데이트하면 타임스탬프 역전 발생!
        # if event.data.get('uses_shared_sram') and actual_transfer_time > 0:
        #     self.shared_sram_bus_busy_until = done_time
        
        if event.data.get('node_tag'):
            done_data['node_tag'] = event.data['node_tag']
        
        transfer_done_event = Event(
            time_us=done_time,
            event_type=EventType.TRANSFER_DONE,
            node_id=node_id,
            data=done_data
        )
        self._add_event(transfer_done_event)
        
        # 전송 시간 통계
        self.total_transfer_time_us += actual_transfer_time
        direction = event.data.get('transfer_direction', 'read')
        self.transfer_time_by_direction[direction] = \
            self.transfer_time_by_direction.get(direction, 0.0) + actual_transfer_time
    
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
        node_tag = self._get_node_tag(node)
        
        # 전송 시간 계산
        transfer_time_us = event.data['transfer_time_us']
        transfer_phase = event.data.get('transfer_phase')
        
        if transfer_phase == 'shared_write':
            self._complete_shared_write(node, event)
            return
        
        # Device type에 따라 분기
        if node.device_type == "npu":
            # NPU 연산
            npu_id = event.data.get('npu_id') or node.npu_id
            npu = self.pim.get_npu(npu_id)
            
            npu_available_time = self.npu_busy_until.get(npu_id, 0.0)
            actual_start_time = max(event.time_us, npu_available_time)
            
            # FLOP 계산
            flops = self._calculate_flops(node)
            compute_time_us = npu.calculate_execution_time(flops)
            
            compute_start_data = {'npu_id': npu_id, 'flops': flops, 'compute_time_us': compute_time_us}
            if node_tag:
                compute_start_data['node_tag'] = node_tag
            compute_start_event = Event(
                time_us=actual_start_time,
                event_type=EventType.COMPUTE_START,
                node_id=node_id,
                data=compute_start_data
            )
            self._add_event(compute_start_event)
            
            compute_done_time = actual_start_time + compute_time_us
            compute_done_data = {'npu_id': npu_id}
            if node_tag:
                compute_done_data['node_tag'] = node_tag
            compute_done_event = Event(
                time_us=compute_done_time,
                event_type=EventType.COMPUTE_DONE,
                node_id=node_id,
                data=compute_done_data
            )
            self._add_event(compute_done_event)
            
            self.npu_busy_until[npu_id] = compute_done_time
            self.running_npu_nodes[node.node_id] = npu_id
        else:
            # eFlash Array 연산
            array_id = event.data['array_id']
            area_id = event.data['area_id']
            
            # Array가 사용 가능한 시간
            array_available_time = self.array_busy_until.get(array_id, 0.0)
            actual_start_time = max(event.time_us, array_available_time)
            
            # 연산 시작 이벤트
            compute_start_data = {'array_id': array_id, 'area_id': area_id}
            if node_tag:
                compute_start_data['node_tag'] = node_tag
            compute_start_event = Event(
                time_us=actual_start_time,
                event_type=EventType.COMPUTE_START,
                node_id=node_id,
                data=compute_start_data
            )
            self._add_event(compute_start_event)
            
            # 연산 실행 시간
            compute_time_us = self._get_array_compute_time(node, array_id)
            
            # 연산 완료 시간
            compute_done_time = actual_start_time + compute_time_us
            
            # 연산 완료 이벤트
            compute_done_data = {'array_id': array_id, 'area_id': area_id}
            if node_tag:
                compute_done_data['node_tag'] = node_tag
            compute_done_event = Event(
                time_us=compute_done_time,
                event_type=EventType.COMPUTE_DONE,
                node_id=node_id,
                data=compute_done_data
            )
            self._add_event(compute_done_event)
            
            transfer_details = event.data.get('details', [])
            if transfer_details:
                for detail in transfer_details:
                    buffer_id = detail.get('buffer_id')
                    if not buffer_id:
                        continue
                    if detail.get('location') == f"array_{array_id}_sram":
                        continue
                    self._record_array_copy(buffer_id, array_id, event.time_us)
            
            # Array busy 시간 업데이트
            self.array_busy_until[array_id] = compute_done_time
            self.running_nodes[node.node_id] = (array_id, area_id)
        
        # 통계 업데이트
        self.total_compute_time_us += compute_time_us
    
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
        
        # 로깅 (선택적)
        # print(f"  [Compute] {node_id} starts at {event.time_us:.2f} us")
    
    def _handle_compute_done(self, event: Event):
        """
        연산 완료 이벤트 처리 (Tile 완료 포함)
        
        Args:
            event: COMPUTE_DONE 이벤트
        """
        node_id = event.node_id

        # 동일 노드가 중복 처리되지 않도록 보호
        if node_id in self.completed_nodes:
            return
        
        node = self.graph.get_node(node_id)
        
        if not node:
            return
        node_tag = self._get_node_tag(node)

        # 실행 상태 업데이트
        if node_id in self.running_nodes:
            del self.running_nodes[node_id]
        if node_id in self.running_npu_nodes:
            del self.running_npu_nodes[node_id]
        
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
            
            # Consumer 노드들 확인하여 저장 위치들 결정
            consumers = self._get_consumer_nodes(node_id)
            locations = self._determine_activation_locations(node, consumers)
            
            
            # Activation 버퍼 생성
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
                'storage_locations': {}
            }
            
            # 여러 위치에 할당 + 위치별 consumer 추적
            for location in locations:
                if location == "shared_sram":
                    sram = self.pim.get_shared_sram()
                elif location.startswith("npu_"):
                    npu_id = int(location.split("_")[1])
                    sram = self.pim.get_npu(npu_id).get_sram()
                else:  # array_X_sram
                    arr_id = int(location.split("_")[1])
                    sram = self.pim.get_array(arr_id).get_sram()
                
                # 이 위치를 실제로 사용할 consumer가 없으면 건너뜀
                producer_device_id = (node.device_type, 
                                     node.npu_id if node.device_type == "npu" else node.array_id)
                location_consumers = self._get_consumers_for_location(consumers, location, producer_device_id)
                if not location_consumers:
                    continue
                
                if location == "shared_sram":
                    # Shared SRAM write는 별도 transfer 이벤트로 진행
                    storage_entry = {
                        'sram': sram,
                        'consumers': location_consumers,
                        'ref_count': len(location_consumers),
                        'is_ready': False
                    }
                    self.activation_lifetimes[buffer_id]['storage_locations'][location] = storage_entry
                    self._schedule_shared_write(node, buffer_id, activation_buffer, event.time_us, node_tag)
                else:
                    # 내부 SRAM에는 즉시 할당 (compute 완료 시점)
                    alloc_success = sram.allocate(buffer_id, activation_buffer.size_bytes, warn=False)
                    if not alloc_success:
                        # 내부 SRAM 할당 실패: shared SRAM fallback 없이 경고만 출력
                        print(f"[ERROR] Internal SRAM allocation failed for {buffer_id} ({activation_buffer.size_bytes/1024:.2f} KB)")
                        print(f"        Location: {location}, Node: {node_id}")
                        print(f"        SRAM: {sram.name}, Used: {sram.used_bytes/1024:.2f} KB / {sram.size_bytes/1024:.2f} KB")
                        print(f"        Note: Shared SRAM fallback is disabled. Data must fit in internal SRAM.")
                        continue
                    
                    storage_entry = {
                        'sram': sram,
                        'consumers': location_consumers,
                        'ref_count': len(location_consumers),
                        'is_ready': True
                    }
                    self.activation_lifetimes[buffer_id]['storage_locations'][location] = storage_entry
                    self.memory_events.append({
                        'time_us': event.time_us,
                        'type': 'alloc',
                        'buffer': buffer_id,
                        'location': location,
                        'size_kb': activation_buffer.size_bytes / 1024,
                        'consumers': list(location_consumers)
                    })
        
        # Compute 완료 즉시 노드 완료 처리 (shared write와 독립적)
        self._finalize_node_completion(node, event.time_us, schedule_ready=True)

    def _schedule_ready_nodes(self, current_time_us: float):
        """
        현재 시점에서 실행 가능한 노드들 스케줄링
        
        Args:
            current_time_us: 현재 시간
        """
        ready_nodes = self.graph.get_ready_nodes(self.completed_nodes)
        
        for node in ready_nodes:
            # 이미 실행 중인 노드는 스케줄링하지 않음
            if node.node_id not in self.running_nodes and node.node_id not in self.running_npu_nodes:
                self._schedule_compute(node, current_time_us)
    
    def run_with_context(self, context) -> Dict:
        """주어진 컨텍스트로 inference 실행"""
        self._bind_context(context)
        try:
            return self._run_context()
        finally:
            self._release_context()

    def _run_context(self) -> Dict:
        ctx = self.ctx
        self.graph.reset_all()
        
        # SRAM 초기화 (이전 inference의 activation 제거)
        self.pim.get_shared_sram().clear()
        for i in range(self.pim.num_arrays):
            self.pim.get_array(i).get_sram().clear()
        
        # 시작 이벤트
        start_event = Event(
            time_us=0.0,
            event_type=EventType.INFERENCE_START,
            data={'batch_size': ctx.input_batch_size, 'input_shape': ctx.input_shape}
        )
        self._add_event(start_event)
        
        # 입력 activation 생성 (메타데이터만)
        self.activation_manager.create_buffer(
            buffer_id="input",
            shape=(ctx.input_batch_size, *ctx.input_shape),
            bytes_per_element=1
        )
        shared_sram = self.pim.get_shared_sram()
        self.activation_manager.allocate_to_sram("input", shared_sram)
        
        # 소스 노드들(dependency 없는 노드) 스케줄링
        source_nodes = self.graph.get_source_nodes()
        for node in source_nodes:
            self._schedule_compute(node, 0.0)
        
        # 이벤트 처리 루프
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time_us = event.time_us
            
            if event.event_type == EventType.TRANSFER_START:
                self._handle_transfer_start(event)
            elif event.event_type == EventType.TRANSFER_DONE:
                self._handle_transfer_done(event)
            elif event.event_type == EventType.COMPUTE_START:
                self._handle_compute_start(event)
            elif event.event_type == EventType.COMPUTE_DONE:
                self._handle_compute_done(event)
                self._schedule_ready_nodes(self.current_time_us)
        
        end_event = Event(
            time_us=self.current_time_us,
            event_type=EventType.INFERENCE_DONE,
            data={'total_nodes': len(self.graph.nodes)}
        )
        self.timeline.append(end_event)
        
        result = {
            'total_time_us': self.current_time_us,
            'total_compute_time_us': self.total_compute_time_us,
            'total_transfer_time_us': self.total_transfer_time_us,
            'transfer_time_by_direction': self.transfer_time_by_direction.copy(),
            'completed_nodes': len(self.completed_nodes),
            'total_nodes': len(self.graph.nodes),
            'deallocated_count': self.deallocated_count,
            'deallocated_bytes': self.deallocated_bytes,
            'timeline': self.timeline.copy(),
            'memory_events': self.memory_events.copy(),
            'graph_stats': self.graph.get_stats(),
            'activation_stats': self.activation_manager.get_stats()
        }
        return result
    
    def get_timeline(self) -> List[Event]:
        """실행 타임라인 반환"""
        return self.timeline.copy()
    
    def print_timeline(self, max_nodes=50, timeline=None):
        """노드별 실행 Timeline 출력"""
        data = timeline if timeline is not None else (self.ctx.timeline if self.ctx else [])
        TimelineFormatter.format_node_timeline(data, max_nodes)
    
    def print_memory_timeline(self, max_events=50, location_filter=None, events=None):
        """메모리 할당/해제 Timeline 출력"""
        data = events if events is not None else (self.ctx.memory_events if self.ctx else [])
        TimelineFormatter.format_memory_timeline(data, max_events, location_filter)
    
    def print_overlap_summary(self, min_duration_us=0.0, max_segments=10, show_all_segments=False, timeline=None):
        """연산/전송 중첩 구간 요약"""
        data = timeline if timeline is not None else (self.ctx.timeline if self.ctx else [])
        TimelineFormatter.print_overlap_summary(
            data,
            min_duration_us=min_duration_us,
            max_segments=max_segments,
            show_all_segments=show_all_segments
        )
    
    def __repr__(self):
        return (f"InferenceScheduler(nodes={len(self.graph.nodes)}, "
                f"current_time={self.current_time_us}us)")
