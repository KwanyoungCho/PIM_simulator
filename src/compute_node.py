from typing import List, Dict, Tuple, Optional, Any


class ComputeNode:
    """연산 그래프의 노드 (Conv, Add, etc.)"""
    
    def __init__(self,
                 node_id: str,
                 node_type: str,
                 array_id: Optional[int] = None,
                 area_id: Optional[int] = None,
                 device_type: str = "eflash",
                 npu_id: Optional[int] = None,
                 weight_tiles: List[str] = None,
                 input_nodes: List[str] = None,
                 input_shape: Tuple[int, ...] = None,
                 output_shape: Tuple[int, ...] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Args:
            node_id: 노드 ID (예: "conv1", "add2")
            node_type: 노드 타입 ("conv", "add", "fc", etc.)
            array_id: weight가 배치된 eFlash Array ID (device_type="eflash"일 때)
            area_id: weight가 배치된 Area ID (device_type="eflash"일 때)
            device_type: 실행 장치 타입 ("eflash" or "npu")
            npu_id: NPU ID (device_type="npu"일 때)
            weight_tiles: 사용하는 weight tile ID 리스트
            input_nodes: 입력 노드 ID 리스트 (dependencies)
            input_shape: 입력 shape
            output_shape: 출력 shape
            metadata: 추가 메타데이터
        """
        self.node_id = node_id
        self.node_type = node_type
        self.device_type = device_type
        self.array_id = array_id
        self.area_id = area_id
        self.npu_id = npu_id
        self.weight_tiles = weight_tiles or []
        self.input_nodes = input_nodes or []
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.metadata = metadata or {}
        
        # 실행 상태
        self.is_ready = False
        self.is_running = False
        self.is_done = False
        self.start_time_us = None
        self.end_time_us = None
    
    def has_dependencies(self) -> bool:
        """의존성이 있는지 확인"""
        return len(self.input_nodes) > 0
    
    def mark_running(self, start_time_us: float):
        """실행 중 표시"""
        self.is_running = True
        self.start_time_us = start_time_us
    
    def mark_done(self, end_time_us: float):
        """실행 완료 표시"""
        self.is_running = False
        self.is_done = True
        self.end_time_us = end_time_us
    
    def get_execution_time(self) -> Optional[float]:
        """실행 시간 반환 (ns)"""
        if self.start_time_us is not None and self.end_time_us is not None:
            return self.end_time_us - self.start_time_us
        return None
    
    def reset(self):
        """상태 초기화"""
        self.is_ready = False
        self.is_running = False
        self.is_done = False
        self.start_time_us = None
        self.end_time_us = None
    
    def __repr__(self):
        deps_str = ",".join(self.input_nodes) if self.input_nodes else "none"
        status = "done" if self.is_done else ("running" if self.is_running else "pending")
        return (f"ComputeNode(id={self.node_id}, "
                f"type={self.node_type}, "
                f"array={self.array_id}, "
                f"area={self.area_id}, "
                f"deps=[{deps_str}], "
                f"status={status})")


class ComputeGraph:
    """연산 그래프"""
    
    def __init__(self):
        """초기화"""
        self.nodes: Dict[str, ComputeNode] = {}
        self.execution_order: List[str] = []
    
    def add_node(self, node: ComputeNode):
        """노드 추가"""
        self.nodes[node.node_id] = node
    
    def get_node(self, node_id: str) -> Optional[ComputeNode]:
        """노드 조회"""
        return self.nodes.get(node_id)
    
    def get_all_nodes(self) -> List[ComputeNode]:
        """모든 노드 반환"""
        return list(self.nodes.values())
    
    def get_ready_nodes(self, completed_nodes: set) -> List[ComputeNode]:
        """
        실행 가능한 노드들 반환 (dependency가 모두 완료된 노드)
        
        Args:
            completed_nodes: 완료된 노드 ID set
            
        Returns:
            실행 가능한 노드 리스트
        """
        ready = []
        for node in self.nodes.values():
            if node.is_done or node.is_running:
                continue
            
            # 모든 dependency가 완료되었는지 확인
            all_deps_done = all(dep in completed_nodes for dep in node.input_nodes)
            if all_deps_done:
                ready.append(node)
        
        return ready
    
    def get_source_nodes(self) -> List[ComputeNode]:
        """소스 노드들 반환 (dependency가 없는 노드)"""
        return [node for node in self.nodes.values() if not node.has_dependencies()]
    
    def reset_all(self):
        """모든 노드 상태 초기화"""
        for node in self.nodes.values():
            node.reset()
        self.execution_order.clear()
    
    def record_execution(self, node_id: str):
        """실행 순서 기록"""
        if node_id not in self.execution_order:
            self.execution_order.append(node_id)
    
    def get_stats(self) -> Dict:
        """통계 정보"""
        return {
            'total_nodes': len(self.nodes),
            'completed_nodes': sum(1 for n in self.nodes.values() if n.is_done),
            'running_nodes': sum(1 for n in self.nodes.values() if n.is_running),
            'pending_nodes': sum(1 for n in self.nodes.values() if not n.is_done and not n.is_running),
            'execution_order': self.execution_order.copy()
        }
    
    def __repr__(self):
        return f"ComputeGraph(nodes={len(self.nodes)})"
