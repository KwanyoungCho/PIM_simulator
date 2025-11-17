from typing import Dict, Any
from enum import Enum


class EventType(Enum):
    """이벤트 타입"""
    COMPUTE_START = "compute_start"
    COMPUTE_DONE = "compute_done"
    TRANSFER_START = "transfer_start"
    TRANSFER_DONE = "transfer_done"
    INFERENCE_START = "inference_start"
    INFERENCE_DONE = "inference_done"


class Event:
    """시뮬레이션 이벤트"""
    
    def __init__(self,
                 time_us: float,
                 event_type: EventType,
                 node_id: str = None,
                 data: Dict[str, Any] = None):
        """
        Args:
            time_us: 이벤트 발생 시간 (nanoseconds)
            event_type: 이벤트 타입
            node_id: 관련 노드 ID
            data: 추가 데이터
        """
        self.time_us = time_us
        self.event_type = event_type
        self.node_id = node_id
        self.data = data or {}
    
    def __lt__(self, other):
        """우선순위 큐를 위한 비교 연산자 (시간 순)"""
        return self.time_us < other.time_us
    
    def __repr__(self):
        return (f"Event(time={self.time_us}ns, "
                f"type={self.event_type.value}, "
                f"node={self.node_id})")
