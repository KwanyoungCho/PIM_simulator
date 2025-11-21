from typing import Dict, List, Tuple, Optional
from .scheduler import InferenceScheduler
from .event import Event
from .activation import ActivationManager


class InferenceContext:
    """단일 inference 실행 컨텍스트"""
    
    def __init__(self,
                 context_id: str,
                 scheduler: InferenceScheduler,
                 input_batch_size: int,
                 input_shape: Tuple[int, ...]):
        """
        Args:
            context_id: 컨텍스트 식별자
            scheduler: 스케줄러
            input_batch_size: 입력 배치 크기
            input_shape: 입력 shape
        """
        self.context_id = context_id
        self.scheduler = scheduler
        self.input_batch_size = input_batch_size
        self.input_shape = input_shape
        
        # 실행 결과
        self.result: Optional[Dict] = None
        self.start_time_us: Optional[float] = None
        self.end_time_us: Optional[float] = None
        self.is_completed = False
        self._reset_state()
    
    def _reset_state(self):
        """실행 상태 초기화"""
        pim = self.scheduler.pim
        self.activation_manager = ActivationManager()
        self.event_queue: List[Event] = []
        self.array_busy_until = {i: 0.0 for i in range(pim.num_arrays)}
        self.npu_busy_until = {i: 0.0 for i in range(pim.num_npus)}
        self.shared_sram_bus_busy_until = 0.0
        self.current_time_us = 0.0
        self.completed_nodes: set = set()
        self.running_nodes: Dict[str, Tuple[int, int]] = {}
        self.running_npu_nodes: Dict[str, int] = {}
        self.scheduled_nodes: set = set()
        self.activation_lifetimes: Dict[str, Dict] = {}
        self.total_compute_time_us = 0.0
        self.total_transfer_time_us = 0.0
        self.timeline: List[Event] = []
        self.deallocated_count = 0
        self.deallocated_bytes = 0
        self.memory_events: List[Dict] = []
        self.transfer_time_by_direction = {'read': 0.0, 'write': 0.0}
    
    def execute(self) -> Dict:
        """
        Inference 실행
        
        Returns:
            실행 결과
        """
        self._reset_state()
        self.result = self.scheduler.run_with_context(self)
        
        self.start_time_us = 0.0
        self.end_time_us = self.result['total_time_us']
        self.is_completed = True
        
        return self.result
    
    def get_latency(self) -> Optional[float]:
        """레이턴시 반환 (ns)"""
        if self.start_time_us is not None and self.end_time_us is not None:
            return self.end_time_us - self.start_time_us
        return None
    
    def get_timeline(self) -> List[Event]:
        """타임라인 반환"""
        return self.scheduler.get_timeline()
    
    def __repr__(self):
        status = "completed" if self.is_completed else "pending"
        latency = f"{self.get_latency():.2f}ns" if self.is_completed else "N/A"
        return (f"InferenceContext(id={self.context_id}, "
                f"batch={self.input_batch_size}, "
                f"status={status}, "
                f"latency={latency})")
