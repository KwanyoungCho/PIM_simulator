from typing import Dict, List, Tuple, Optional
from .scheduler import InferenceScheduler
from .event import Event


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
    
    def execute(self) -> Dict:
        """
        Inference 실행
        
        Returns:
            실행 결과
        """
        self.result = self.scheduler.run_inference(
            self.input_batch_size,
            self.input_shape
        )
        
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
    
    # def get_stats(self) -> Dict:
    #     """통계 정보"""
    #     return {
    #         'context_id': self.context_id,
    #         'input_batch_size': self.input_batch_size,
    #         'input_shape': self.input_shape,
    #         'latency_us': self.get_latency(),
    #         'is_completed': self.is_completed,
    #         'result': self.result
    #     }
    
    def __repr__(self):
        status = "completed" if self.is_completed else "pending"
        latency = f"{self.get_latency():.2f}ns" if self.is_completed else "N/A"
        return (f"InferenceContext(id={self.context_id}, "
                f"batch={self.input_batch_size}, "
                f"status={status}, "
                f"latency={latency})")
