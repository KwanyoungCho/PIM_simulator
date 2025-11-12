from typing import Dict, List, Tuple, Optional
from .scheduler import InferenceScheduler
from .compute_node import ComputeGraph
from .event import Event


class InferenceContext:
    """단일 inference 실행 컨텍스트 (pipeline 지원)"""
    
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
        self.start_time_ns: Optional[float] = None
        self.end_time_ns: Optional[float] = None
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
        
        self.start_time_ns = 0.0
        self.end_time_ns = self.result['total_time_ns']
        self.is_completed = True
        
        return self.result
    
    def get_latency(self) -> Optional[float]:
        """레이턴시 반환 (ns)"""
        if self.start_time_ns is not None and self.end_time_ns is not None:
            return self.end_time_ns - self.start_time_ns
        return None
    
    def get_timeline(self) -> List[Event]:
        """타임라인 반환"""
        return self.scheduler.get_timeline()
    
    def get_stats(self) -> Dict:
        """통계 정보"""
        return {
            'context_id': self.context_id,
            'input_batch_size': self.input_batch_size,
            'input_shape': self.input_shape,
            'latency_ns': self.get_latency(),
            'is_completed': self.is_completed,
            'result': self.result
        }
    
    def __repr__(self):
        status = "completed" if self.is_completed else "pending"
        latency = f"{self.get_latency():.2f}ns" if self.is_completed else "N/A"
        return (f"InferenceContext(id={self.context_id}, "
                f"batch={self.input_batch_size}, "
                f"status={status}, "
                f"latency={latency})")


class PipelineManager:
    """여러 inference를 pipeline으로 관리"""
    
    def __init__(self, scheduler: InferenceScheduler):
        """
        Args:
            scheduler: 공유 스케줄러
        """
        self.scheduler = scheduler
        self.contexts: List[InferenceContext] = []
        self.completed_contexts: List[InferenceContext] = []
    
    def add_inference(self,
                      context_id: str,
                      input_batch_size: int,
                      input_shape: Tuple[int, ...]) -> InferenceContext:
        """
        새로운 inference 추가
        
        Args:
            context_id: 컨텍스트 ID
            input_batch_size: 배치 크기
            input_shape: 입력 shape
            
        Returns:
            생성된 InferenceContext
        """
        context = InferenceContext(
            context_id=context_id,
            scheduler=self.scheduler,
            input_batch_size=input_batch_size,
            input_shape=input_shape
        )
        self.contexts.append(context)
        return context
    
    def run_sequential(self) -> List[Dict]:
        """
        순차 실행 (각 inference가 완료된 후 다음 시작)
        
        Returns:
            각 inference 결과 리스트
        """
        results = []
        for context in self.contexts:
            result = context.execute()
            results.append(result)
            self.completed_contexts.append(context)
        
        return results
    
    def get_total_throughput(self) -> Optional[float]:
        """
        전체 처리량 계산 (inferences per second)
        
        Returns:
            처리량 (inf/s)
        """
        if not self.completed_contexts:
            return None
        
        total_time_ns = sum(ctx.get_latency() for ctx in self.completed_contexts)
        if total_time_ns == 0:
            return None
        
        total_time_s = total_time_ns / 1e9
        return len(self.completed_contexts) / total_time_s
    
    def get_average_latency(self) -> Optional[float]:
        """평균 레이턴시 (ns)"""
        if not self.completed_contexts:
            return None
        
        latencies = [ctx.get_latency() for ctx in self.completed_contexts]
        return sum(latencies) / len(latencies)
    
    def get_stats(self) -> Dict:
        """통계 정보"""
        return {
            'total_contexts': len(self.contexts),
            'completed_contexts': len(self.completed_contexts),
            'average_latency_ns': self.get_average_latency(),
            'throughput_per_sec': self.get_total_throughput(),
            'contexts': [ctx.get_stats() for ctx in self.completed_contexts]
        }
    
    def clear(self):
        """모든 컨텍스트 제거"""
        self.contexts.clear()
        self.completed_contexts.clear()
    
    def __repr__(self):
        return (f"PipelineManager(total={len(self.contexts)}, "
                f"completed={len(self.completed_contexts)})")
