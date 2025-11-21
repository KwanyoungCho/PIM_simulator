from typing import Optional
from .sram import SRAMBuffer


class NPU:
    """
    Neural Processing Unit (NPU)
    TOPS 기반 연산 시간 계산
    FC (fully-connected) 연산 처리
    """
    
    def __init__(self,
                 npu_id: int,
                 tops: float = 10.0,
                 sram_size_bytes: int = 2 * 1024 * 1024):
        """
        Args:
            npu_id: NPU 식별자
            tops: 연산 성능 (Tera Operations Per Second)
            sram_size_bytes: NPU 전용 SRAM 크기 (bytes)
        """
        self.npu_id = npu_id
        self.tops = tops  # TOPS (INT8)
        self.ops_per_us = tops * 1e6  # operations per microsecond
        
        # NPU 전용 SRAM
        self.sram = SRAMBuffer(
            name=f"npu_{npu_id}_sram",
            size_bytes=sram_size_bytes
        )
        
        # 스케줄링용
        self.busy_until = 0.0  # us
    
    def calculate_execution_time(self, num_operations: int) -> float:
        """
        연산 시간 계산 (TOPS 기반)
        
        Args:
            num_operations: 총 연산 수 (INT8 operations)
            
        Returns:
            실행 시간 (us)
        """
        if num_operations <= 0:
            return 0.0
        
        return num_operations / self.ops_per_us
    
    def get_sram(self) -> SRAMBuffer:
        """NPU 전용 SRAM 반환"""
        return self.sram
    
    def set_busy_until(self, time_us: float):
        """NPU busy 시간 설정"""
        self.busy_until = max(self.busy_until, time_us)
    
    def get_busy_until(self) -> float:
        """NPU busy 시간 반환"""
        return self.busy_until
    
    def __repr__(self):
        return (f"NPU(id={self.npu_id}, "
                f"tops={self.tops}, "
                f"sram={self.sram.size_bytes/1024/1024:.1f}MB)")
