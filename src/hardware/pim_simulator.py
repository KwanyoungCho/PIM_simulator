from typing import List, Dict, Optional, Tuple
from .eflash_array import eFlashArray
from .npu import NPU
from .sram import SRAMBuffer


class PIMSimulator:
    """PIM (Processing-In-Memory) 시뮬레이터 - eFlash Arrays + NPUs + 공유 SRAM"""
    
    def __init__(self,
                 num_arrays: int = 4,
                 num_npus: int = 0,
                 area_execution_time_us: float = 1.5,
                 npu_tops: float = 10.0,
                 array_sram_size_bytes: int = 1024 * 1024,  # 각 Array의 내부 SRAM: 1MB
                 npu_sram_size_bytes: int = 2 * 1024 * 1024,  # 각 NPU의 SRAM: 2MB
                 shared_sram_size_bytes: int = 10 * 1024 * 1024):  # 공유 SRAM: 10MB
        """
        Args:
            num_arrays: eFlash Array 개수
            num_npus: NPU 개수
            area_execution_time_us: 각 Area 실행 시간 (microseconds)
            npu_tops: NPU 성능 (TOPS)
            array_sram_size_bytes: 각 eFlash Array 내부 SRAM 크기 (bytes)
            npu_sram_size_bytes: 각 NPU SRAM 크기 (bytes)
            shared_sram_size_bytes: 공유 SRAM 크기 (bytes)
        """
        self.num_arrays = num_arrays
        self.num_npus = num_npus
        
        # eFlash Arrays
        self.eflash_arrays: List[eFlashArray] = [
            eFlashArray(
                array_id=i,
                area_execution_time_us=area_execution_time_us,
                sram_size_bytes=array_sram_size_bytes
            ) for i in range(num_arrays)
        ]
        
        # NPUs
        self.npus: List[NPU] = [
            NPU(
                npu_id=i,
                tops=npu_tops,
                sram_size_bytes=npu_sram_size_bytes
            ) for i in range(num_npus)
        ]
        
        # Shared SRAM
        self.shared_sram = SRAMBuffer(
            size_bytes=shared_sram_size_bytes,
            name="Shared_SRAM"
        )
        
        self.total_execution_time_us = 0.0
        
    def get_array(self, array_id: int) -> eFlashArray:
        """특정 eFlash Array 반환"""
        if array_id < 0 or array_id >= self.num_arrays:
            raise ValueError(f"Invalid array_id: {array_id}. Must be 0-{self.num_arrays-1}")
        return self.eflash_arrays[array_id]
    
    def get_npu(self, npu_id: int) -> NPU:
        """특정 NPU 반환"""
        if npu_id < 0 or npu_id >= self.num_npus:
            raise ValueError(f"Invalid npu_id: {npu_id}. Must be 0-{self.num_npus-1}")
        return self.npus[npu_id]
    
    def get_shared_sram(self) -> SRAMBuffer:
        """공유 SRAM 반환"""
        return self.shared_sram
    
    def place_weight_on_array(self,
                              array_id: int,
                              area_id: int,
                              weight_id: str,
                              shape: Tuple[int, int],
                              metadata: Optional[Dict] = None,
                              target_row_range: Optional[Tuple[int, int]] = None) -> bool:
        """
        특정 Array의 특정 Area에 weight 배치 (reduction dimension 패킹 지원)
        
        Args:
            array_id: eFlash Array ID
            area_id: Area ID
            weight_id: Weight 식별자
            shape: (output_dim, reduction_dim)
            metadata: 메타데이터
            target_row_range: 배치할 row 범위 (start, end). None이면 자동 할당
            
        Returns:
            배치 성공 여부
        """
        array = self.get_array(array_id)
        return array.place_weight(area_id, weight_id, shape, metadata, target_row_range)
    
    def get_total_stats(self) -> Dict:
        """전체 PIM 시스템 통계"""
        array_stats = []
        total_tiles = 0
        total_array_exec_time = 0.0
        
        for array in self.eflash_arrays:
            stats = array.get_total_utilization()
            array_stats.append(stats)
            total_tiles += stats['total_tiles']
            total_array_exec_time = max(total_array_exec_time, stats['total_execution_time_ns'])
        
        return {
            'num_arrays': self.num_arrays,
            'total_tiles': total_tiles,
            'total_execution_time_ns': total_array_exec_time,
            'array_stats': array_stats,
            'shared_sram_stats': self.shared_sram.get_stats()
        }
    
    def clear_all(self):
        """모든 Array 및 SRAM 초기화"""
        for array in self.eflash_arrays:
            array.clear_all()
        self.shared_sram.clear()
        self.total_execution_time_ns = 0.0
    
    def __repr__(self):
        return (f"PIMSimulator(arrays={self.num_arrays}, "
                f"exec_time={self.total_execution_time_ns}ns, "
                f"shared_sram={self.shared_sram})")
