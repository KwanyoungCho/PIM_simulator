from typing import List, Dict, Optional, Tuple
from .eflash_array import eFlashArray
from .sram import SRAMBuffer


class PIMSimulator:
    """PIM (Processing-In-Memory) 시뮬레이터 - 여러 eFlash Array + 공유 SRAM"""
    
    def __init__(self,
                 num_arrays: int = 4,
                 area_execution_time_ns: float = 100.0,
                 array_sram_size_bytes: int = 1024 * 1024,  # 각 Array의 내부 SRAM: 1MB
                 shared_sram_size_bytes: int = 10 * 1024 * 1024):  # 공유 SRAM: 10MB
        """
        Args:
            num_arrays: eFlash Array 개수
            area_execution_time_ns: 각 Area 실행 시간 (nanoseconds)
            array_sram_size_bytes: 각 eFlash Array 내부 SRAM 크기 (bytes)
            shared_sram_size_bytes: 공유 SRAM 크기 (bytes)
        """
        self.num_arrays = num_arrays
        self.eflash_arrays: List[eFlashArray] = [
            eFlashArray(
                array_id=i,
                area_execution_time_ns=area_execution_time_ns,
                sram_size_bytes=array_sram_size_bytes
            ) for i in range(num_arrays)
        ]
        self.shared_sram = SRAMBuffer(
            size_bytes=shared_sram_size_bytes,
            name="Shared_SRAM"
        )
        
        self.total_execution_time_ns = 0.0
        
    def get_array(self, array_id: int) -> eFlashArray:
        """특정 eFlash Array 반환"""
        if array_id < 0 or array_id >= self.num_arrays:
            raise ValueError(f"Invalid array_id: {array_id}. Must be 0-{self.num_arrays-1}")
        return self.eflash_arrays[array_id]
    
    def get_shared_sram(self) -> SRAMBuffer:
        """공유 SRAM 반환"""
        return self.shared_sram
    
    def place_weight_on_array(self,
                              array_id: int,
                              area_id: int,
                              weight_id: str,
                              shape: Tuple[int, int],
                              metadata: Optional[Dict] = None) -> bool:
        """
        특정 Array의 특정 Area에 weight 배치
        
        Args:
            array_id: eFlash Array ID
            area_id: Area ID
            weight_id: Weight 식별자
            shape: (output_dim, reduction_dim)
            metadata: 메타데이터
            
        Returns:
            배치 성공 여부
        """
        array = self.get_array(array_id)
        return array.place_weight(area_id, weight_id, shape, metadata)
    
    def auto_place_weight(self,
                          weight_id: str,
                          shape: Tuple[int, int],
                          metadata: Optional[Dict] = None) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        자동으로 적절한 Array와 Area에 weight 배치
        
        Args:
            weight_id: Weight 식별자
            shape: (output_dim, reduction_dim)
            metadata: 메타데이터
            
        Returns:
            (배치 성공 여부, Array ID, Area ID)
        """
        for array in self.eflash_arrays:
            success, area_id = array.auto_place_weight(weight_id, shape, metadata)
            if success:
                return True, array.array_id, area_id
        
        return False, None, None
    
    def execute_area(self, array_id: int, area_id: int) -> Dict:
        """
        특정 Array의 특정 Area 실행
        
        Args:
            array_id: eFlash Array ID
            area_id: Area ID
            
        Returns:
            실행 결과
        """
        array = self.get_array(array_id)
        result = array.execute_area(area_id)
        self.total_execution_time_ns = max(
            self.total_execution_time_ns,
            array.total_execution_time_ns
        )
        return result
    
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
