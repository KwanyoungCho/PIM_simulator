from typing import List, Dict, Optional, Tuple
from .area import Area
from .sram import SRAMBuffer


class eFlashArray:
    """8개의 Area를 가진 eFlash Array + 내부 SRAM"""
    
    NUM_AREAS = 8
    
    def __init__(self, 
                 array_id: int = 0, 
                 area_execution_time_ns: float = 100.0,
                 sram_size_bytes: int = 1024 * 1024):  # 기본 1MB
        """
        Args:
            array_id: eFlash Array 식별자
            area_execution_time_ns: Area 하나를 실행하는데 걸리는 시간 (nanoseconds)
            sram_size_bytes: 내부 SRAM 크기 (bytes)
        """
        self.array_id = array_id
        self.areas: List[Area] = [Area(area_id=i) for i in range(self.NUM_AREAS)]
        self.sram = SRAMBuffer(size_bytes=sram_size_bytes, name=f"eFlash_{array_id}_SRAM")
        
        self.current_active_area: Optional[int] = None
        self.area_execution_time_ns = area_execution_time_ns  # Area 1회 실행 시간
        self.total_execution_time_ns = 0.0  # 총 실행 시간
        self.execution_count = 0  # 실행 횟수
    
    def place_weight(self, 
                     area_id: int,
                     weight_id: str,
                     shape: Tuple[int, int],  # (output_dim, reduction_dim)
                     metadata: Optional[Dict] = None) -> bool:
        """
        특정 Area에 weight 배치
        
        Args:
            area_id: 배치할 Area ID (0-7)
            weight_id: Weight 식별자
            shape: (output_dim, reduction_dim)
            metadata: 추가 메타데이터
            
        Returns:
            배치 성공 여부
        """
        if area_id < 0 or area_id >= self.NUM_AREAS:
            raise ValueError(f"Invalid area_id: {area_id}. Must be 0-{self.NUM_AREAS-1}")
        
        return self.areas[area_id].place_weight(weight_id, shape, metadata)
    
    def auto_place_weight(self,
                          weight_id: str,
                          shape: Tuple[int, int],
                          metadata: Optional[Dict] = None) -> Tuple[bool, Optional[int]]:
        """
        자동으로 적절한 Area에 weight 배치 (first-fit)
        
        Args:
            weight_id: Weight 식별자
            shape: (output_dim, reduction_dim)
            metadata: 추가 메타데이터
            
        Returns:
            (배치 성공 여부, 배치된 Area ID)
        """
        output_dim, reduction_dim = shape
        
        for area in self.areas:
            if area.can_place_weight(output_dim, reduction_dim):
                success = area.place_weight(weight_id, shape, metadata)
                if success:
                    return True, area.area_id
        
        return False, None
    
    def execute_area(self, area_id: int) -> Dict:
        """
        특정 Area 실행 (area_execution_time_ns만큼 소요)
        해당 Area의 모든 tile에 대해 reduction 완료
        
        Args:
            area_id: 실행할 Area ID
            
        Returns:
            실행 결과 (tile별 output shape 등)
        """
        if area_id < 0 or area_id >= self.NUM_AREAS:
            raise ValueError(f"Invalid area_id: {area_id}")
        
        self.current_active_area = area_id
        self.total_execution_time_ns += self.area_execution_time_ns
        self.execution_count += 1
        
        area = self.areas[area_id]
        results = {
            'area_id': area_id,
            'execution_time_ns': self.area_execution_time_ns,
            'total_time_ns': self.total_execution_time_ns,
            'tiles_executed': []
        }
        
        for tile in area.tiles:
            # Reduction이 완료되면 output_dim 크기의 결과가 나옴
            tile_result = {
                'weight_id': tile.weight_id,
                'output_shape': (tile.output_dim,),  # Reduction 후 1D 출력
                'metadata': tile.metadata
            }
            results['tiles_executed'].append(tile_result)
        
        return results
    
    def get_area(self, area_id: int) -> Area:
        """Area 객체 반환"""
        if area_id < 0 or area_id >= self.NUM_AREAS:
            raise ValueError(f"Invalid area_id: {area_id}")
        return self.areas[area_id]
    
    def get_sram(self) -> SRAMBuffer:
        """SRAM 객체 반환"""
        return self.sram
    
    def get_total_utilization(self) -> Dict:
        """전체 Array의 사용률 정보 반환"""
        total_tiles = sum(len(area.tiles) for area in self.areas)
        utilization_per_area = [area.get_utilization_ratio() for area in self.areas]
        avg_utilization = sum(utilization_per_area) / len(utilization_per_area) if utilization_per_area else 0.0
        
        return {
            'array_id': self.array_id,
            'total_tiles': total_tiles,
            'utilization_per_area': utilization_per_area,
            'average_utilization': avg_utilization,
            'total_execution_time_ns': self.total_execution_time_ns,
            'execution_count': self.execution_count,
            'area_execution_time_ns': self.area_execution_time_ns,
            'sram_stats': self.sram.get_stats()
        }
    
    def clear_all(self):
        """모든 Area 및 SRAM 초기화"""
        for area in self.areas:
            area.clear()
        self.sram.clear()
        self.current_active_area = None
        self.total_execution_time_ns = 0.0
        self.execution_count = 0
    
    def __repr__(self):
        return (f"eFlashArray(id={self.array_id}, "
                f"areas={self.NUM_AREAS}, "
                f"exec_time={self.total_execution_time_ns}ns, "
                f"sram={self.sram})")
