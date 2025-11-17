from typing import List, Dict, Optional
from .weight_tile import WeightTile


class Area:
    """하나의 Area (최대 128×1280)"""
    
    MAX_OUTPUT_DIM = 128
    MAX_REDUCTION_DIM = 1280
    
    def __init__(self, area_id: int):
        """
        Args:
            area_id: Area 식별자 (0-7)
        """
        self.area_id = area_id
        self.tiles: List[WeightTile] = []
        self.utilization = 0  # 현재 사용된 행 수
        
    def can_place_weight(self, output_dim: int, reduction_dim: int) -> bool:
        """
        새로운 weight를 배치할 수 있는지 확인
        
        Args:
            output_dim: 배치할 weight의 output dimension
            reduction_dim: 배치할 weight의 reduction dimension
            
        Returns:
            배치 가능 여부
        """
        if reduction_dim > self.MAX_REDUCTION_DIM:
            return False
        if self.utilization + output_dim > self.MAX_OUTPUT_DIM:
            return False
        return True
    
    def place_weight(self, 
                     weight_id: str,
                     shape: tuple,  # (output_dim, reduction_dim)
                     metadata: Optional[Dict] = None) -> bool:
        """
        Area에 weight를 배치
        
        Args:
            weight_id: Weight 식별자
            shape: (output_dim, reduction_dim)
            metadata: 추가 메타데이터
            
        Returns:
            배치 성공 여부
        """
        output_dim, reduction_dim = shape
        
        if not self.can_place_weight(output_dim, reduction_dim):
            return False
        
        # 배치 위치 결정
        start_row = self.utilization
        end_row = start_row + output_dim
        
        # WeightTile 생성 및 추가
        tile = WeightTile(
            weight_id=weight_id,
            shape=(output_dim, reduction_dim),
            position=(start_row, end_row),
            metadata=metadata
        )
        
        self.tiles.append(tile)
        self.utilization = end_row
        
        return True
    
    def clear(self):
        """Area 초기화"""
        self.tiles.clear()
        self.utilization = 0
    
    def __repr__(self):
        return (f"Area(id={self.area_id}, "
                f"utilization={self.utilization}/{self.MAX_OUTPUT_DIM}, "
                f"tiles={len(self.tiles)})")
