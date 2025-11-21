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
        # Row 범위별 사용된 reduction dimension: {(start_row, end_row): used_reduction_dim}
        self.row_usage: Dict[tuple, int] = {}
        
    def can_place_weight(self, output_dim: int, reduction_dim: int, 
                        target_row_range: Optional[tuple] = None) -> bool:
        """
        새로운 weight를 배치할 수 있는지 확인
        
        Args:
            output_dim: 배치할 weight의 output dimension
            reduction_dim: 배치할 weight의 reduction dimension
            target_row_range: 배치할 row 범위 (start, end). None이면 자동 할당
            
        Returns:
            배치 가능 여부
        """
        if reduction_dim > self.MAX_REDUCTION_DIM:
            return False
            
        if target_row_range:
            # 특정 row 범위에 배치 시도
            start_row, end_row = target_row_range
            if end_row - start_row != output_dim:
                return False
            if start_row < 0 or end_row > self.MAX_OUTPUT_DIM:
                return False
            
            # 해당 row 범위의 reduction dimension 확인
            if target_row_range in self.row_usage:
                used_reduction = self.row_usage[target_row_range]
                if used_reduction + reduction_dim > self.MAX_REDUCTION_DIM:
                    return False
            return True
        else:
            # 새로운 row 범위 할당 시도
            # 현재 사용된 row의 최대값 확인
            max_used_row = max((end for _, end in self.row_usage.keys()), default=0)
            if max_used_row + output_dim > self.MAX_OUTPUT_DIM:
                return False
            return True
    
    def place_weight(self, 
                     weight_id: str,
                     shape: tuple,  # (output_dim, reduction_dim)
                     metadata: Optional[Dict] = None,
                     target_row_range: Optional[tuple] = None) -> bool:
        """
        Area에 weight를 배치 (reduction dimension으로 패킹 지원)
        
        Args:
            weight_id: Weight 식별자
            shape: (output_dim, reduction_dim)
            metadata: 추가 메타데이터
            target_row_range: 배치할 row 범위 (start, end). None이면 자동 할당
            
        Returns:
            배치 성공 여부
        """
        output_dim, reduction_dim = shape
        
        if not self.can_place_weight(output_dim, reduction_dim, target_row_range):
            return False
        
        # Row 범위 결정
        if target_row_range:
            start_row, end_row = target_row_range
        else:
            # 새로운 row 범위 할당
            max_used_row = max((end for _, end in self.row_usage.keys()), default=0)
            start_row = max_used_row
            end_row = start_row + output_dim
        
        row_range = (start_row, end_row)
        
        # Reduction dimension 업데이트
        if row_range not in self.row_usage:
            self.row_usage[row_range] = 0
        self.row_usage[row_range] += reduction_dim
        
        # WeightTile 생성 및 추가
        tile = WeightTile(
            weight_id=weight_id,
            shape=(output_dim, reduction_dim),
            position=(start_row, end_row),
            metadata=metadata
        )
        
        self.tiles.append(tile)
        
        return True
    
    def clear(self):
        """Area 초기화"""
        self.tiles.clear()
        self.row_usage.clear()
    
    def __repr__(self):
        max_row = max((end for _, end in self.row_usage.keys()), default=0)
        return (f"Area(id={self.area_id}, "
                f"max_row_used={max_row}/{self.MAX_OUTPUT_DIM}, "
                f"tiles={len(self.tiles)})")
