from typing import Dict, Optional, Tuple


class WeightTile:
    """Area 내 배치된 하나의 weight tile (실제 데이터 없이 메타정보만)"""
    
    def __init__(self, 
                 weight_id: str,
                 shape: Tuple[int, int],  # (output_dim, reduction_dim)
                 position: Tuple[int, int],  # (start_row, end_row)
                 metadata: Optional[Dict] = None):
        """
        Args:
            weight_id: Weight 식별자
            shape: (output_dim, reduction_dim)
            position: Area 내 위치 (start_row, end_row)
            metadata: layer name, type 등 추가 정보
        """
        self.weight_id = weight_id
        self.output_dim, self.reduction_dim = shape
        self.start_row, self.end_row = position
        self.metadata = metadata or {}
        
        # Validation
        assert self.end_row - self.start_row == self.output_dim, \
            f"Position range ({self.start_row}-{self.end_row}) doesn't match output_dim ({self.output_dim})"
    
    def get_shape(self) -> Tuple[int, int]:
        """Weight shape 반환"""
        return (self.output_dim, self.reduction_dim)
    
    # def get_position(self) -> Tuple[int, int]:
    #     """Area 내 위치 반환"""
    #     return (self.start_row, self.end_row)
    
    def __repr__(self):
        return (f"WeightTile(id={self.weight_id}, "
                f"shape={self.output_dim}x{self.reduction_dim}, "
                f"rows={self.start_row}-{self.end_row})")
