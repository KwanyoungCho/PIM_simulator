from typing import Dict, Optional, Tuple


class SRAMBuffer:
    """SRAM 버퍼 - 데이터 이름과 크기만 관리"""
    
    def __init__(self, size_bytes: int, name: str = "SRAM"):
        """
        Args:
            size_bytes: SRAM 총 용량 (bytes)
            name: SRAM 식별자
        """
        self.name = name
        self.size_bytes = size_bytes
        self.used_bytes = 0
        self.data_entries: Dict[str, int] = {}  # {data_name: size_in_bytes}
        
    def can_allocate(self, size_bytes: int) -> bool:
        """
        주어진 크기를 할당할 수 있는지 확인
        
        Args:
            size_bytes: 할당할 크기 (bytes)
            
        Returns:
            할당 가능 여부
        """
        return (self.used_bytes + size_bytes) <= self.size_bytes
    
    def allocate(self, data_name: str, size_bytes: int) -> bool:
        """
        SRAM에 데이터 할당
        
        Args:
            data_name: 데이터 이름/식별자
            size_bytes: 데이터 크기 (bytes)
            
        Returns:
            할당 성공 여부
        """
        if data_name in self.data_entries:
            print(f"Warning: {data_name} already exists in SRAM. Overwriting.")
            self.deallocate(data_name)
            
        if not self.can_allocate(size_bytes):
            return False
        
        self.data_entries[data_name] = size_bytes
        self.used_bytes += size_bytes
        return True
    
    def deallocate(self, data_name: str) -> bool:
        """
        SRAM에서 데이터 해제
        
        Args:
            data_name: 데이터 이름/식별자
            
        Returns:
            해제 성공 여부
        """
        if data_name not in self.data_entries:
            return False
        
        size = self.data_entries[data_name]
        del self.data_entries[data_name]
        self.used_bytes -= size
        return True
    
    def get_data_size(self, data_name: str) -> Optional[int]:
        """데이터 크기 조회"""
        return self.data_entries.get(data_name)
    
    def clear(self):
        """SRAM 전체 초기화"""
        self.data_entries.clear()
        self.used_bytes = 0
    
    def get_utilization(self) -> float:
        """사용률 반환 (0.0 ~ 1.0)"""
        return self.used_bytes / self.size_bytes if self.size_bytes > 0 else 0.0
    
    def get_available_bytes(self) -> int:
        """사용 가능한 바이트 수 반환"""
        return self.size_bytes - self.used_bytes
    
    def get_stats(self) -> Dict:
        """SRAM 통계 정보 반환"""
        return {
            'name': self.name,
            'total_size_bytes': self.size_bytes,
            'used_bytes': self.used_bytes,
            'available_bytes': self.get_available_bytes(),
            'utilization': self.get_utilization(),
            'num_entries': len(self.data_entries),
            'entries': dict(self.data_entries)
        }
    
    def __repr__(self):
        return f"SRAMBuffer(name={self.name}, size={self.size_bytes}B, used={self.used_bytes}B, util={self.get_utilization():.2%})"
