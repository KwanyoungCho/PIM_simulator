from typing import Tuple, Optional


class ActivationBuffer:
    """Activation 데이터 버퍼 (실제 데이터 없이 메타정보만)"""
    
    def __init__(self,
                 buffer_id: str,
                 shape: Tuple[int, ...],
                 bytes_per_element: int = 1):
        """
        Args:
            buffer_id: 버퍼 식별자 (예: "conv1_output_batch0")
            shape: Activation shape (예: (batch, height, width, channels))
            bytes_per_element: 원소당 바이트 (8bit = 1byte)
        
        Note:
            위치 정보는 storage_locations에서 관리 (multi-location 지원)
        """
        self.buffer_id = buffer_id
        self.shape = shape
        self.bytes_per_element = bytes_per_element
        
        # 크기 계산
        self.num_elements = 1
        for dim in shape:
            self.num_elements *= dim
        self.size_bytes = self.num_elements * bytes_per_element
    
    def get_size_bytes(self) -> int:
        """버퍼 크기 (bytes) 반환"""
        return self.size_bytes
    
    def get_shape(self) -> Tuple[int, ...]:
        """Shape 반환"""
        return self.shape
    
    def __repr__(self):
        shape_str = "x".join(str(d) for d in self.shape)
        return (f"ActivationBuffer(id={self.buffer_id}, "
                f"shape={shape_str}, "
                f"size={self.size_bytes}B)")


class ActivationManager:
    """Activation 버퍼 관리자"""
    
    def __init__(self):
        """초기화"""
        self.buffers = {}  # {buffer_id: ActivationBuffer}
        self.node_outputs = {}  # {node_id: buffer_id} - 각 노드의 출력 버퍼
    
    def create_buffer(self,
                      buffer_id: str,
                      shape: Tuple[int, ...],
                      bytes_per_element: int = 1) -> ActivationBuffer:
        """
        새로운 activation 버퍼 생성 (메타데이터만)
        
        Args:
            buffer_id: 버퍼 ID
            shape: Activation shape
            bytes_per_element: 원소당 바이트
            
        Returns:
            생성된 ActivationBuffer
        
        Note:
            위치 정보는 storage_locations에서 관리 (multi-location 지원)
        """
        buffer = ActivationBuffer(buffer_id, shape, bytes_per_element)
        self.buffers[buffer_id] = buffer
        return buffer
    
    def register_node_output(self, node_id: str, buffer_id: str):
        """노드의 출력 버퍼 등록"""
        self.node_outputs[node_id] = buffer_id
    
    def get_buffer(self, buffer_id: str) -> Optional[ActivationBuffer]:
        """버퍼 조회"""
        return self.buffers.get(buffer_id)
    
    def get_node_output_buffer(self, node_id: str) -> Optional[ActivationBuffer]:
        """노드의 출력 버퍼 조회"""
        buffer_id = self.node_outputs.get(node_id)
        if buffer_id:
            return self.buffers.get(buffer_id)
        return None
    
    def allocate_to_sram(self, buffer_id: str, sram_buffer, location_name: str) -> bool:
        """
        SRAM에 버퍼 할당
        
        Args:
            buffer_id: 할당할 버퍼 ID
            sram_buffer: SRAMBuffer 객체
            location_name: 위치 이름
            
        Returns:
            할당 성공 여부
        """
        buffer = self.buffers.get(buffer_id)
        if not buffer:
            print(f"[Error] Buffer {buffer_id} not found")
            return False
        
        # 실제 SRAM에 할당 (위치 정보는 storage_locations에서 관리)
        success = sram_buffer.allocate(buffer_id, buffer.size_bytes)
        return success
    
    def deallocate_from_sram(self, buffer_id: str, sram_buffer) -> bool:
        """SRAM에서 버퍼 해제"""
        buffer = self.buffers.get(buffer_id)
        if not buffer:
            return False
        
        return sram_buffer.deallocate(buffer_id)
    
    def clear(self):
        """모든 버퍼 제거"""
        self.buffers.clear()
        self.node_outputs.clear()
    
    def get_stats(self):
        """통계 정보"""
        total_bytes = sum(buf.size_bytes for buf in self.buffers.values())
        return {
            'num_buffers': len(self.buffers),
            'total_bytes': total_bytes,
            'buffers': {buf_id: str(buf) for buf_id, buf in self.buffers.items()}
        }
    
    def __repr__(self):
        return f"ActivationManager(buffers={len(self.buffers)})"
