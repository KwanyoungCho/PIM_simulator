"""
Graph Validation - Weight tiling 및 shape 검증
"""
from typing import Dict, List, Tuple
from .compute_node import ComputeGraph, ComputeNode
from .pim_simulator import PIMSimulator


class GraphValidator:
    """연산 그래프 검증"""
    
    def __init__(self, graph: ComputeGraph, pim: PIMSimulator):
        self.graph = graph
        self.pim = pim
        self.errors = []
        self.warnings = []
    
    def validate(self) -> bool:
        """
        전체 그래프 검증
        
        Returns:
            검증 성공 여부
        """
        self.errors = []
        self.warnings = []
        
        # 1. 노드별 검증
        for node_id, node in self.graph.nodes.items():
            self._validate_node(node)
        
        # 2. Weight tile 배치 검증
        self._validate_weight_placement()
        
        # 3. 결과 출력
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_node(self, node: ComputeNode):
        """개별 노드 검증"""
        
        # Conv 노드만 검증 (concat, add는 제외)
        if node.node_type not in ["conv"]:
            return
        
        # 1. Weight tile 존재 확인
        if not node.weight_tiles:
            self.errors.append(f"[{node.node_id}] No weight tiles specified")
            return
        
        # 2. Input shape와 weight reduction dim 검증
        self._validate_weight_shape(node)
    
    def _validate_weight_shape(self, node: ComputeNode):
        """
        Weight shape 검증 (im2col 고려)
        
        Conv: reduction_dim = kernel_h × kernel_w × input_channels
        """
        if 'kernel' not in node.metadata:
            self.warnings.append(f"[{node.node_id}] No kernel info in metadata")
            return
        
        # Kernel size 파싱
        kernel_str = node.metadata['kernel']
        if 'x' in kernel_str:
            k_h, k_w = map(int, kernel_str.split('x'))
        else:
            k_h = k_w = int(kernel_str[0])
        
        # Input channels (CHW format: Channels, Height, Width)
        if len(node.input_shape) == 3:
            input_channels = node.input_shape[0]  # CHW: Channels는 첫 번째
        else:
            self.errors.append(f"[{node.node_id}] Invalid input shape: {node.input_shape}")
            return
        
        # 예상 reduction dim
        expected_reduction_dim = k_h * k_w * input_channels
        
        # Output channels (CHW format: Channels, Height, Width)
        if len(node.output_shape) == 3:
            output_channels = node.output_shape[0]  # CHW: Channels는 첫 번째
        else:
            self.errors.append(f"[{node.node_id}] Invalid output shape: {node.output_shape}")
            return
        
        # Weight tile shape 검증
        for tile_id in node.weight_tiles:
            tile_shape = self._get_weight_tile_shape(tile_id, node.array_id)
            if tile_shape is None:
                self.errors.append(f"[{node.node_id}] Weight tile '{tile_id}' not found")
                continue
            
            tile_output_dim, tile_reduction_dim = tile_shape
            
            # Reduction dim 검증
            if tile_reduction_dim != expected_reduction_dim:
                self.errors.append(
                    f"[{node.node_id}] Weight tile '{tile_id}' reduction dim mismatch: "
                    f"expected {expected_reduction_dim}, got {tile_reduction_dim}"
                )
            
            # Output dim 검증 (tile들의 합이 output_channels와 일치해야 함)
            # 단일 tile이면 exact match, multi-tile이면 합산 확인 필요
            if len(node.weight_tiles) == 1:
                if tile_output_dim != output_channels:
                    self.errors.append(
                        f"[{node.node_id}] Weight tile '{tile_id}' output dim mismatch: "
                        f"expected {output_channels}, got {tile_output_dim}"
                    )
    
    def _get_weight_tile_shape(self, tile_id: str, array_id: int) -> Tuple[int, int]:
        """
        Weight tile의 shape 조회
        
        Returns:
            (output_dim, reduction_dim) 또는 None
        """
        array = self.pim.get_array(array_id)
        for area in array.areas:
            for tile in area.tiles:
                if tile.weight_id == tile_id:
                    return tile.get_shape()
        
        # 다른 Array에서도 찾아보기
        for arr in self.pim.eflash_arrays:
            for area in arr.areas:
                for tile in area.tiles:
                    if tile.weight_id == tile_id:
                        return tile.get_shape()
        
        return None
    
    def _validate_weight_placement(self):
        """모든 weight tile이 배치되었는지 확인"""
        for node_id, node in self.graph.nodes.items():
            if node.node_type not in ["conv"]:
                continue
            
            for tile_id in node.weight_tiles:
                if not self._is_weight_placed(tile_id):
                    self.errors.append(
                        f"[{node_id}] Weight tile '{tile_id}' is not placed on any array"
                    )
    
    def _is_weight_placed(self, tile_id: str) -> bool:
        """Weight tile이 배치되었는지 확인"""
        for array in self.pim.eflash_arrays:
            for area in array.areas:
                for tile in area.tiles:
                    if tile.weight_id == tile_id:
                        return True
        return False
    
    def _print_results(self):
        """검증 결과 출력"""
        print("\n" + "=" * 80)
        print("GRAPH VALIDATION")
        print("=" * 80)
        
        if self.errors:
            print(f"\n❌ {len(self.errors)} Error(s) found:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} Warning(s):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All validations passed!")
        
        print("=" * 80)
