"""
Graph Validation - Weight tiling 및 shape 검증
"""
from typing import Dict, List, Tuple
from .compute_node import ComputeGraph, ComputeNode
from ..hardware import PIMSimulator


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
        
        # 3. Area별 reduction dimension 패킹 검증
        self._validate_area_packing()
        
        # 4. 결과 출력
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_node(self, node: ComputeNode):
        """개별 노드 검증"""
        
        # Conv 노드 중 eFlash만 검증 (NPU 노드는 제외)
        if node.node_type not in ["conv"]:
            return
        if node.device_type != "eflash":
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
        # Kernel size 가져오기 ('kernel' 또는 'kernel_size')
        kernel_size = node.metadata.get('kernel') or node.metadata.get('kernel_size')
        if kernel_size is None:
            self.warnings.append(f"[{node.node_id}] No kernel info in metadata")
            return
        
        # Kernel size 파싱
        if isinstance(kernel_size, str):
            kernel_str = kernel_size
            if 'x' in kernel_str:
                k_h, k_w = map(int, kernel_str.split('x'))
            else:
                k_h = k_w = int(kernel_str[0])
        elif isinstance(kernel_size, (list, tuple)):
            k_h, k_w = kernel_size
        else:
            # int일 경우
            k_h = k_w = kernel_size
        
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
        for tile in node.weight_tiles:
            # tile은 WeightTile 객체 또는 string일 수 있음
            if isinstance(tile, str):
                tile_id = tile
                tile_shape = self._get_weight_tile_shape(tile_id, node.array_id)
            else:
                # WeightTile 객체
                tile_id = tile.weight_id
                tile_shape = tile.get_shape()
            
            if tile_shape is None:
                self.errors.append(f"[{node.node_id}] Weight tile '{tile_id}' not found")
                continue
            
            tile_output_dim, tile_reduction_dim = tile_shape
            
            # Reduction dim 검증 (tiling된 경우 부분적일 수 있음)
            # tile_reduction_dim은 expected_reduction_dim 이하여야 함
            if tile_reduction_dim > expected_reduction_dim:
                self.errors.append(
                    f"[{node.node_id}] Weight tile '{tile_id}' reduction dim too large: "
                    f"expected <= {expected_reduction_dim}, got {tile_reduction_dim}"
                )
            
            # Output dim 검증 (tiling된 경우 부분적일 수 있음)
            # tile_output_dim은 output_channels 이하여야 함
            if tile_output_dim > output_channels:
                self.errors.append(
                    f"[{node.node_id}] Weight tile '{tile_id}' output dim too large: "
                    f"expected <= {output_channels}, got {tile_output_dim}"
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
            if node.device_type != "eflash":
                continue
            
            for tile in node.weight_tiles:
                # tile은 WeightTile 객체 또는 string일 수 있음
                if isinstance(tile, str):
                    tile_id = tile
                else:
                    tile_id = tile.weight_id
                
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
    
    def _validate_area_packing(self):
        """Area별 reduction dimension 패킹 검증"""
        MAX_OUTPUT_DIM = 128
        MAX_REDUCTION_DIM = 1280
        
        total_wasted_reduction = 0
        total_wasted_output = 0
        high_utilization_count = 0
        
        for array_idx, array in enumerate(self.pim.eflash_arrays):
            for area in array.areas:
                if not hasattr(area, 'row_usage') or not area.row_usage:
                    continue  # 빈 area 건너뛰기
                
                # 각 row 범위 검증
                for (start_row, end_row), used_reduction in area.row_usage.items():
                    row_size = end_row - start_row
                    
                    # Output dimension 초과 검증
                    if end_row > MAX_OUTPUT_DIM:
                        self.errors.append(
                            f"[Array{array_idx}.Area{area.area_id}] Row range ({start_row}-{end_row}) "
                            f"exceeds MAX_OUTPUT_DIM ({MAX_OUTPUT_DIM})"
                        )
                    
                    # Reduction dimension 초과 검증
                    if used_reduction > MAX_REDUCTION_DIM:
                        self.errors.append(
                            f"[Array{array_idx}.Area{area.area_id}] Rows {start_row}-{end_row}: "
                            f"Reduction dimension ({used_reduction}) exceeds MAX ({MAX_REDUCTION_DIM})"
                        )
                    
                    # Reduction dimension 활용도 계산
                    reduction_util = (used_reduction / MAX_REDUCTION_DIM) * 100
                    wasted_reduction = MAX_REDUCTION_DIM - used_reduction
                    total_wasted_reduction += wasted_reduction
                    
                    # 높은 활용도 카운트 (>80%)
                    if reduction_util > 80:
                        high_utilization_count += 1
                    
                    # 낮은 활용도 경고 (<30%)
                    if reduction_util < 30:
                        self.warnings.append(
                            f"[Array{array_idx}.Area{area.area_id}] Rows {start_row}-{end_row}: "
                            f"Low reduction utilization ({reduction_util:.1f}%)"
                        )
                
                # Output dimension 활용도 계산
                max_row_used = max((end for _, end in area.row_usage.keys()), default=0)
                output_util = (max_row_used / MAX_OUTPUT_DIM) * 100
                wasted_output = MAX_OUTPUT_DIM - max_row_used
                total_wasted_output += wasted_output
                
                # 낮은 output 활용도 경고 (<50%)
                if max_row_used > 0 and output_util < 50:
                    self.warnings.append(
                        f"[Array{array_idx}.Area{area.area_id}] Low output utilization "
                        f"({output_util:.1f}%, only {max_row_used}/{MAX_OUTPUT_DIM} rows used)"
                    )
        
        # 통계 저장 (출력에서 사용)
        self.packing_stats = {
            'total_wasted_reduction': total_wasted_reduction,
            'total_wasted_output': total_wasted_output,
            'high_utilization_count': high_utilization_count
        }
    
    def _print_results(self):
        """검증 결과 출력"""
        print("\n" + "=" * 80)
        print("🔍 GRAPH VALIDATION RESULTS")
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
        
        # Weight Packing 통계
        if hasattr(self, 'packing_stats'):
            stats = self.packing_stats
            print(f"\n📊 Weight Packing Statistics:")
            print(f"  • High utilization row ranges (>80%): {stats['high_utilization_count']}")
            if stats['total_wasted_reduction'] > 0:
                print(f"  • Total wasted reduction dimension: {stats['total_wasted_reduction']:.0f} "
                      f"(~{stats['total_wasted_reduction']/1280:.1f} full columns)")
            if stats['total_wasted_output'] > 0:
                print(f"  • Total wasted output dimension: {stats['total_wasted_output']:.0f} "
                      f"(~{stats['total_wasted_output']/128:.1f} full areas)")
        
        print("=" * 80)
