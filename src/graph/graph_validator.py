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
    
    # ==================== Helper Functions ====================
    
    @staticmethod
    def _is_continuous_coverage(ranges: List[Tuple[int, int]], expected_start: int, expected_end: int) -> Tuple[bool, str]:
        """
        범위들이 [expected_start, expected_end)를 연속적으로 완전히 커버하는지 확인
        
        Returns:
            (is_valid, error_message)
        """
        if not ranges:
            return False, "No ranges provided"
        
        # 범위들을 시작 위치로 정렬
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        
        # 첫 범위가 expected_start에서 시작하는지 확인
        if sorted_ranges[0][0] != expected_start:
            return False, f"First range starts at {sorted_ranges[0][0]}, expected {expected_start}"
        
        # 연속성 및 완전성 확인
        current_end = sorted_ranges[0][1]
        
        for i in range(1, len(sorted_ranges)):
            start, end = sorted_ranges[i]
            
            # Gap 확인
            if start > current_end:
                return False, f"Gap found: [{current_end}, {start})"
            
            # Overlap 확인
            if start < current_end:
                return False, f"Overlap found: previous ends at {current_end}, next starts at {start}"
            
            current_end = end
        
        # 마지막 범위가 expected_end까지 도달하는지 확인
        if current_end != expected_end:
            return False, f"Last range ends at {current_end}, expected {expected_end}"
        
        return True, ""
    
    @staticmethod
    def _has_overlap(ranges: List[Tuple[int, int]]) -> bool:
        """범위들이 중복되는지 확인"""
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        
        for i in range(len(sorted_ranges) - 1):
            if sorted_ranges[i][1] > sorted_ranges[i + 1][0]:
                return True
        
        return False
    
    @staticmethod
    def _calculate_conv_output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
        """Conv 연산의 출력 spatial size 계산"""
        return (input_size - kernel_size + 2 * padding) // stride + 1
    
    def validate(self) -> bool:
        """
        전체 그래프 검증
        
        Returns:
            검증 성공 여부
        """
        self.errors = []
        self.warnings = []
        
        # 1. Graph topology 검증 (cycle, 존재하지 않는 노드 참조)
        self._validate_graph_topology()
        
        # 2. 노드별 기본 검증
        for node_id, node in self.graph.nodes.items():
            self._validate_node(node)
        
        # 3. Weight tile 배치 검증
        self._validate_weight_placement()
        
        # 4. Tiling completeness 검증 (tile들이 완전히 커버하는지)
        self._validate_tiling_completeness()
        
        # 5. Shape propagation 검증 (input + weight → output)
        self._validate_shape_propagation()
        
        # 6. Area별 reduction dimension 패킹 검증
        self._validate_area_packing()
        
        # 7. Graph preprocessing 결과 검증 (확장된 그래프)
        self._validate_graph_preprocessing()
        
        # 8. 결과 출력
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
    
    def _validate_graph_topology(self):
        """Graph topology 검증: cycle, 존재하지 않는 노드 참조 등"""
        
        # 1. 모든 input_nodes가 존재하는지 확인
        for node_id, node in self.graph.nodes.items():
            for inp_id in node.input_nodes:
                if inp_id not in self.graph.nodes:
                    self.errors.append(
                        f"[{node_id}] References non-existent input node '{inp_id}'"
                    )
        
        # 2. Cycle 검출 (DFS)
        visited = set()
        rec_stack = set()
        cycle_path = []
        
        def has_cycle_dfs(node_id, path):
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            if node_id in self.graph.nodes:
                for inp_id in self.graph.nodes[node_id].input_nodes:
                    if inp_id not in visited:
                        if has_cycle_dfs(inp_id, path):
                            return True
                    elif inp_id in rec_stack:
                        # Cycle 발견
                        cycle_start_idx = path.index(inp_id)
                        cycle_nodes = path[cycle_start_idx:] + [inp_id]
                        self.errors.append(
                            f"Cycle detected in graph: {' → '.join(cycle_nodes)}"
                        )
                        return True
            
            path.pop()
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.graph.nodes:
            if node_id not in visited:
                has_cycle_dfs(node_id, [])
    
    def _validate_tiling_completeness(self):
        """Weight tiling이 완전한지 검증: tile들이 원본 weight를 완전히 커버하는지"""
        
        for node_id, node in self.graph.nodes.items():
            # Conv 노드이고 eFlash인 경우만 검증
            if node.node_type != "conv" or node.device_type != "eflash":
                continue
            
            if not node.weight_tiles:
                continue
            
            # Metadata에서 tiling 정보 추출
            output_ranges = []
            reduction_ranges = []
            
            for tile in node.weight_tiles:
                if isinstance(tile, str):
                    # String인 경우 실제 tile 객체 찾기
                    tile_obj = self._find_weight_tile(tile, node.array_id)
                    if tile_obj is None:
                        continue
                    metadata = tile_obj.metadata
                else:
                    metadata = tile.metadata
                
                if metadata is None:
                    continue
                
                # Output range
                if 'output_start' in metadata and 'output_end' in metadata:
                    output_ranges.append((metadata['output_start'], metadata['output_end']))
                
                # Reduction range
                if 'reduction_start' in metadata and 'reduction_end' in metadata:
                    reduction_ranges.append((metadata['reduction_start'], metadata['reduction_end']))
            
            # Output shape에서 expected dimensions 계산
            if len(node.output_shape) >= 1:
                expected_output_channels = node.output_shape[0]
            else:
                continue
            
            # Expected reduction dimension 계산
            kernel_size = node.metadata.get('kernel') or node.metadata.get('kernel_size')
            if kernel_size is None:
                continue
            
            if isinstance(kernel_size, str):
                if 'x' in kernel_size:
                    k_h, k_w = map(int, kernel_size.split('x'))
                else:
                    k_h = k_w = int(kernel_size[0])
            elif isinstance(kernel_size, (list, tuple)):
                k_h, k_w = kernel_size
            else:
                k_h = k_w = kernel_size
            
            if len(node.input_shape) >= 1:
                input_channels = node.input_shape[0]
                expected_reduction_dim = k_h * k_w * input_channels
            else:
                continue
            
            # Output dimension completeness 검증
            if output_ranges:
                is_valid, error_msg = self._is_continuous_coverage(
                    output_ranges, 0, expected_output_channels
                )
                if not is_valid:
                    self.errors.append(
                        f"[{node_id}] Output dimension tiling incomplete: {error_msg}"
                    )
            
            # Reduction dimension completeness 검증
            if reduction_ranges:
                is_valid, error_msg = self._is_continuous_coverage(
                    reduction_ranges, 0, expected_reduction_dim
                )
                if not is_valid:
                    self.errors.append(
                        f"[{node_id}] Reduction dimension tiling incomplete: {error_msg}"
                    )
    
    def _find_weight_tile(self, tile_id: str, array_id: int):
        """Weight tile 객체 찾기"""
        # 지정된 array에서 먼저 찾기
        if array_id is not None and array_id < len(self.pim.eflash_arrays):
            array = self.pim.eflash_arrays[array_id]
            for area in array.areas:
                for tile in area.tiles:
                    if tile.weight_id == tile_id:
                        return tile
        
        # 모든 array에서 찾기
        for array in self.pim.eflash_arrays:
            for area in array.areas:
                for tile in area.tiles:
                    if tile.weight_id == tile_id:
                        return tile
        
        return None
    
    def _validate_shape_propagation(self):
        """Shape propagation 검증: input shape + weight → output shape"""
        
        for node_id, node in self.graph.nodes.items():
            if node.node_type == "conv":
                self._validate_conv_shape(node)
            elif node.node_type == "concat":
                self._validate_concat_shape(node)
            elif node.node_type == "add":
                self._validate_add_shape(node)
    
    def _validate_conv_shape(self, node: ComputeNode):
        """Conv 노드의 shape 검증"""
        
        if len(node.input_shape) != 3 or len(node.output_shape) != 3:
            return  # Shape 정보가 부족하면 스킵
        
        C_in, H_in, W_in = node.input_shape
        C_out, H_out, W_out = node.output_shape
        
        # Kernel size
        kernel_size = node.metadata.get('kernel') or node.metadata.get('kernel_size')
        if kernel_size is None:
            return
        
        if isinstance(kernel_size, str):
            if 'x' in kernel_size:
                k_h, k_w = map(int, kernel_size.split('x'))
            else:
                k_h = k_w = int(kernel_size[0])
        elif isinstance(kernel_size, (list, tuple)):
            k_h, k_w = kernel_size
        else:
            k_h = k_w = kernel_size
        
        # Stride, padding
        stride = node.metadata.get('stride', 1)
        padding = node.metadata.get('padding', 0)
        
        # Output spatial size 계산
        H_out_calc = self._calculate_conv_output_size(H_in, k_h, stride, padding)
        W_out_calc = self._calculate_conv_output_size(W_in, k_w, stride, padding)
        
        # 검증
        if H_out_calc != H_out:
            self.errors.append(
                f"[{node.node_id}] Output height mismatch: "
                f"calculated {H_out_calc}, specified {H_out}"
            )
        
        if W_out_calc != W_out:
            self.errors.append(
                f"[{node.node_id}] Output width mismatch: "
                f"calculated {W_out_calc}, specified {W_out}"
            )
        
        # Weight tile들의 총 output channels 확인 (eFlash인 경우)
        if node.device_type == "eflash" and node.weight_tiles:
            total_output_from_tiles = 0
            for tile in node.weight_tiles:
                if isinstance(tile, str):
                    tile_shape = self._get_weight_tile_shape(tile, node.array_id)
                else:
                    tile_shape = tile.get_shape()
                
                if tile_shape:
                    # Output dimension별로 고유한 tile만 카운트
                    tile_metadata = tile.metadata if not isinstance(tile, str) else self._find_weight_tile(tile, node.array_id).metadata
                    if tile_metadata and 'reduction_tile_idx' in tile_metadata and tile_metadata['reduction_tile_idx'] == 0:
                        total_output_from_tiles += tile_shape[0]
            
            if total_output_from_tiles > 0 and total_output_from_tiles != C_out:
                self.warnings.append(
                    f"[{node.node_id}] Total output channels from tiles ({total_output_from_tiles}) "
                    f"!= expected ({C_out})"
                )
    
    def _validate_concat_shape(self, node: ComputeNode):
        """Concat 노드의 shape 검증"""
        
        if not node.input_nodes or len(node.output_shape) < 1:
            return
        
        # 입력들의 채널 수 합계 계산
        expected_output_channels = 0
        for inp_id in node.input_nodes:
            if inp_id in self.graph.nodes:
                inp_node = self.graph.nodes[inp_id]
                if len(inp_node.output_shape) >= 1:
                    expected_output_channels += inp_node.output_shape[0]
        
        actual_output_channels = node.output_shape[0]
        
        if expected_output_channels > 0 and actual_output_channels != expected_output_channels:
            self.errors.append(
                f"[{node.node_id}] Concat output channels mismatch: "
                f"sum of inputs = {expected_output_channels}, specified = {actual_output_channels}"
            )
    
    def _validate_add_shape(self, node: ComputeNode):
        """Add 노드의 shape 검증"""
        
        if not node.input_nodes:
            return
        
        # 모든 입력이 같은 shape을 가져야 함
        shapes = []
        for inp_id in node.input_nodes:
            if inp_id in self.graph.nodes:
                shapes.append(tuple(self.graph.nodes[inp_id].output_shape))
        
        if len(shapes) > 1 and len(set(shapes)) > 1:
            self.errors.append(
                f"[{node.node_id}] Add operation requires all inputs to have same shape, "
                f"but got: {shapes}"
            )
        
        # Output shape도 입력과 동일해야 함
        if shapes and tuple(node.output_shape) != shapes[0]:
            self.errors.append(
                f"[{node.node_id}] Add output shape {tuple(node.output_shape)} "
                f"doesn't match input shape {shapes[0]}"
            )
    
    def _validate_graph_preprocessing(self):
        """Graph preprocessing 결과 검증: reduce/concat 노드 구조"""
        
        for node_id, node in self.graph.nodes.items():
            # Reduce 노드 검증
            if "reduce" in node_id.lower() and "tile" in node_id.lower():
                self._validate_reduce_node(node)
            
            # Tiled concat 노드 검증
            if "concat" in node_id.lower() and "tile" in node_id.lower():
                self._validate_tiled_concat_node(node)
    
    def _validate_reduce_node(self, node: ComputeNode):
        """Reduce 노드 검증: 모든 input이 같은 output tile index를 가져야 함"""
        
        if not node.input_nodes:
            return
        
        output_tile_indices = set()
        for inp_id in node.input_nodes:
            if inp_id in self.graph.nodes:
                inp_node = self.graph.nodes[inp_id]
                if inp_node.metadata and 'output_tile_idx' in inp_node.metadata:
                    output_tile_indices.add(inp_node.metadata['output_tile_idx'])
        
        if len(output_tile_indices) > 1:
            self.errors.append(
                f"[{node.node_id}] Reduce node has inputs with different output tile indices: {output_tile_indices}"
            )
    
    def _validate_tiled_concat_node(self, node: ComputeNode):
        """Tiled concat 노드 검증: 입력들이 서로 다른 output tile index를 가져야 함"""
        
        if not node.input_nodes:
            return
        
        output_tile_indices = []
        for inp_id in node.input_nodes:
            if inp_id in self.graph.nodes:
                inp_node = self.graph.nodes[inp_id]
                if inp_node.metadata and 'output_tile_idx' in inp_node.metadata:
                    idx = inp_node.metadata['output_tile_idx']
                    if idx in output_tile_indices:
                        self.errors.append(
                            f"[{node.node_id}] Concat node has duplicate output tile index {idx}"
                        )
                    output_tile_indices.append(idx)
    
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
