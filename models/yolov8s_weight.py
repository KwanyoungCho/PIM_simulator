"""
YOLOv8s Weight Tile 배치
Conv layer의 weight를 im2col 형태로 변환 후 eFlash Array Area에 배치
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import WeightTile
from typing import List, Tuple, Dict
import math


def calculate_im2col_shape(conv_metadata: Dict, in_channels: int, out_channels: int) -> Tuple[int, int]:
    """
    Conv layer의 im2col 변환 후 weight shape 계산
    
    Returns:
        (output_dim, reduction_dim) tuple
    """
    kernel_size = conv_metadata.get('kernel_size', 3)
    if isinstance(kernel_size, tuple):
        k_h, k_w = kernel_size
    else:
        k_h = k_w = kernel_size
    
    output_dim = out_channels
    reduction_dim = in_channels * k_h * k_w
    
    return output_dim, reduction_dim


def create_weight_tiles_for_conv(node_id: str,
                                  in_channels: int,
                                  out_channels: int,
                                  conv_metadata: Dict,
                                  max_output_dim: int = 128,
                                  max_reduction_dim: int = 1280) -> List[Dict]:
    """
    Conv layer를 위한 weight tile 생성
    """
    output_dim, reduction_dim = calculate_im2col_shape(conv_metadata, in_channels, out_channels)
    
    tiles = []
    
    # Reduction dimension tiling
    if reduction_dim > max_reduction_dim:
        num_reduction_tiles = math.ceil(reduction_dim / max_reduction_dim)
    else:
        num_reduction_tiles = 1
    
    # Output dimension tiling
    if output_dim > max_output_dim:
        num_output_tiles = math.ceil(output_dim / max_output_dim)
    else:
        num_output_tiles = 1
    
    # Tile 생성
    for out_idx in range(num_output_tiles):
        out_start = out_idx * max_output_dim
        out_end = min(out_start + max_output_dim, output_dim)
        tile_output_dim = out_end - out_start
        
        for red_idx in range(num_reduction_tiles):
            red_start = red_idx * max_reduction_dim
            red_end = min(red_start + max_reduction_dim, reduction_dim)
            tile_reduction_dim = red_end - red_start
            
            # Tile ID 생성
            if num_output_tiles > 1 or num_reduction_tiles > 1:
                tile_id = f"{node_id}_tile_o{out_idx}_r{red_idx}"
            else:
                tile_id = f"{node_id}_weight"
            
            tile_info = {
                'weight_id': tile_id,
                'shape': (tile_output_dim, tile_reduction_dim),
                'metadata': {
                    'node_id': node_id,
                    'total_output_dim': output_dim,
                    'total_reduction_dim': reduction_dim,
                    'output_tile_idx': out_idx,
                    'reduction_tile_idx': red_idx,
                    'num_output_tiles': num_output_tiles,
                    'num_reduction_tiles': num_reduction_tiles
                }
            }
            tiles.append(tile_info)
    
    return tiles


def assign_tiles_to_areas(graph, num_arrays: int = 20) -> Dict:
    """
    YOLOv8s 그래프의 모든 Conv 노드에 대해 weight tile을 생성하고
    array0 area0부터 순차적으로 배치 (YOLOv5s와 동일한 방식)
    
    Args:
        graph: ComputeGraph
        num_arrays: 사용할 eFlash array 수
    
    Returns:
        배치 결과 딕셔너리
        {
            'tiles': [...],
            'placement': {node_id: [(array_id, area_id, tile_info), ...]},
            'array_utilization': {array_id: {area_id: utilization}}
        }
    """
    MAX_OUTPUT_DIM = 128
    MAX_REDUCTION_DIM = 1280
    NUM_AREAS_PER_ARRAY = 8
    
    # 결과 저장
    all_tiles = []
    placement = {}
    
    # Area별 row 사용 현황: {(array_id, area_id): {(start_row, end_row): used_reduction_dim}}
    area_row_usage = {}
    
    # 현재 배치 위치 (전역 순차 배치)
    current_array = 0
    current_area = 0
    current_row = 0  # 현재 area 내 row 위치
    
    print("="*70)
    print("YOLOv8s Weight Tile Assignment")
    print("="*70)
    print(f"Area constraints: MAX_OUTPUT_DIM={MAX_OUTPUT_DIM}, MAX_REDUCTION_DIM={MAX_REDUCTION_DIM}")
    print(f"Total arrays: {num_arrays}, Areas per array: {NUM_AREAS_PER_ARRAY}")
    print()
    
    # Conv 노드만 필터링 (순서대로)
    conv_nodes = [(node.node_id, node) for node in graph.get_all_nodes() 
                  if node.node_type == "conv"]
    
    print(f"Total Conv nodes: {len(conv_nodes)}")
    print()
    
    # 각 Conv 노드에 대해 weight tile 생성 및 배치
    for node_id, node in conv_nodes:
        in_channels = node.input_shape[0]
        out_channels = node.output_shape[0]
        
        # Weight tile 생성
        tiles = create_weight_tiles_for_conv(
            node_id, in_channels, out_channels, node.metadata,
            max_output_dim=MAX_OUTPUT_DIM,
            max_reduction_dim=MAX_REDUCTION_DIM
        )
        
        # Tiling 정보 출력
        if len(tiles) > 1:
            output_dim, reduction_dim = calculate_im2col_shape(node.metadata, in_channels, out_channels)
            num_output_tiles = (output_dim + MAX_OUTPUT_DIM - 1) // MAX_OUTPUT_DIM
            num_reduction_tiles = (reduction_dim + MAX_REDUCTION_DIM - 1) // MAX_REDUCTION_DIM
            
            reason = []
            if output_dim > MAX_OUTPUT_DIM:
                reason.append("out")
            if reduction_dim > MAX_REDUCTION_DIM:
                reason.append("red")
            reason_str = "+".join(reason)
            
            print(f"  🔀 {node_id}: {output_dim}×{reduction_dim} → {num_output_tiles}×{num_reduction_tiles} tiles ({reason_str} dim)")
        
        node_placements = []
        
        # 각 tile을 배치
        for tile_info in tiles:
            tile_shape = tile_info['shape']
            tile_output_dim = tile_shape[0]
            tile_reduction_dim = tile_shape[1]
            
            placed = False
            attempts = 0
            max_attempts = num_arrays * NUM_AREAS_PER_ARRAY * 2
            
            while not placed and attempts < max_attempts:
                key = (current_array, current_area)
                
                # 현재 area의 row usage 초기화
                if key not in area_row_usage:
                    area_row_usage[key] = {}
                
                # 이 tile을 배치할 수 있는 row 범위 찾기
                found_row = False
                
                # 기존에 같은 크기의 row 범위가 있는지 확인 (reduction dimension 패킹)
                for (start_row, end_row), used_reduction in area_row_usage[key].items():
                    if (end_row - start_row == tile_output_dim and 
                        used_reduction + tile_reduction_dim <= MAX_REDUCTION_DIM):
                        # 같은 row 범위에 패킹 가능!
                        area_row_usage[key][(start_row, end_row)] += tile_reduction_dim
                        
                        placement_info = {
                            'array_id': current_array,
                            'area_id': current_area,
                            'tile_info': tile_info,
                            'row_range': (start_row, end_row)
                        }
                        node_placements.append(placement_info)
                        all_tiles.append(tile_info)
                        placed = True
                        found_row = True
                        break
                
                # 새로운 row 범위 할당 시도
                if not found_row:
                    # 현재 row 위치에서 배치 가능한지 확인
                    if current_row + tile_output_dim <= MAX_OUTPUT_DIM and tile_reduction_dim <= MAX_REDUCTION_DIM:
                        # 새 row 범위 할당
                        row_range = (current_row, current_row + tile_output_dim)
                        area_row_usage[key][row_range] = tile_reduction_dim
                        
                        placement_info = {
                            'array_id': current_array,
                            'area_id': current_area,
                            'tile_info': tile_info,
                            'row_range': row_range
                        }
                        node_placements.append(placement_info)
                        all_tiles.append(tile_info)
                        placed = True
                        current_row += tile_output_dim
                    else:
                        # 현재 area가 꽉 참 → 다음 area로 이동
                        current_area += 1
                        current_row = 0
                        if current_area >= NUM_AREAS_PER_ARRAY:
                            current_area = 0
                            current_array += 1
                            if current_array >= num_arrays:
                                current_array = 0  # Wrap around
                        attempts += 1
                
                if not placed and found_row:
                    attempts += 1
            
            if not placed:
                print(f"  ❌ ERROR: Cannot place tile {tile_info['weight_id']}, Shape: {tile_shape}")
        
        placement[node_id] = node_placements
    
    # 통계 출력 - 각 array/area에 배치된 weight 상세 정보
    print()
    print("="*70)
    print("Weight Tile Placement Summary")
    print("="*70)
    print(f"Total tiles created: {len(all_tiles)}")
    print(f"Total Conv nodes: {len(conv_nodes)}")
    print()
    
    # Array/Area별 배치된 weight 정보 수집 + reduction 사용량
    array_area_weights = {}
    array_area_reduction = {}
    for node_id, node_placements in placement.items():
        for p in node_placements:
            array_id = p['array_id']
            area_id = p['area_id']
            tile_id = p['tile_info']['weight_id']
            tile_shape = p['tile_info']['shape']
            
            key = (array_id, area_id)
            if key not in array_area_weights:
                array_area_weights[key] = []
            array_area_weights[key].append(tile_id)
            
            if key not in array_area_reduction:
                array_area_reduction[key] = 0
            array_area_reduction[key] += tile_shape[1]
    
    # 간결한 출력
    for array_id in range(num_arrays):
        has_weights = any((array_id, area_id) in area_row_usage 
                         for area_id in range(NUM_AREAS_PER_ARRAY))
        
        if not has_weights:
            continue  # 빈 array는 건너뛰기
        
        print(f"\n📦 Array {array_id}")
        
        for area_id in range(NUM_AREAS_PER_ARRAY):
            key = (array_id, area_id)
            
            if key not in area_row_usage or not area_row_usage[key]:
                continue  # 빈 area는 건너뛰기
            
            # 이 area에 배치된 weight 목록
            if key in array_area_weights:
                weights = array_area_weights[key]
                reduction_used = array_area_reduction.get(key, 0)
                print(f"  Area {area_id} ({len(weights)} tiles, red={reduction_used}/{MAX_REDUCTION_DIM}):", end="")
                
                # 각 weight의 크기 정보
                weight_strs = []
                for weight_id in weights:
                    # 해당 tile의 shape 찾기
                    tile_shape = None
                    for node_id, node_placements in placement.items():
                        for p in node_placements:
                            if (p['array_id'] == array_id and 
                                p['area_id'] == area_id and 
                                p['tile_info']['weight_id'] == weight_id):
                                tile_shape = p['tile_info']['shape']
                                break
                        if tile_shape:
                            break
                    
                    # Weight ID 간결하게 표시
                    node_name = weight_id.replace('_weight', '').replace('_tile', '')
                    if '_o' in node_name and '_r' in node_name:
                        base_name = node_name.split('_o')[0]
                        tile_suffix = node_name.split('_o')[1]
                        weight_strs.append(f"{base_name}[{tile_suffix}]({tile_shape[0]}×{tile_shape[1]})" if tile_shape else f"{base_name}[{tile_suffix}]")
                    else:
                        weight_strs.append(f"{node_name}({tile_shape[0]}×{tile_shape[1]})" if tile_shape else node_name)
                
                # 한 줄에 표시
                print(" " + ", ".join(weight_strs))
    
    print(f"\n{'='*70}")
    
    return {
        'tiles': all_tiles,
        'placement': placement,
        'area_row_usage': area_row_usage
    }


def apply_weights_to_graph(graph, placement: Dict):
    """
    배치 결과를 그래프의 노드에 적용 (tile별 위치 정보 포함)
    """
    for node in graph.get_all_nodes():
        if node.node_type == "conv" and node.node_id in placement:
            node_placements = placement[node.node_id]
            
            # 첫 번째 tile의 array/area로 노드 위치 업데이트 (대표 위치)
            if node_placements:
                first_placement = node_placements[0]
                node.array_id = first_placement['array_id']
                node.area_id = first_placement['area_id']
            
            # Weight tiles를 노드에 할당 (각 tile의 배치 위치 정보 포함)
            weight_tiles = []
            current_row = 0
            for p in node_placements:
                tile_info = p['tile_info']
                tile_output_dim = tile_info['shape'][0]
                
                # Tile의 실제 배치 위치 정보를 metadata에 추가
                tile_metadata = tile_info['metadata'].copy() if tile_info['metadata'] else {}
                tile_metadata['array_id'] = p['array_id']
                tile_metadata['area_id'] = p['area_id']
                tile_metadata['row_range'] = p.get('row_range', (0, tile_output_dim))
                
                weight_tile = WeightTile(
                    weight_id=tile_info['weight_id'],
                    shape=tile_info['shape'],
                    position=(current_row, current_row + tile_output_dim),
                    metadata=tile_metadata
                )
                weight_tiles.append(weight_tile)
                current_row += tile_output_dim
            
            node.weight_tiles = weight_tiles
    
    return graph
