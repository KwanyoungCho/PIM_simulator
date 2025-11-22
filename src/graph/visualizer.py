"""
Graph and Weight Placement Visualizer

PIM 시뮬레이터의 weight 배치 상태를 시각화
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, Any


def visualize_weight_placement(pim, placement_result: Dict[str, Any], 
                               output_file: str = 'weight_placement.png'):
    """
    PIM array의 weight 배치를 시각화
    
    Args:
        pim: PIMSimulator instance
        placement_result: assign_tiles_to_areas의 결과
        output_file: 저장할 이미지 파일명
    """
    print("\n[Weight Placement 시각화]")
    
    # 배치 정보 수집
    array_data = {}  # array_id -> {area_id -> [tiles]}
    
    for node_id, placements in placement_result['placement'].items():
        for p in placements:
            array_id = p['array_id']
            area_id = p['area_id']
            tile_info = p['tile_info']
            row_range = p.get('row_range', (0, tile_info['shape'][0]))
            
            if array_id not in array_data:
                array_data[array_id] = {}
            if area_id not in array_data[array_id]:
                array_data[array_id][area_id] = []
            
            array_data[array_id][area_id].append({
                'weight_id': tile_info['weight_id'],
                'shape': tile_info['shape'],  # (output_dim, reduction_dim)
                'row_range': row_range,
                'node_id': node_id
            })
    
    # 사용된 array 개수
    num_arrays_used = len(array_data)
    
    # Grid 설정 (4열)
    cols = 4
    rows = (num_arrays_used + cols - 1) // cols
    
    fig = plt.figure(figsize=(20, 5 * rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # 색상 맵 (utilization)
    cmap = plt.cm.RdYlGn
    
    # eFlashArray의 Area 속성 가져오기
    sample_area = pim.eflash_arrays[0].areas[0]
    MAX_OUTPUT_DIM = sample_area.MAX_OUTPUT_DIM
    MAX_REDUCTION_DIM = sample_area.MAX_REDUCTION_DIM
    AREAS_PER_ARRAY = pim.eflash_arrays[0].NUM_AREAS
    
    for idx, array_id in enumerate(sorted(array_data.keys())):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Array 전체 크기 (8 areas × output_dim)
        total_height = AREAS_PER_ARRAY * MAX_OUTPUT_DIM
        total_width = MAX_REDUCTION_DIM
        
        # 배경 (빈 공간은 흰색)
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_height)
        ax.set_aspect('auto')
        
        # Area 구분선 그리기
        for area_idx in range(AREAS_PER_ARRAY + 1):
            y = area_idx * MAX_OUTPUT_DIM
            ax.axhline(y=y, color='black', linewidth=1.5, alpha=0.7)
        
        # Area별 weight tile 그리기
        area_tiles = array_data[array_id]
        total_tiles = 0
        total_reduction_used = 0
        
        for area_id in range(AREAS_PER_ARRAY):
            if area_id not in area_tiles:
                continue
            
            tiles = area_tiles[area_id]
            total_tiles += len(tiles)
            
            # Area 기준 y offset
            area_y_offset = area_id * MAX_OUTPUT_DIM
            
            # Tile들을 row_range 순으로 정렬
            tiles_sorted = sorted(tiles, key=lambda t: (t['row_range'][0], t['row_range'][1]))
            
            # 같은 row에서 reduction dimension 누적을 추적
            row_red_offset = {}  # {(row_start, row_end): current_x_offset}
            
            for tile in tiles_sorted:
                output_dim, reduction_dim = tile['shape']
                row_start, row_end = tile['row_range']
                row_key = (row_start, row_end)
                
                # 위치 계산
                y = area_y_offset + row_start
                height = row_end - row_start
                width = reduction_dim
                
                # 같은 row_range의 이전 tile 뒤에 배치
                x_offset = row_red_offset.get(row_key, 0)
                row_red_offset[row_key] = x_offset + reduction_dim
                
                total_reduction_used += reduction_dim
                
                # Utilization (reduction dimension 기준)
                utilization = reduction_dim / MAX_REDUCTION_DIM
                color = cmap(utilization)
                
                # 사각형 그리기
                rect = mpatches.Rectangle(
                    (x_offset, y), width, height,
                    linewidth=0.5,
                    edgecolor='black',
                    facecolor=color,
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # 텍스트 (weight_id, shape)
                # weight_id 간소화
                weight_id = tile['weight_id']
                
                # backbone_, neck_, head_, fused_ 제거
                weight_name = weight_id.replace('backbone_', '').replace('neck_', '').replace('head_', '')
                weight_name = weight_name.replace('fused_', '')
                
                # weight 단어 제거
                weight_name = weight_name.replace('_weight', '').replace('weight_', '')
                
                # 타일 번호 처리 (명확한 구분자 사용)
                import re
                
                tile_suffix = ""
                
                # partial 단어 제거 (partial은 같은 weight의 일부이므로 구별 안함)
                weight_name = re.sub(r'_?partial\d+', '', weight_name)
                
                # [#_r#] 패턴 처리
                bracket_match = re.search(r'\[(\d+)_r(\d+)\]', weight_name)
                if bracket_match:
                    o_idx = int(bracket_match.group(1))
                    r_idx = int(bracket_match.group(2))
                    # 2진법 방식: partial 무시하고 [o_r#]만으로 tile 번호 결정
                    tile_num = o_idx * 2 + r_idx
                    tile_suffix = f".t{tile_num}"
                    base_name = weight_name.split('[')[0]
                    weight_name = base_name + tile_suffix
                else:
                    # _tile_o#_r# 패턴 처리
                    tile_match = re.search(r'_tile_o(\d+)_r(\d+)', weight_name)
                    if tile_match:
                        o_idx = int(tile_match.group(1))
                        r_idx = int(tile_match.group(2))
                        tile_num = o_idx * 2 + r_idx
                        tile_suffix = f".t{tile_num}"
                        base_name = re.sub(r'_tile_o\d+_r\d+', '', weight_name)
                        weight_name = base_name + tile_suffix
                
                # 너무 길면 축약 (16자 이상)
                if len(weight_name) > 16:
                    parts = weight_name.split('_')
                    # .t# 부분 분리
                    has_tile_suffix = '.t' in weight_name
                    if has_tile_suffix:
                        tile_part = weight_name[weight_name.rfind('.t'):]
                        parts = weight_name[:weight_name.rfind('.t')].split('_')
                    else:
                        tile_part = ""
                    
                    # 주요 부분만 유지
                    if len(parts) > 3:
                        weight_name = '_'.join(parts[:3]) + tile_part
                    else:
                        weight_name = '_'.join(parts) + tile_part
                
                # 언더바를 점으로 변경
                weight_name = weight_name.replace('_', '.')
                
                text = f"{weight_name}\n{output_dim}×{reduction_dim}"
                
                # 텍스트 위치 (중앙)
                text_x = x_offset + width / 2
                text_y = y + height / 2
                
                # 폰트 크기 조절 (tile 크기에 따라 - width와 height 모두 고려)
                height_based = height / 18  # 더 크게 (20 → 18)
                width_based = width / 120   # 더 크게 (150 → 120)
                fontsize = min(8, max(4, min(height_based, width_based)))  # 범위 확대 (3.5~6 → 4~8)
                
                ax.text(
                    text_x, text_y, text,
                    ha='center', va='center',
                    fontsize=fontsize,
                    color='black',
                    weight='bold'
                )
        
        # 축 설정
        ax.set_xlabel('Reduction Dim', fontsize=10)
        ax.set_ylabel('Output Dim (by Area)', fontsize=10)
        ax.set_title(
            f'Array {array_id} ({total_tiles} tiles)',
            fontsize=12, fontweight='bold'
        )
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Area 레이블 추가 (오른쪽)
        for area_idx in range(AREAS_PER_ARRAY):
            y = area_idx * MAX_OUTPUT_DIM + MAX_OUTPUT_DIM / 2
            ax.text(
                total_width * 1.02, y, f'A{area_idx}',
                ha='left', va='center',
                fontsize=8, color='blue',
                weight='bold'
            )
    
    # Colorbar (utilization)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.get_axes(), orientation='horizontal', 
                       fraction=0.05, pad=0.05, aspect=40)
    cbar.set_label('Reduction Dimension Utilization', fontsize=12)
    
    # 전체 제목
    total_tiles = len(placement_result.get('tiles', []))
    fig.suptitle(
        f'YOLOv8s Weight Placement Visualization\n'
        f'Arrays: {num_arrays_used}, Total Tiles: {total_tiles}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # 저장
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_file}")
    
    # 화면에 표시
    plt.show()
