"""
YOLOv8s Inference Simulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.yolov8s import create_yolov8s_full
from models.yolov8s_weight import assign_tiles_to_areas, apply_weights_to_graph
from src import (
    PIMSimulator,
    InferenceScheduler,
    InferenceContext,
    GraphPreprocessor,
    GraphValidator,
)


def setup_pim_with_yolov8s_weights(graph, num_arrays=20):
    """
    PIM 시뮬레이터 생성 및 YOLOv8s weight 배치
    
    Args:
        graph: YOLOv8s ComputeGraph
        num_arrays: eFlash array 수
        
    Returns:
        (PIMSimulator, placement_result)
    """
    print("\n[PIM 시뮬레이터 설정]")
    print(f"  - eFlash Arrays: {num_arrays}")
    print(f"  - NPUs: 0")
    print(f"  - Area execution time: 1.5 us")
    
    # PIM 생성
    pim = PIMSimulator(
        num_arrays=num_arrays,
        num_npus=0,
        area_execution_time_us=1.5,
        npu_tops=10.0,
        array_sram_size_bytes=600 * 1024 * 1024,  # 600MB per array
        npu_sram_size_bytes=2 * 1024 * 1024,      # 2MB (unused)
        shared_sram_size_bytes=100 * 1024 * 1024  # 100MB
    )
    
    # Weight tile 배치 계획
    print("\n[Weight Tile 배치 계획]")
    result = assign_tiles_to_areas(graph, num_arrays=num_arrays)
    
    # 그래프에 weight tiles 정보 추가
    print("\n[그래프에 Weight Tiles 적용]")
    apply_weights_to_graph(graph, result['placement'])
    
    # PIM Array에 실제로 weight 배치 (reduction dimension 패킹 포함)
    print("\n[PIM Array에 Weight 배치 중...]")
    placement = result['placement']
    placed_count = 0
    failed_count = 0
    
    for node_id, placements in placement.items():
        for p in placements:
            array_id = p['array_id']
            area_id = p['area_id']
            tile_info = p['tile_info']
            row_range = p.get('row_range')  # (start_row, end_row)
            
            success = pim.place_weight_on_array(
                array_id=array_id,
                area_id=area_id,
                weight_id=tile_info['weight_id'],
                shape=tile_info['shape'],
                metadata=tile_info['metadata'],
                target_row_range=row_range
            )
            
            if success:
                placed_count += 1
            else:
                failed_count += 1
                print(f"  ❌ Failed: {tile_info['weight_id']} on Array{array_id}.Area{area_id} rows{row_range}")
    
    print(f"\n✅ Weight 배치 완료: {placed_count} tiles placed, {failed_count} failed")
    
    # Weight tile 검증
    print("\n[Weight Tile 검증]")
    validator = GraphValidator(graph, pim)
    validator.validate()
    
    return pim, result


def main():
    print("=" * 80)
    print("YOLOv8s Inference Simulation")
    print("=" * 80)
    
    # 1. 그래프 생성
    print("\n[1/4] Creating YOLOv8s graph...")
    graph = create_yolov8s_full()
    print(f"  - Total nodes: {len(graph.get_all_nodes())}")
    print(f"  - Conv nodes: {len([n for n in graph.get_all_nodes() if n.node_type == 'conv'])}")
    
    # 2. PIM 설정 및 weight 배치
    print("\n[2/4] Setting up PIM simulator...")
    pim, placements = setup_pim_with_yolov8s_weights(graph, num_arrays=20)
    
    # 3. Graph preprocessing (tiled nodes → sub graph)
    print("\n[3/4] Preprocessing graph...")
    graph = GraphPreprocessor.expand_tiled_nodes(graph)
    print(f"  - Expanded nodes: {len(graph.get_all_nodes())}")
    
    # 4. Inference 실행
    print("\n[4/4] Running inference...")
    scheduler = InferenceScheduler(pim, graph, shared_sram_bandwidth_kb_per_us=3.2)
    context = InferenceContext("yolov8s", scheduler, input_batch_size=1, input_shape=(640, 640, 3))
    result = context.execute()
    
    # 4. 결과 출력
    print("\n" + "=" * 80)
    print("📊 Inference Results")
    print("=" * 80)
    print(f"Total time: {result['total_time_us']/1000:.2f} ms")
    print(f"  - Compute time: {result['total_compute_time_us']/1000:.2f} ms")
    print(f"  - Transfer time: {result['total_transfer_time_us']/1000:.2f} ms")
    print(f"Throughput: {1000000/result['total_time_us']:.2f} FPS")
    print(f"Completed nodes: {result['completed_nodes']} / {result['total_nodes']}")
    
    # 5. Timeline 출력
    # 5-1. 노드별 Execution Timeline (Transfer + Compute)
    scheduler.print_timeline(max_nodes=990, timeline=context.timeline)
    
    # 5-2. 메모리 Timeline (All locations)
    scheduler.print_memory_timeline(max_events=900, location_filter=None, events=context.memory_events)
    
    # 5-3. 연산/전송 중첩 구간 분석
    scheduler.print_overlap_summary(
        min_duration_us=0.0,
        max_segments=0,
        show_all_segments=True,
        timeline=context.timeline
    )
  
    print("\n" + "=" * 80)
    print("✅ Simulation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
