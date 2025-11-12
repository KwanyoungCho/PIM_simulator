"""
Event-driven Inference 시뮬레이션 예제
간단한 CNN 그래프를 사용하여 dependency-aware 스케줄링 테스트
"""

from src.pim_simulator import PIMSimulator
from src.compute_node import ComputeNode, ComputeGraph
from src.scheduler import InferenceScheduler
from src.inference_context import InferenceContext, PipelineManager


def create_simple_cnn_graph():
    """
    간단한 CNN 그래프 생성
    
    구조:
    Input (224x224x3)
      |
      v
    Conv1 (Array0, Area0) -> 64x112x112
      |
      +---> Conv2a (Array0, Area1) -> 128x56x56
      |
      +---> Conv2b (Array1, Area0) -> 64x112x112  (병렬)
              |
              v
            Conv3 (Array1, Area1) -> 128x56x56
              |
              v
            Add (Conv2a + Conv3) -> 128x56x56
              |
              v
            Conv4 (Array0, Area2) -> 256x28x28
    """
    graph = ComputeGraph()
    
    # Conv1: 입력 처리
    conv1 = ComputeNode(
        node_id="conv1",
        node_type="conv",
        array_id=0,
        area_id=0,
        weight_tiles=["conv1_w"],
        input_nodes=[],  # 소스 노드
        input_shape=(224, 224, 3),
        output_shape=(64, 112, 112),
        metadata={'kernel': '3x3', 'stride': 2}
    )
    graph.add_node(conv1)
    
    # Conv2a: Conv1의 출력 사용
    conv2a = ComputeNode(
        node_id="conv2a",
        node_type="conv",
        array_id=0,
        area_id=1,
        weight_tiles=["conv2a_w"],
        input_nodes=["conv1"],
        input_shape=(64, 112, 112),
        output_shape=(128, 56, 56),
        metadata={'kernel': '3x3', 'stride': 2}
    )
    graph.add_node(conv2a)
    
    # Conv2b: Conv1의 출력 사용 (병렬 브랜치)
    conv2b = ComputeNode(
        node_id="conv2b",
        node_type="conv",
        array_id=1,
        area_id=0,
        weight_tiles=["conv2b_w"],
        input_nodes=["conv1"],
        input_shape=(64, 112, 112),
        output_shape=(64, 112, 112),
        metadata={'kernel': '1x1', 'stride': 1}
    )
    graph.add_node(conv2b)
    
    # Conv3: Conv2b의 출력 사용
    conv3 = ComputeNode(
        node_id="conv3",
        node_type="conv",
        array_id=1,
        area_id=1,
        weight_tiles=["conv3_w"],
        input_nodes=["conv2b"],
        input_shape=(64, 112, 112),
        output_shape=(128, 56, 56),
        metadata={'kernel': '3x3', 'stride': 2}
    )
    graph.add_node(conv3)
    
    # Add: Residual connection (conv2a + conv3)
    add1 = ComputeNode(
        node_id="add1",
        node_type="add",
        array_id=0,
        area_id=3,
        weight_tiles=[],  # Add는 weight 없음
        input_nodes=["conv2a", "conv3"],
        input_shape=(128, 56, 56),
        output_shape=(128, 56, 56),
        metadata={'operation': 'elementwise_add'}
    )
    graph.add_node(add1)
    
    # Conv4: Add 결과 사용
    conv4 = ComputeNode(
        node_id="conv4",
        node_type="conv",
        array_id=0,
        area_id=2,
        weight_tiles=["conv4_w"],
        input_nodes=["add1"],
        input_shape=(128, 56, 56),
        output_shape=(256, 28, 28),
        metadata={'kernel': '3x3', 'stride': 2}
    )
    graph.add_node(conv4)
    
    return graph


def setup_pim_with_weights(graph: ComputeGraph):
    """
    PIM 시뮬레이터 생성 및 weight 배치
    
    Args:
        graph: 연산 그래프
        
    Returns:
        PIMSimulator
    """
    # PIM 생성 (2개 Array, 각 Area 실행시간 100ns)
    pim = PIMSimulator(
        num_arrays=2,
        area_execution_time_ns=100.0,
        array_sram_size_bytes=2 * 1024 * 1024,  # 2MB
        shared_sram_size_bytes=20 * 1024 * 1024  # 20MB
    )
    
    # Weight 배치 (그래프 노드 정보 기반)
    weight_placements = [
        ("conv1_w", 0, 0, (64, 512)),    # Array0, Area0
        ("conv2a_w", 0, 1, (128, 1024)), # Array0, Area1
        ("conv2b_w", 1, 0, (64, 256)),   # Array1, Area0
        ("conv3_w", 1, 1, (128, 1024)),  # Array1, Area1
        ("conv4_w", 0, 2, (128, 1280)),  # Array0, Area2
    ]
    
    print("\n[Weight 배치]")
    for weight_id, array_id, area_id, shape in weight_placements:
        success = pim.place_weight_on_array(
            array_id=array_id,
            area_id=area_id,
            weight_id=weight_id,
            shape=shape,
            metadata={'layer': weight_id}
        )
        if success:
            print(f"✓ {weight_id:15s} -> Array {array_id}, Area {area_id}, shape {shape}")
        else:
            print(f"✗ {weight_id:15s} 배치 실패")
    
    return pim


def main():
    print("=" * 80)
    print("EVENT-DRIVEN INFERENCE SIMULATION")
    print("=" * 80)
    
    # 1. 그래프 생성
    print("\n[1. 연산 그래프 생성]")
    graph = create_simple_cnn_graph()
    print(f"총 노드 수: {len(graph.nodes)}")
    for node_id, node in graph.nodes.items():
        deps = ", ".join(node.input_nodes) if node.input_nodes else "None"
        print(f"  - {node_id:10s}: deps=[{deps:20s}], Array{node.array_id}.Area{node.area_id}")
    
    # 2. PIM 설정 및 weight 배치
    print("\n[2. PIM 시뮬레이터 설정]")
    pim = setup_pim_with_weights(graph)
    
    # 3. 스케줄러 생성
    print("\n[3. 스케줄러 생성]")
    scheduler = InferenceScheduler(
        pim_simulator=pim,
        compute_graph=graph,
        shared_sram_bandwidth_gbps=10.0  # 10 GB/s
    )
    print(f"✓ 스케줄러 생성 완료: {scheduler}")
    
    # 4. 단일 Inference 실행
    print("\n[4. Inference 실행]")
    print("-" * 80)
    
    context = InferenceContext(
        context_id="inference_0",
        scheduler=scheduler,
        input_batch_size=1,
        input_shape=(224, 224, 3)
    )
    
    result = context.execute()
    
    print(f"\n실행 완료!")
    print(f"  - 총 실행 시간: {result['total_time_ns']:.2f} ns ({result['total_time_ns']/1e6:.4f} ms)")
    print(f"  - 연산 시간: {result['total_compute_time_ns']:.2f} ns")
    print(f"  - 전송 시간: {result['total_transfer_time_ns']:.2f} ns")
    print(f"  - 완료된 노드: {result['completed_nodes']}/{result['total_nodes']}")
    print(f"\n[메모리 관리 개선]")
    print(f"  - 해제된 activation: {result['deallocated_count']}")
    print(f"  - 해제된 메모리: {result['deallocated_bytes'] / 1024:.2f} KB")
    
    # 5. 타임라인 출력
    scheduler.print_timeline()
    
    # 6. 그래프 실행 순서 확인
    print("\n[실행 순서]")
    exec_order = result['graph_stats']['execution_order']
    if exec_order:
        for i, node_id in enumerate(exec_order, 1):
            node = graph.get_node(node_id)
            if node:
                exec_time = node.get_execution_time()
                print(f"  {i}. {node_id:10s}: {node.start_time_ns:8.2f}ns ~ {node.end_time_ns:8.2f}ns "
                      f"(duration: {exec_time:.2f}ns) [Array{node.array_id}]")
    else:
        print("  (실행 순서 정보 없음)")
    
    # 7. Activation 메모리 사용량 및 저장 위치
    print("\n[Activation 저장 위치 분석]")
    print("  (Multi-location 지원: 같은 activation이 여러 SRAM에 저장 가능)")
    
    # Activation lifetime 정보 출력
    if hasattr(scheduler, 'activation_lifetimes'):
        print(f"  \n현재 남아있는 activations:")
        for buf_id, lifetime in scheduler.activation_lifetimes.items():
            locations = list(lifetime['storage_locations'].keys())
            consumers = lifetime['consumers']
            print(f"    - {buf_id}: {locations}")
            print(f"      Consumers: {consumers}, Ref count: {lifetime['ref_count']}")
    
    print("\n[Activation 메모리 통계]")
    act_stats = result['activation_stats']
    print(f"  - 총 버퍼 수: {act_stats['num_buffers']}")
    print(f"  - 총 크기: {act_stats['total_bytes'] / 1024:.2f} KB")
    
    # 8. SRAM 사용률
    print("\n[SRAM 사용률]")
    shared_sram = pim.get_shared_sram()
    sram_stats = shared_sram.get_stats()
    print(f"  - 공유 SRAM: {sram_stats['used_bytes']/1024:.2f} KB / "
          f"{sram_stats['total_size_bytes']/1024/1024:.2f} MB "
          f"({sram_stats['utilization']:.2%})")
    
    for i in range(pim.num_arrays):
        array = pim.get_array(i)
        array_sram = array.get_sram()
        array_stats = array_sram.get_stats()
        print(f"  - Array {i} SRAM: {array_stats['used_bytes']/1024:.2f} KB / "
              f"{array_stats['total_size_bytes']/1024/1024:.2f} MB "
              f"({array_stats['utilization']:.2%})")
    
    # 9. Pipeline 시뮬레이션 (여러 입력)
    print("\n" + "=" * 80)
    print("PIPELINE SIMULATION (3 consecutive inferences)")
    print("=" * 80)
    
    pipeline = PipelineManager(scheduler)
    
    # 3개의 inference 추가
    for i in range(3):
        pipeline.add_inference(
            context_id=f"inference_{i}",
            input_batch_size=1,
            input_shape=(224, 224, 3)
        )
    
    # 순차 실행
    pipeline_results = pipeline.run_sequential()
    
    print(f"\n파이프라인 완료!")
    pipeline_stats = pipeline.get_stats()
    print(f"  - 완료된 inference: {pipeline_stats['completed_contexts']}")
    print(f"  - 평균 레이턴시: {pipeline_stats['average_latency_ns']:.2f} ns "
          f"({pipeline_stats['average_latency_ns']/1e6:.4f} ms)")
    print(f"  - 처리량: {pipeline_stats['throughput_per_sec']:.2f} inferences/sec")
    
    print("\n" + "=" * 80)
    print("시뮬레이션 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
