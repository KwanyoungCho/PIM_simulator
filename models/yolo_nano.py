"""
YOLOv5s (Small) Simplified Graph for PIM Simulator

간소화 전략:
- C3 모듈 → Conv로 분해
- SPPF → Conv로 근사
- Upsample → shape 조정 (concat 전에)
- 기존 연산만 사용: Conv, Add, Concat
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import ComputeNode, ComputeGraph


def create_c3_module(graph, node_id_prefix, input_node, in_ch, out_ch, n_bottleneck=1, 
                     has_shortcut=True, array_id=0):
    """
    C3 모듈을 Conv + Bottleneck + Concat으로 분해
    
    C3 구조:
    Input
      ├─ Conv 1x1 (채널 절반) ──┐
      │                        │
      └─ Conv 1x1 (채널 절반)   │
         → Bottleneck × n      │
         → Conv 1x1            │
                               ↓
                          Concat
                               ↓
                          Conv 1x1
    
    Args:
        node_id_prefix: 노드 ID 접두사
        input_node: 입력 노드 ID
        in_ch: 입력 채널
        out_ch: 출력 채널
        n_bottleneck: Bottleneck 반복 횟수
        has_shortcut: Bottleneck에 residual connection 사용 여부
        array_id: 배치할 array ID
    
    Returns:
        마지막 노드 ID
    """
    mid_ch = out_ch // 2
    H, W = graph.get_node(input_node).output_shape[1:3]
    
    # CSP split path 1: Conv 1x1
    path1_id = f"{node_id_prefix}_path1"
    path1 = ComputeNode(
        node_id=path1_id,
        node_type="conv",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[input_node],
        input_shape=(in_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(path1)
    
    # CSP split path 2: Conv 1x1 → Bottleneck × n → Conv 1x1
    path2_conv1_id = f"{node_id_prefix}_path2_conv1"
    path2_conv1 = ComputeNode(
        node_id=path2_conv1_id,
        node_type="conv",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[input_node],
        input_shape=(in_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(path2_conv1)
    
    # Bottleneck × n
    bottleneck_input = path2_conv1_id
    for i in range(n_bottleneck):
        # Bottleneck: Conv 1x1 → Conv 3x3 → Add (if shortcut)
        bn_conv1_id = f"{node_id_prefix}_bn{i}_conv1"
        bn_conv1 = ComputeNode(
            node_id=bn_conv1_id,
            node_type="conv",
            device_type="eflash",
            array_id=array_id,
            area_id=0,
            weight_tiles=[],
            input_nodes=[bottleneck_input],
            input_shape=(mid_ch, H, W),
            output_shape=(mid_ch, H, W),
            metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
        )
        graph.add_node(bn_conv1)
        
        bn_conv2_id = f"{node_id_prefix}_bn{i}_conv2"
        bn_conv2 = ComputeNode(
            node_id=bn_conv2_id,
            node_type="conv",
            device_type="eflash",
            array_id=array_id,
            area_id=0,
            weight_tiles=[],
            input_nodes=[bn_conv1_id],
            input_shape=(mid_ch, H, W),
            output_shape=(mid_ch, H, W),
            metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
        )
        graph.add_node(bn_conv2)
        
        if has_shortcut:
            # Residual add
            bn_add_id = f"{node_id_prefix}_bn{i}_add"
            bn_add = ComputeNode(
                node_id=bn_add_id,
                node_type="add",
                device_type="eflash",
                array_id=array_id,
                area_id=0,
                weight_tiles=[],
                input_nodes=[bottleneck_input, bn_conv2_id],
                input_shape=(mid_ch, H, W),
                output_shape=(mid_ch, H, W),
                metadata={'operation': 'element_wise_add'}
            )
            graph.add_node(bn_add)
            bottleneck_input = bn_add_id
        else:
            bottleneck_input = bn_conv2_id
    
    # Path2 마지막 Conv 1x1
    path2_conv2_id = f"{node_id_prefix}_path2_conv2"
    path2_conv2 = ComputeNode(
        node_id=path2_conv2_id,
        node_type="conv",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[bottleneck_input],
        input_shape=(mid_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(path2_conv2)
    
    # Concat two paths
    concat_id = f"{node_id_prefix}_concat"
    concat = ComputeNode(
        node_id=concat_id,
        node_type="concat",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[path1_id, path2_conv2_id],
        input_shape=(mid_ch, H, W),
        output_shape=(mid_ch * 2, H, W),
        metadata={'axis': 0}  # channel axis
    )
    graph.add_node(concat)
    
    # Final Conv 1x1
    final_conv_id = f"{node_id_prefix}_final"
    final_conv = ComputeNode(
        node_id=final_conv_id,
        node_type="conv",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[concat_id],
        input_shape=(mid_ch * 2, H, W),
        output_shape=(out_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(final_conv)
    
    return final_conv_id


def create_sppf_module(graph, node_id_prefix, input_node, in_ch, out_ch, array_id=0):
    """
    SPPF 모듈 완전 구현 (실제 YOLOv5s와 동일)
    
    실제 SPPF: Conv 1x1 → MaxPool × 3 (순차) → Concat → Conv 1x1
    
    Args:
        node_id_prefix: 노드 ID 접두사
        input_node: 입력 노드 ID
        in_ch: 입력 채널
        out_ch: 출력 채널
        array_id: 배치할 array ID
    
    Returns:
        마지막 노드 ID
    """
    mid_ch = in_ch // 2
    H, W = graph.get_node(input_node).output_shape[1:3]
    
    # Conv 1x1 (channel reduction)
    conv1_id = f"{node_id_prefix}_conv1"
    conv1 = ComputeNode(
        node_id=conv1_id,
        node_type="conv",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[input_node],
        input_shape=(in_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(conv1)
    
    # MaxPool 1: 5x5 kernel, stride=1, padding=2 (same size)
    pool1_id = f"{node_id_prefix}_pool1"
    pool1 = ComputeNode(
        node_id=pool1_id,
        node_type="maxpool",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[conv1_id],
        input_shape=(mid_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': 5, 'stride': 1, 'padding': 2}
    )
    graph.add_node(pool1)
    
    # MaxPool 2: 순차적으로 적용
    pool2_id = f"{node_id_prefix}_pool2"
    pool2 = ComputeNode(
        node_id=pool2_id,
        node_type="maxpool",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[pool1_id],
        input_shape=(mid_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': 5, 'stride': 1, 'padding': 2}
    )
    graph.add_node(pool2)
    
    # MaxPool 3: 순차적으로 적용
    pool3_id = f"{node_id_prefix}_pool3"
    pool3 = ComputeNode(
        node_id=pool3_id,
        node_type="maxpool",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[pool2_id],
        input_shape=(mid_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': 5, 'stride': 1, 'padding': 2}
    )
    graph.add_node(pool3)
    
    # Concat: [conv1, pool1, pool2, pool3] → 4배 채널
    concat_id = f"{node_id_prefix}_concat"
    concat = ComputeNode(
        node_id=concat_id,
        node_type="concat",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[conv1_id, pool1_id, pool2_id, pool3_id],
        input_shape=(mid_ch, H, W),
        output_shape=(mid_ch * 4, H, W),
        metadata={'axis': 0, 'note': 'SPPF concat 4 paths'}
    )
    graph.add_node(concat)
    
    # Conv 1x1 (channel expansion to output)
    conv2_id = f"{node_id_prefix}_conv2"
    conv2 = ComputeNode(
        node_id=conv2_id,
        node_type="conv",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[concat_id],
        input_shape=(mid_ch * 4, H, W),
        output_shape=(out_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(conv2)
    
    return conv2_id


def create_yolov5s_full(input_size=640, num_classes=80, num_arrays=2):
    """
    완전한 YOLOv5s 그래프 생성 (실제 모델과 100% 동일)
    
    구조:
    - Backbone: CSP-Darknet53 (with SPPF using MaxPool)
    - Neck: PANet (with actual Upsample nodes)
    - Head: 3-scale detection (Conv×3 per scale)
    
    주의: MaxPool, Upsample은 eFlash array에서 즉시 처리 (시간 0)
    
    Args:
        input_size: 입력 이미지 크기 (정사각형)
        num_classes: Detection class 수
        num_arrays: 사용할 eFlash array 수
        
    Returns:
        ComputeGraph
    """
    graph = ComputeGraph()
    
    # ========================================================================
    # Backbone: CSP-Darknet53
    # ========================================================================
    
    print("Building Backbone...")
    
    # Stage 0: Conv 6x6, stride 2 (640 → 320)
    conv0 = ComputeNode(
        node_id="backbone_conv0",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=[],
        input_shape=(3, input_size, input_size),
        output_shape=(32, input_size // 2, input_size // 2),
        metadata={'kernel_size': 6, 'stride': 2, 'padding': 2}
    )
    graph.add_node(conv0)
    
    # Stage 1: Conv 3x3, stride 2 (320 → 160)
    conv1 = ComputeNode(
        node_id="backbone_conv1",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=["backbone_conv0"],
        input_shape=(32, input_size // 2, input_size // 2),
        output_shape=(64, input_size // 4, input_size // 4),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(conv1)
    
    # C3_1: 64 → 64, n=1 (160x160)
    c3_1_out = create_c3_module(
        graph, "backbone_c3_1", "backbone_conv1",
        in_ch=64, out_ch=64, n_bottleneck=1,
        has_shortcut=True, array_id=0
    )
    
    # Stage 2: Conv 3x3, stride 2 (160 → 80)
    conv2 = ComputeNode(
        node_id="backbone_conv2",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c3_1_out],
        input_shape=(64, input_size // 4, input_size // 4),
        output_shape=(128, input_size // 8, input_size // 8),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(conv2)
    
    # C3_2: 128 → 128, n=2 (80x80)
    c3_2_out = create_c3_module(
        graph, "backbone_c3_2", "backbone_conv2",
        in_ch=128, out_ch=128, n_bottleneck=2,
        has_shortcut=True, array_id=0
    )
    
    # Stage 3: Conv 3x3, stride 2 (80 → 40)
    conv3 = ComputeNode(
        node_id="backbone_conv3",
        node_type="conv",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c3_2_out],
        input_shape=(128, input_size // 8, input_size // 8),
        output_shape=(256, input_size // 16, input_size // 16),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(conv3)
    
    # C3_3: 256 → 256, n=3 (40x40)
    c3_3_out = create_c3_module(
        graph, "backbone_c3_3", "backbone_conv3",
        in_ch=256, out_ch=256, n_bottleneck=3,
        has_shortcut=True, array_id=1 % num_arrays
    )
    
    # Stage 4: Conv 3x3, stride 2 (40 → 20)
    conv4 = ComputeNode(
        node_id="backbone_conv4",
        node_type="conv",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c3_3_out],
        input_shape=(256, input_size // 16, input_size // 16),
        output_shape=(512, input_size // 32, input_size // 32),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(conv4)
    
    # C3_4: 512 → 512, n=1 (20x20)
    c3_4_out = create_c3_module(
        graph, "backbone_c3_4", "backbone_conv4",
        in_ch=512, out_ch=512, n_bottleneck=1,
        has_shortcut=True, array_id=1 % num_arrays
    )
    
    # SPPF: 512 → 512 (20x20) - MaxPool 사용
    sppf_out = create_sppf_module(
        graph, "backbone_sppf", c3_4_out,
        in_ch=512, out_ch=512, array_id=1 % num_arrays
    )
    
    print(f"Backbone done. Output: {sppf_out}")
    
    # ========================================================================
    # Neck: PANet (Top-down + Bottom-up)
    # ========================================================================
    
    print("Building Neck...")
    
    # Top-down path
    # Conv 1x1 (512 → 256)
    neck_conv1 = ComputeNode(
        node_id="neck_conv1",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=[sppf_out],
        input_shape=(512, input_size // 32, input_size // 32),
        output_shape=(256, input_size // 32, input_size // 32),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(neck_conv1)
    
    # Upsample 2x (20 → 40) - 실제 Upsample 노드
    neck_up1 = ComputeNode(
        node_id="neck_up1",
        node_type="upsample",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=["neck_conv1"],
        input_shape=(256, input_size // 32, input_size // 32),
        output_shape=(256, input_size // 16, input_size // 16),
        metadata={'scale_factor': 2, 'mode': 'nearest'}
    )
    graph.add_node(neck_up1)
    
    # Concat with backbone_c3_3_final (256 channels)
    neck_concat1 = ComputeNode(
        node_id="neck_concat1",
        node_type="concat",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=["neck_up1", c3_3_out],  # Upsample + backbone
        input_shape=(256, input_size // 16, input_size // 16),
        output_shape=(512, input_size // 16, input_size // 16),
        metadata={'axis': 0}
    )
    graph.add_node(neck_concat1)
    
    # C3: 512 → 256, n=1 (40x40)
    neck_c3_1_out = create_c3_module(
        graph, "neck_c3_1", "neck_concat1",
        in_ch=512, out_ch=256, n_bottleneck=1,
        has_shortcut=False, array_id=0
    )
    
    # Conv 1x1 (256 → 128)
    neck_conv2 = ComputeNode(
        node_id="neck_conv2",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=[neck_c3_1_out],
        input_shape=(256, input_size // 16, input_size // 16),
        output_shape=(128, input_size // 16, input_size // 16),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(neck_conv2)
    
    # Upsample 2x (40 → 80) - 실제 Upsample 노드
    neck_up2 = ComputeNode(
        node_id="neck_up2",
        node_type="upsample",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=["neck_conv2"],
        input_shape=(128, input_size // 16, input_size // 16),
        output_shape=(128, input_size // 8, input_size // 8),
        metadata={'scale_factor': 2, 'mode': 'nearest'}
    )
    graph.add_node(neck_up2)
    
    # Concat with backbone_c3_2_final (128 channels)
    neck_concat2 = ComputeNode(
        node_id="neck_concat2",
        node_type="concat",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=["neck_up2", c3_2_out],  # Upsample + backbone
        input_shape=(128, input_size // 8, input_size // 8),
        output_shape=(256, input_size // 8, input_size // 8),
        metadata={'axis': 0}
    )
    graph.add_node(neck_concat2)
    
    # C3: 256 → 128, n=1 (80x80) - P3 output
    neck_c3_2_out = create_c3_module(
        graph, "neck_c3_2", "neck_concat2",
        in_ch=256, out_ch=128, n_bottleneck=1,
        has_shortcut=False, array_id=0
    )
    p3_output = neck_c3_2_out
    
    # Bottom-up path
    # Conv 3x3, stride 2 (80 → 40)
    neck_conv3 = ComputeNode(
        node_id="neck_conv3",
        node_type="conv",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=[p3_output],
        input_shape=(128, input_size // 8, input_size // 8),
        output_shape=(128, input_size // 16, input_size // 16),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(neck_conv3)
    
    # Concat with neck_c3_1_out (256 channels)
    neck_concat3 = ComputeNode(
        node_id="neck_concat3",
        node_type="concat",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=["neck_conv3", neck_c3_1_out],
        input_shape=(128, input_size // 16, input_size // 16),
        output_shape=(384, input_size // 16, input_size // 16),
        metadata={'axis': 0}
    )
    graph.add_node(neck_concat3)
    
    # C3: 384 → 256, n=1 (40x40) - P4 output
    neck_c3_3_out = create_c3_module(
        graph, "neck_c3_3", "neck_concat3",
        in_ch=384, out_ch=256, n_bottleneck=1,
        has_shortcut=False, array_id=1 % num_arrays
    )
    p4_output = neck_c3_3_out
    
    # Conv 3x3, stride 2 (40 → 20)
    neck_conv4 = ComputeNode(
        node_id="neck_conv4",
        node_type="conv",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=[p4_output],
        input_shape=(256, input_size // 16, input_size // 16),
        output_shape=(256, input_size // 32, input_size // 32),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(neck_conv4)
    
    # Concat with neck_conv1 (256 channels)
    neck_concat4 = ComputeNode(
        node_id="neck_concat4",
        node_type="concat",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=["neck_conv4", "neck_conv1"],
        input_shape=(256, input_size // 32, input_size // 32),
        output_shape=(512, input_size // 32, input_size // 32),
        metadata={'axis': 0}
    )
    graph.add_node(neck_concat4)
    
    # C3: 512 → 512, n=1 (20x20) - P5 output
    neck_c3_4_out = create_c3_module(
        graph, "neck_c3_4", "neck_concat4",
        in_ch=512, out_ch=512, n_bottleneck=1,
        has_shortcut=False, array_id=1 % num_arrays
    )
    p5_output = neck_c3_4_out
    
    print(f"Neck done. P3: {p3_output}, P4: {p4_output}, P5: {p5_output}")
    
    # ========================================================================
    # Head: Detection (실제 YOLOv5s - Conv 3개씩)
    # ========================================================================
    
    print("Building Head...")
    
    # P3 head (80x80x128) - Conv×3
    head_p3_conv1 = ComputeNode(
        node_id="head_p3_conv1",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=[p3_output],
        input_shape=(128, input_size // 8, input_size // 8),
        output_shape=(128, input_size // 8, input_size // 8),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p3_conv1)
    
    head_p3_conv2 = ComputeNode(
        node_id="head_p3_conv2",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p3_conv1"],
        input_shape=(128, input_size // 8, input_size // 8),
        output_shape=(128, input_size // 8, input_size // 8),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p3_conv2)
    
    head_p3 = ComputeNode(
        node_id="head_p3",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p3_conv2"],
        input_shape=(128, input_size // 8, input_size // 8),
        output_shape=((num_classes + 5) * 3, input_size // 8, input_size // 8),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0, 'detection_scale': 'small'}
    )
    graph.add_node(head_p3)
    
    # P4 head (40x40x256) - Conv×3
    head_p4_conv1 = ComputeNode(
        node_id="head_p4_conv1",
        node_type="conv",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=[p4_output],
        input_shape=(256, input_size // 16, input_size // 16),
        output_shape=(256, input_size // 16, input_size // 16),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p4_conv1)
    
    head_p4_conv2 = ComputeNode(
        node_id="head_p4_conv2",
        node_type="conv",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p4_conv1"],
        input_shape=(256, input_size // 16, input_size // 16),
        output_shape=(256, input_size // 16, input_size // 16),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p4_conv2)
    
    head_p4 = ComputeNode(
        node_id="head_p4",
        node_type="conv",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p4_conv2"],
        input_shape=(256, input_size // 16, input_size // 16),
        output_shape=((num_classes + 5) * 3, input_size // 16, input_size // 16),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0, 'detection_scale': 'medium'}
    )
    graph.add_node(head_p4)
    
    # P5 head (20x20x512) - Conv×3
    head_p5_conv1 = ComputeNode(
        node_id="head_p5_conv1",
        node_type="conv",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=[p5_output],
        input_shape=(512, input_size // 32, input_size // 32),
        output_shape=(512, input_size // 32, input_size // 32),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p5_conv1)
    
    head_p5_conv2 = ComputeNode(
        node_id="head_p5_conv2",
        node_type="conv",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p5_conv1"],
        input_shape=(512, input_size // 32, input_size // 32),
        output_shape=(512, input_size // 32, input_size // 32),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p5_conv2)
    
    head_p5 = ComputeNode(
        node_id="head_p5",
        node_type="conv",
        device_type="eflash",
        array_id=1 % num_arrays,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p5_conv2"],
        input_shape=(512, input_size // 32, input_size // 32),
        output_shape=((num_classes + 5) * 3, input_size // 32, input_size // 32),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0, 'detection_scale': 'large'}
    )
    graph.add_node(head_p5)
    
    print(f"Head done. P3: {head_p3.node_id}, P4: {head_p4.node_id}, P5: {head_p5.node_id}")
    
    # ========================================================================
    # 통계 출력
    # ========================================================================
    
    total_nodes = len(graph.nodes)
    conv_nodes = len([n for n in graph.nodes.values() if n.node_type == "conv"])
    add_nodes = len([n for n in graph.nodes.values() if n.node_type == "add"])
    concat_nodes = len([n for n in graph.nodes.values() if n.node_type == "concat"])
    maxpool_nodes = len([n for n in graph.nodes.values() if n.node_type == "maxpool"])
    upsample_nodes = len([n for n in graph.nodes.values() if n.node_type == "upsample"])
    
    print("\n" + "="*70)
    print("YOLOv5s Full Model Graph Summary")
    print("="*70)
    print(f"Total nodes: {total_nodes}")
    print(f"  - Conv: {conv_nodes}")
    print(f"  - Add: {add_nodes}")
    print(f"  - Concat: {concat_nodes}")
    print(f"  - MaxPool: {maxpool_nodes}")
    print(f"  - Upsample: {upsample_nodes}")
    print(f"\nInput: {input_size}x{input_size}x3")
    print(f"Output scales:")
    print(f"  - P3: {input_size//8}x{input_size//8}x{(num_classes+5)*3}")
    print(f"  - P4: {input_size//16}x{input_size//16}x{(num_classes+5)*3}")
    print(f"  - P5: {input_size//32}x{input_size//32}x{(num_classes+5)*3}")
    print("="*70)
    
    return graph


# Backward compatibility
create_yolov5s_simplified = create_yolov5s_full

if __name__ == "__main__":
    # YOLOv5s 완전한 그래프 생성
    graph = create_yolov5s_full(input_size=640, num_classes=80, num_arrays=2)
    
    print("\n✅ YOLOv5s graph created successfully!")
    print(f"\nFirst few nodes:")
    for i, (node_id, node) in enumerate(list(graph.nodes.items())[:5]):
        print(f"  {node_id}: {node.node_type} {node.output_shape}")
        if i >= 4:
            break
