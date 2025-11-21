"""
YOLOv8s (Small) Graph for PIM Simulator

YOLOv8 특징:
- C3 → C2f 모듈 (C3보다 가벼운 구조)
- Anchor-free detection head
- Focus 모듈 없음 (Conv로 시작)

간소화 전략:
- C2f 모듈 → Conv로 분해
- SPPF → Conv + MaxPool로 근사
- Decoupled head → Conv로 근사
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import ComputeNode, ComputeGraph


def create_c2f_module(graph, node_id_prefix, input_node, in_ch, out_ch, n_bottleneck=1,
                      has_shortcut=True, array_id=0):
    """
    C2f 모듈을 Conv + Bottleneck + Concat으로 분해
    
    C2f 구조 (Faster C3):
    Input
      ├─ Conv 1x1 (채널 절반) ──┐
      │                        │
      └─ Conv 1x1 (채널 절반)   │
         → Bottleneck × n      │
         (각 bottleneck 출력도 concat)
                               ↓
                          Concat (all)
                               ↓
                          Conv 1x1
    """
    mid_ch = out_ch // 2
    H, W = graph.get_node(input_node).output_shape[1:3]
    
    # Path 1: Conv 1x1 (bypass)
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
    
    # Path 2: Conv 1x1 → Bottlenecks
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
    
    # Bottlenecks (각 출력을 concat에 사용)
    concat_inputs = [path1_id, path2_conv1_id]
    bottleneck_input = path2_conv1_id
    
    for i in range(n_bottleneck):
        # Bottleneck: Conv 3x3 → Conv 3x3
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
            metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
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
        
        concat_inputs.append(bn_conv2_id)
        bottleneck_input = bn_conv2_id
    
    # Concat all paths (같은 array에서 실행)
    concat_id = f"{node_id_prefix}_concat"
    concat_ch = mid_ch * (2 + n_bottleneck)
    concat_node = ComputeNode(
        node_id=concat_id,
        node_type="concat",
        device_type="eflash",
        array_id=array_id,  # 같은 array 유지
        area_id=0,
        weight_tiles=[],
        input_nodes=concat_inputs,
        input_shape=(concat_ch, H, W),
        output_shape=(concat_ch, H, W),
        metadata={'axis': 0}
    )
    graph.add_node(concat_node)
    
    # Final Conv 1x1
    final_id = f"{node_id_prefix}_final"
    final_conv = ComputeNode(
        node_id=final_id,
        node_type="conv",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[concat_id],
        input_shape=(concat_ch, H, W),
        output_shape=(out_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(final_conv)
    
    return final_id


def create_sppf_module(graph, node_id_prefix, input_node, in_ch, out_ch, pool_size=5, array_id=0):
    """
    SPPF (Spatial Pyramid Pooling - Fast) 모듈
    
    구조: Conv 1x1 → MaxPool → MaxPool → MaxPool → Concat → Conv 1x1
    """
    H, W = graph.get_node(input_node).output_shape[1:3]
    mid_ch = in_ch // 2
    
    # Conv 1x1
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
    
    # MaxPool 1
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
        metadata={'kernel_size': pool_size, 'stride': 1, 'padding': pool_size//2}
    )
    graph.add_node(pool1)
    
    # MaxPool 2
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
        metadata={'kernel_size': pool_size, 'stride': 1, 'padding': pool_size//2}
    )
    graph.add_node(pool2)
    
    # MaxPool 3
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
        metadata={'kernel_size': pool_size, 'stride': 1, 'padding': pool_size//2}
    )
    graph.add_node(pool3)
    
    # Concat (같은 array에서 실행)
    concat_id = f"{node_id_prefix}_concat"
    concat_ch = mid_ch * 4
    concat_node = ComputeNode(
        node_id=concat_id,
        node_type="concat",
        device_type="eflash",
        array_id=array_id,  # 같은 array 유지
        area_id=0,
        weight_tiles=[],
        input_nodes=[conv1_id, pool1_id, pool2_id, pool3_id],
        input_shape=(concat_ch, H, W),
        output_shape=(concat_ch, H, W),
        metadata={'axis': 0}
    )
    graph.add_node(concat_node)
    
    # Final Conv 1x1
    conv2_id = f"{node_id_prefix}_conv2"
    conv2 = ComputeNode(
        node_id=conv2_id,
        node_type="conv",
        device_type="eflash",
        array_id=array_id,
        area_id=0,
        weight_tiles=[],
        input_nodes=[concat_id],
        input_shape=(concat_ch, H, W),
        output_shape=(out_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(conv2)
    
    return conv2_id


def create_yolov8s_full():
    """
    YOLOv8s 전체 네트워크 생성
    
    구조:
    - Backbone: Conv → C2f × 4 → SPPF
    - Neck: Upsample + Concat + C2f × 2, Conv + Concat + C2f × 2
    - Head: Detect × 3 (P3, P4, P5)
    
    Returns:
        ComputeGraph
    """
    graph = ComputeGraph()
    
    # ========================================
    # Input
    # ========================================
    input_node = ComputeNode(
        node_id="input",
        node_type="input",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=[],
        input_shape=(3, 640, 640),
        output_shape=(3, 640, 640),
        metadata={}
    )
    graph.add_node(input_node)
    
    # ========================================
    # Backbone
    # ========================================
    
    # Stem: Conv 3x3 (stride=2)
    backbone_conv1 = ComputeNode(
        node_id="backbone_conv1",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=["input"],
        input_shape=(3, 640, 640),
        output_shape=(32, 320, 320),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(backbone_conv1)
    
    # Stage 1: Conv + C2f
    backbone_conv2 = ComputeNode(
        node_id="backbone_conv2",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=["backbone_conv1"],
        input_shape=(32, 320, 320),
        output_shape=(64, 160, 160),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(backbone_conv2)
    
    c2f_1 = create_c2f_module(graph, "backbone_c2f_1", "backbone_conv2", 
                              in_ch=64, out_ch=64, n_bottleneck=1, array_id=0)
    
    # Stage 2: Conv + C2f
    backbone_conv3 = ComputeNode(
        node_id="backbone_conv3",
        node_type="conv",
        device_type="eflash",
        array_id=0,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c2f_1],
        input_shape=(64, 160, 160),
        output_shape=(128, 80, 80),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(backbone_conv3)
    
    c2f_2 = create_c2f_module(graph, "backbone_c2f_2", "backbone_conv3",
                              in_ch=128, out_ch=128, n_bottleneck=2, array_id=0)
    
    # Stage 3: Conv + C2f (P3)
    backbone_conv4 = ComputeNode(
        node_id="backbone_conv4",
        node_type="conv",
        device_type="eflash",
        array_id=2,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c2f_2],
        input_shape=(128, 80, 80),
        output_shape=(256, 40, 40),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(backbone_conv4)
    
    c2f_3 = create_c2f_module(graph, "backbone_c2f_3", "backbone_conv4",
                              in_ch=256, out_ch=256, n_bottleneck=2, array_id=1)
    
    # Stage 4: Conv + C2f (P4)
    backbone_conv5 = ComputeNode(
        node_id="backbone_conv5",
        node_type="conv",
        device_type="eflash",
        array_id=3,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c2f_3],
        input_shape=(256, 40, 40),
        output_shape=(512, 20, 20),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(backbone_conv5)
    
    c2f_4 = create_c2f_module(graph, "backbone_c2f_4", "backbone_conv5",
                              in_ch=512, out_ch=512, n_bottleneck=1, array_id=3)
    
    # SPPF (P5)
    sppf = create_sppf_module(graph, "backbone_sppf", c2f_4,
                             in_ch=512, out_ch=512, pool_size=5, array_id=4)
    
    # ========================================
    # Neck (PAN-FPN)
    # ========================================
    
    # Upsample 1 (SPPF와 같은 array=4)
    neck_up1 = ComputeNode(
        node_id="neck_up1",
        node_type="upsample",
        device_type="eflash",
        array_id=4,  # SPPF와 같은 array
        area_id=0,
        weight_tiles=[],
        input_nodes=[sppf],
        input_shape=(512, 20, 20),
        output_shape=(512, 40, 40),
        metadata={'scale_factor': 2}
    )
    graph.add_node(neck_up1)
    
    # Concat 1 (P5 + P4, neck_up1과 같은 array=4)
    neck_concat1 = ComputeNode(
        node_id="neck_concat1",
        node_type="concat",
        device_type="eflash",
        array_id=4,  # neck_up1과 같은 array
        area_id=0,
        weight_tiles=[],
        input_nodes=["neck_up1", c2f_3],
        input_shape=(768, 40, 40),
        output_shape=(768, 40, 40),
        metadata={'axis': 0}
    )
    graph.add_node(neck_concat1)
    
    # C2f 5
    c2f_5 = create_c2f_module(graph, "neck_c2f_5", "neck_concat1",
                              in_ch=768, out_ch=256, n_bottleneck=1, array_id=5)
    
    # Upsample 2 (c2f_5와 같은 array=5)
    neck_up2 = ComputeNode(
        node_id="neck_up2",
        node_type="upsample",
        device_type="eflash",
        array_id=5,  # c2f_5와 같은 array
        area_id=0,
        weight_tiles=[],
        input_nodes=[c2f_5],
        input_shape=(256, 40, 40),
        output_shape=(256, 80, 80),
        metadata={'scale_factor': 2}
    )
    graph.add_node(neck_up2)
    
    # Concat 2 (P4 + P3, neck_up2와 같은 array=5)
    neck_concat2 = ComputeNode(
        node_id="neck_concat2",
        node_type="concat",
        device_type="eflash",
        array_id=5,  # neck_up2와 같은 array
        area_id=0,
        weight_tiles=[],
        input_nodes=["neck_up2", c2f_2],
        input_shape=(384, 80, 80),
        output_shape=(384, 80, 80),
        metadata={'axis': 0}
    )
    graph.add_node(neck_concat2)
    
    # C2f 6 (P3 output)
    c2f_6 = create_c2f_module(graph, "neck_c2f_6", "neck_concat2",
                              in_ch=384, out_ch=128, n_bottleneck=1, array_id=5)
    
    # Conv down 1
    neck_conv1 = ComputeNode(
        node_id="neck_conv1",
        node_type="conv",
        device_type="eflash",
        array_id=7,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c2f_6],
        input_shape=(128, 80, 80),
        output_shape=(128, 40, 40),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(neck_conv1)
    
    # Concat 3 (neck_conv1과 같은 array=7)
    neck_concat3 = ComputeNode(
        node_id="neck_concat3",
        node_type="concat",
        device_type="eflash",
        array_id=5,  # neck_conv1과 같은 array
        area_id=0,
        weight_tiles=[],
        input_nodes=["neck_conv1", c2f_5],
        input_shape=(384, 40, 40),
        output_shape=(384, 40, 40),
        metadata={'axis': 0}
    )
    graph.add_node(neck_concat3)
    
    # C2f 7 (P4 output)
    c2f_7 = create_c2f_module(graph, "neck_c2f_7", "neck_concat3",
                              in_ch=384, out_ch=256, n_bottleneck=1, array_id=6)
    
    # Conv down 2
    neck_conv2 = ComputeNode(
        node_id="neck_conv2",
        node_type="conv",
        device_type="eflash",
        array_id=9,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c2f_7],
        input_shape=(256, 40, 40),
        output_shape=(256, 20, 20),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(neck_conv2)
    
    # Concat 4 (neck_conv2와 같은 array=9)
    neck_concat4 = ComputeNode(
        node_id="neck_concat4",
        node_type="concat",
        device_type="eflash",
        array_id=9,  # neck_conv2와 같은 array
        area_id=0,
        weight_tiles=[],
        input_nodes=["neck_conv2", sppf],
        input_shape=(768, 20, 20),
        output_shape=(768, 20, 20),
        metadata={'axis': 0}
    )
    graph.add_node(neck_concat4)
    
    # C2f 8 (P5 output)
    c2f_8 = create_c2f_module(graph, "neck_c2f_8", "neck_concat4",
                              in_ch=768, out_ch=512, n_bottleneck=1, array_id=8)
    
    # ========================================
    # Detection Head (Decoupled)
    # ========================================
    
    # P3 Head (80x80)
    head_p3_conv1 = ComputeNode(
        node_id="head_p3_conv1",
        node_type="conv",
        device_type="eflash",
        array_id=5,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c2f_6],
        input_shape=(128, 80, 80),
        output_shape=(128, 80, 80),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p3_conv1)
    
    head_p3_conv2 = ComputeNode(
        node_id="head_p3_conv2",
        node_type="conv",
        device_type="eflash",
        array_id=11,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p3_conv1"],
        input_shape=(128, 80, 80),
        output_shape=(64, 80, 80),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p3_conv2)
    
    head_p3 = ComputeNode(
        node_id="head_p3",
        node_type="conv",
        device_type="eflash",
        array_id=11,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p3_conv2"],
        input_shape=(64, 80, 80),
        output_shape=(80, 80, 80),  # 80 classes (COCO)
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(head_p3)
    
    # P4 Head (40x40)
    head_p4_conv1 = ComputeNode(
        node_id="head_p4_conv1",
        node_type="conv",
        device_type="eflash",
        array_id=12,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c2f_7],
        input_shape=(256, 40, 40),
        output_shape=(256, 40, 40),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p4_conv1)
    
    head_p4_conv2 = ComputeNode(
        node_id="head_p4_conv2",
        node_type="conv",
        device_type="eflash",
        array_id=12,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p4_conv1"],
        input_shape=(256, 40, 40),
        output_shape=(128, 40, 40),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p4_conv2)
    
    head_p4 = ComputeNode(
        node_id="head_p4",
        node_type="conv",
        device_type="eflash",
        array_id=12,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p4_conv2"],
        input_shape=(128, 40, 40),
        output_shape=(80, 40, 40),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(head_p4)
    
    # P5 Head (20x20)
    head_p5_conv1 = ComputeNode(
        node_id="head_p5_conv1",
        node_type="conv",
        device_type="eflash",
        array_id=13,
        area_id=0,
        weight_tiles=[],
        input_nodes=[c2f_8],
        input_shape=(512, 20, 20),
        output_shape=(512, 20, 20),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p5_conv1)
    
    head_p5_conv2 = ComputeNode(
        node_id="head_p5_conv2",
        node_type="conv",
        device_type="eflash",
        array_id=13,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p5_conv1"],
        input_shape=(512, 20, 20),
        output_shape=(256, 20, 20),
        metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
    )
    graph.add_node(head_p5_conv2)
    
    head_p5 = ComputeNode(
        node_id="head_p5",
        node_type="conv",
        device_type="eflash",
        array_id=13,
        area_id=0,
        weight_tiles=[],
        input_nodes=["head_p5_conv2"],
        input_shape=(256, 20, 20),
        output_shape=(80, 20, 20),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(head_p5)
    
    return graph
