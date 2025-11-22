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
from models.yolov8s_utils import (
    create_c2f_module,
    create_fused_concat_c2f_module,
    create_fused_up_concat_c2f_module,
    create_sppf_module
)


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
        device_type="npu",
        npu_id=0,
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
    # NPU mode (comment out for eFlash mode)
    backbone_conv1 = ComputeNode(
        node_id="backbone_conv1",
        node_type="conv",
        device_type="npu",
        npu_id=0,
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
        device_type="npu",
        npu_id=0,
        weight_tiles=[],
        input_nodes=["backbone_conv1"],
        input_shape=(32, 320, 320),
        output_shape=(64, 160, 160),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(backbone_conv2)
    
    c2f_1 = create_c2f_module(graph, "backbone_c2f_1", "backbone_conv2", 
                              in_ch=64, out_ch=64, n_bottleneck=1, device_type="npu")
    
    # Stage 2: Conv + C2f
    backbone_conv3 = ComputeNode(
        node_id="backbone_conv3",
        node_type="conv",
        device_type="npu",
        npu_id=0,
        weight_tiles=[],
        input_nodes=[c2f_1],
        input_shape=(64, 160, 160),
        output_shape=(128, 80, 80),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(backbone_conv3)
    
    c2f_2 = create_c2f_module(graph, "backbone_c2f_2", "backbone_conv3",
                              in_ch=128, out_ch=128, n_bottleneck=2, device_type="npu")
    
    # Stage 3: Conv + C2f (P3)
    backbone_conv4 = ComputeNode(
        node_id="backbone_conv4",
        node_type="conv",
        device_type="npu",
        npu_id=0,
        weight_tiles=[],
        input_nodes=[c2f_2],
        input_shape=(128, 80, 80),
        output_shape=(256, 40, 40),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(backbone_conv4)
    
    c2f_3 = create_c2f_module(graph, "backbone_c2f_3", "backbone_conv4",
                              in_ch=256, out_ch=256, n_bottleneck=2, device_type="eflash", use_fused=True)
    
    # Stage 4: Conv + C2f (P4)
    backbone_conv5 = ComputeNode(
        node_id="backbone_conv5",
        node_type="conv",
        device_type="eflash",
        weight_tiles=[],
        input_nodes=[c2f_3],
        input_shape=(256, 40, 40),
        output_shape=(512, 20, 20),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(backbone_conv5)
    
    c2f_4 = create_c2f_module(graph, "backbone_c2f_4", "backbone_conv5",
                              in_ch=512, out_ch=512, n_bottleneck=1, device_type="eflash")
    
    # SPPF (P5)
    sppf = create_sppf_module(graph, "backbone_sppf", c2f_4,
                             in_ch=512, out_ch=512, pool_size=5)
    
    # ========================================
    # Neck (PAN-FPN)
    # ========================================
    
    # Fused: Upsample 1 + Concat 1 + C2f 5 (메모리 최적화)
    c2f_5 = create_fused_up_concat_c2f_module(
        graph, 
        node_id_prefix="neck_fused_up1_c2f5",
        up_input_id=sppf,           # upsample 입력: SPPF (512ch, 20x20)
        residual_input_id=c2f_3,    # residual 입력: C2f_3 (256ch, 40x40)
        up_in_ch=512,
        residual_in_ch=256,
        out_ch=256,                 # 최종 출력
        H=40, W=40,                 # upsample 후 크기
        n_splits=4
    )
    
    # # 기존 방식 (메모리 비효율적)
    # # Upsample 1
    # neck_up1 = ComputeNode(
    #     node_id="neck_up1",
    #     node_type="upsample",
    #     device_type="eflash",
    #     weight_tiles=[],
    #     input_nodes=[sppf],
    #     input_shape=(512, 20, 20),
    #     output_shape=(512, 40, 40),
    #     metadata={'scale_factor': 2}
    # )
    # graph.add_node(neck_up1)
    # 
    # # Concat 1 (P5 + P4)
    # neck_concat1 = ComputeNode(
    #     node_id="neck_concat1",
    #     node_type="concat",
    #     device_type="eflash",
    #     weight_tiles=[],
    #     input_nodes=["neck_up1", c2f_3],
    #     input_shape=(768, 40, 40),
    #     output_shape=(768, 40, 40),
    #     metadata={'axis': 0}
    # )
    # graph.add_node(neck_concat1)
    # 
    # # C2f 5
    # c2f_5 = create_c2f_module(graph, "neck_c2f_5", "neck_concat1",
    #                           in_ch=768, out_ch=256, n_bottleneck=1, device_type="eflash", use_fused=True)
    
    # Fused: Upsample 2 + Concat 2 + C2f 6 (메모리 최적화)
    # c2f_6 = create_fused_up_concat_c2f_module(
    #     graph,
    #     node_id_prefix="neck_fused_up2_c2f6",
    #     up_input_id=c2f_5,          # upsample 입력: C2f_5 (256ch, 40x40)
    #     residual_input_id=c2f_2,    # residual 입력: C2f_2 (128ch, 80x80)
    #     up_in_ch=256,
    #     residual_in_ch=128,
    #     out_ch=128,                 # 최종 출력
    #     H=80, W=80,                 # upsample 후 크기
    #     n_splits=4
    # )
    
    # # 기존 방식 (메모리 비효율적)
    # Upsample 2
    neck_up2 = ComputeNode(
        node_id="neck_up2",
        node_type="upsample",
        # device_type="eflash",
        device_type="npu",
        npu_id=0,
        weight_tiles=[],
        input_nodes=[c2f_5],
        input_shape=(256, 40, 40),
        output_shape=(256, 80, 80),
        metadata={'scale_factor': 2}
    )
    graph.add_node(neck_up2)
    
    # Concat 2 (P4 + P3)
    neck_concat2 = ComputeNode(
        node_id="neck_concat2",
        node_type="concat",
        # device_type="eflash",
        device_type="npu",
        npu_id=0,
        weight_tiles=[],
        input_nodes=["neck_up2", c2f_2],
        input_shape=(384, 80, 80),
        output_shape=(384, 80, 80),
        metadata={'axis': 0}
    )
    graph.add_node(neck_concat2)
    
    # C2f 6 (P3 output)
    # c2f_6 = create_c2f_module(graph, "neck_c2f_6", "neck_concat2",
    #                           in_ch=384, out_ch=128, n_bottleneck=1, device_type="eflash", use_fused=True)
    c2f_6 = create_c2f_module(graph, "neck_c2f_6", "neck_concat2",
                              in_ch=384, out_ch=128, n_bottleneck=1, device_type="npu")
    
    # Conv down 1
    neck_conv1 = ComputeNode(
        node_id="neck_conv1",
        node_type="conv",
        # device_type="eflash",
        device_type="npu",
        npu_id=0,
        weight_tiles=[],
        input_nodes=[c2f_6],
        input_shape=(128, 80, 80),
        output_shape=(128, 40, 40),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(neck_conv1)
    
    # Fused: Concat 3 + C2f 7 (메모리 최적화)
    c2f_7 = create_fused_concat_c2f_module(
        graph,
        node_id_prefix="neck_fused_concat_c2f7",
        input1_id="neck_conv1",     # main input: neck_conv1 (128ch, 40x40)
        input2_id=c2f_5,             # residual input: c2f_5 (256ch, 40x40)
        input1_ch=128,
        input2_ch=256,
        out_ch=256,                  # 최종 출력
        H=40, W=40
    )
    
    # # 기존 방식 (메모리 비효율적)
    # # Concat 3
    # neck_concat3 = ComputeNode(
    #     node_id="neck_concat3",
    #     node_type="concat",
    #     device_type="eflash",
    #     weight_tiles=[],
    #     input_nodes=["neck_conv1", c2f_5],
    #     input_shape=(384, 40, 40),
    #     output_shape=(384, 40, 40),
    #     metadata={'axis': 0}
    # )
    # graph.add_node(neck_concat3)
    # 
    # # C2f 7 (P4 output)
    # c2f_7 = create_c2f_module(graph, "neck_c2f_7", "neck_concat3",
    #                           in_ch=384, out_ch=256, n_bottleneck=1, device_type="eflash", use_fused=True)
    
    # Conv down 2
    neck_conv2 = ComputeNode(
        node_id="neck_conv2",
        node_type="conv",
        device_type="eflash",
        weight_tiles=[],
        input_nodes=[c2f_7],
        input_shape=(256, 40, 40),
        output_shape=(256, 20, 20),
        metadata={'kernel_size': 3, 'stride': 2, 'padding': 1}
    )
    graph.add_node(neck_conv2)
    
    # Concat 4
    neck_concat4 = ComputeNode(
        node_id="neck_concat4",
        node_type="concat",
        device_type="eflash",
        weight_tiles=[],
        input_nodes=["neck_conv2", sppf],
        input_shape=(768, 20, 20),
        output_shape=(768, 20, 20),
        metadata={'axis': 0}
    )
    graph.add_node(neck_concat4)
    
    # C2f 8 (P5 output)
    c2f_8 = create_c2f_module(graph, "neck_c2f_8", "neck_concat4",
                              in_ch=768, out_ch=512, n_bottleneck=1, device_type="eflash")
    
    # ========================================
    # Detection Head (Decoupled)
    # ========================================
    
    # P3 Head (80x80)
    head_p3_conv1 = ComputeNode(
        node_id="head_p3_conv1",
        node_type="conv",
        device_type="npu",
        npu_id=0,
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
        device_type="npu",
        npu_id=0,
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
        device_type="npu",
        npu_id=0,
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
        weight_tiles=[],
        input_nodes=["head_p5_conv2"],
        input_shape=(256, 20, 20),
        output_shape=(80, 20, 20),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(head_p5)
    
    return graph
