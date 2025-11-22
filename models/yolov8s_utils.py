"""
YOLOv8s Utility Modules

C2f, SPPF 등 YOLOv8s의 핵심 모듈들을 정의
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import ComputeNode


def create_c2f_module(graph, node_id_prefix, input_node, in_ch, out_ch, n_bottleneck=1,
                      has_shortcut=False, device_type="eflash", use_fused=False):
    """
    C2f 모듈을 Conv + Bottleneck + Concat으로 분해
    
    Args:
        use_fused: eFlash에서 concat 제거하고 partial conv + reduce로 최적화 (메모리 절약)
    
    C2f 구조 (Faster C3):
    Input
      ├─ Conv 1x1 (채널 절반) ──┐
      │                        │
      └─ Conv 1x1 (채널 절반)   │
         → Bottleneck × n      │
         (각 bottleneck 출력도 concat)
                               ↓
                          Concat (all)  [use_fused=False]
                               ↓        or
                          Conv 1x1      Partial Conv + Reduce [use_fused=True, eFlash only]
    """
    mid_ch = out_ch // 2
    H, W = graph.get_node(input_node).output_shape[1:3]
    
    # Path 1: Conv 1x1 (bypass)
    path1_id = f"{node_id_prefix}_path1"
    if device_type == "npu":
        path1 = ComputeNode(
            node_id=path1_id,
            node_type="conv",
            device_type=device_type,
            npu_id=0,
            weight_tiles=[],
            input_nodes=[input_node],
            input_shape=(in_ch, H, W),
            output_shape=(mid_ch, H, W),
            metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
        )
    else:
        path1 = ComputeNode(
            node_id=path1_id,
            node_type="conv",
            device_type=device_type,
            weight_tiles=[],
            input_nodes=[input_node],
            input_shape=(in_ch, H, W),
            output_shape=(mid_ch, H, W),
            metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
        )
    graph.add_node(path1)
    
    # Path 2: Conv 1x1 → Bottlenecks
    path2_conv1_id = f"{node_id_prefix}_path2_conv1"
    if device_type == "npu":
        path2_conv1 = ComputeNode(
            node_id=path2_conv1_id,
            node_type="conv",
            device_type=device_type,
            npu_id=0,
            weight_tiles=[],
            input_nodes=[input_node],
            input_shape=(in_ch, H, W),
            output_shape=(mid_ch, H, W),
            metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
        )
    else:
        path2_conv1 = ComputeNode(
            node_id=path2_conv1_id,
            node_type="conv",
            device_type=device_type,
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
        if device_type == "npu":
            bn_conv1 = ComputeNode(
                node_id=bn_conv1_id,
                node_type="conv",
                device_type=device_type,
                npu_id=0,
                weight_tiles=[],
                input_nodes=[bottleneck_input],
                input_shape=(mid_ch, H, W),
                output_shape=(mid_ch, H, W),
                metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
            )
        else:
            bn_conv1 = ComputeNode(
                node_id=bn_conv1_id,
                node_type="conv",
                device_type=device_type,
                weight_tiles=[],
                input_nodes=[bottleneck_input],
                input_shape=(mid_ch, H, W),
                output_shape=(mid_ch, H, W),
                metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
            )
        graph.add_node(bn_conv1)
        
        bn_conv2_id = f"{node_id_prefix}_bn{i}_conv2"
        if device_type == "npu":
            bn_conv2 = ComputeNode(
                node_id=bn_conv2_id,
                node_type="conv",
                device_type=device_type,
                npu_id=0,
                weight_tiles=[],
                input_nodes=[bn_conv1_id],
                input_shape=(mid_ch, H, W),
                output_shape=(mid_ch, H, W),
                metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
            )
        else:
            bn_conv2 = ComputeNode(
                node_id=bn_conv2_id,
                node_type="conv",
                device_type=device_type,
                weight_tiles=[],
                input_nodes=[bn_conv1_id],
                input_shape=(mid_ch, H, W),
                output_shape=(mid_ch, H, W),
                metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
            )
        graph.add_node(bn_conv2)
        
        concat_inputs.append(bn_conv2_id)
        bottleneck_input = bn_conv2_id
    
    # eFlash + use_fused: partial conv + reduce (메모리 절약, 순차 실행)
    # 그 외: 기존 concat + final conv
    if device_type == "eflash" and use_fused:
        # 순차 실행 강제: partial → reduce → partial → reduce 패턴
        prev_reduce = None
        
        for i, source_id in enumerate(concat_inputs):
            # partial conv의 input: source + 이전 reduce (순차 실행 강제)
            partial_inputs = [source_id]
            if prev_reduce is not None:
                partial_inputs.append(prev_reduce)  # dependency for sequential execution
            
            partial_id = f"{node_id_prefix}_partial{i}"
            partial_conv = ComputeNode(
                node_id=partial_id,
                node_type="conv",
                device_type=device_type,
                weight_tiles=[],
                input_nodes=partial_inputs,
                input_shape=(mid_ch, H, W),
                output_shape=(out_ch, H, W),
                metadata={
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0,
                    'reduction_offset': i * mid_ch
                }
            )
            graph.add_node(partial_conv)
            
            # 첫 번째는 그대로 유지, 이후부터 reduce로 누적
            if i == 0:
                prev_reduce = partial_id  # 첫 번째는 reduce 없이 다음 단계로
            else:
                reduce_id = f"{node_id_prefix}_reduce{i-1}"
                reduce_node = ComputeNode(
                    node_id=reduce_id,
                    node_type="reduce",
                    device_type=device_type,
                    weight_tiles=[],
                    input_nodes=[prev_reduce, partial_id],
                    input_shape=(out_ch, H, W),
                    output_shape=(out_ch, H, W),
                    metadata={'operation': 'add'}
                )
                graph.add_node(reduce_node)
                prev_reduce = reduce_id
        
        return prev_reduce
    else:
        # 기존 concat + final conv
        concat_id = f"{node_id_prefix}_concat"
        concat_ch = mid_ch * (2 + n_bottleneck)
        if device_type == "npu":
            concat_node = ComputeNode(
                node_id=concat_id,
                node_type="concat",
                device_type=device_type,
                npu_id=0,
                weight_tiles=[],
                input_nodes=concat_inputs,
                input_shape=(concat_ch, H, W),
                output_shape=(concat_ch, H, W),
                metadata={'axis': 0}
            )
        else:
            concat_node = ComputeNode(
                node_id=concat_id,
                node_type="concat",
                device_type=device_type,
                weight_tiles=[],
                input_nodes=concat_inputs,
                input_shape=(concat_ch, H, W),
                output_shape=(concat_ch, H, W),
                metadata={'axis': 0}
            )
        graph.add_node(concat_node)
        
        final_id = f"{node_id_prefix}_final"
        if device_type == "npu":
            final_conv = ComputeNode(
                node_id=final_id,
                node_type="conv",
                device_type=device_type,
                npu_id=0,
                weight_tiles=[],
                input_nodes=[concat_id],
                input_shape=(concat_ch, H, W),
                output_shape=(out_ch, H, W),
                metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
            )
        else:
            final_conv = ComputeNode(
                node_id=final_id,
                node_type="conv",
                device_type=device_type,
                weight_tiles=[],
                input_nodes=[concat_id],
                input_shape=(concat_ch, H, W),
                output_shape=(out_ch, H, W),
                metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
            )
        graph.add_node(final_conv)
        return final_id


def create_fused_concat_c2f_module(graph, node_id_prefix, input1_id, input2_id,
                                   input1_ch, input2_ch, out_ch, 
                                   H, W, n_bottleneck=1, device_type="eflash"):
    """
    Concat + C2f를 메모리 최적화된 형태로 구현 (C2f 구조 완전 유지)
    
    핵심 아이디어:
    - Convolution weight splitting을 활용
    - 원래: W * concat(x1, x2) = W1*x1 + W2*x2 (수학적으로 동일)
    - Split 없이 두 입력을 각각 한 번씩 conv하고 add로 reduce
    
    구조:
    1. input1 → path1 conv → mid_ch (W1 사용)
    2. input2 → path2 conv → mid_ch (W2 사용)
    3. Reduce (ADD) → merged (mid_ch)
    4. Bottleneck들 (merged 기반)
    5. 모든 outputs → partial convs + reduce
    
    Args:
        n_bottleneck: bottleneck 개수 (기본 1)
    """
    mid_ch = out_ch // 2
    
    # === Phase 1: Path1/Path2 Conv + Reduce (concat 대체) ===
    # Weight splitting: W = [W1 | W2]
    # W * concat(x1, x2) = W1*x1 + W2*x2
    
    # 1. Input1 → path1 conv (W1 part)
    path1_id = f"{node_id_prefix}_path1_conv"
    path1 = ComputeNode(
        node_id=path1_id,
        node_type="conv",
        device_type=device_type,
        weight_tiles=[],
        input_nodes=[input1_id],
        input_shape=(input1_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(path1)
    
    # 2. Input2 → path2 conv (W2 part)
    path2_id = f"{node_id_prefix}_path2_conv"
    path2 = ComputeNode(
        node_id=path2_id,
        node_type="conv",
        device_type=device_type,
        weight_tiles=[],
        input_nodes=[input2_id],
        input_shape=(input2_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(path2)
    
    # 3. Reduce (ADD: W1*x1 + W2*x2)
    reduce_id = f"{node_id_prefix}_path_reduce"
    reduce_node = ComputeNode(
        node_id=reduce_id,
        node_type="reduce",
        device_type=device_type,
        weight_tiles=[],
        input_nodes=[path1_id, path2_id],
        input_shape=(mid_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'operation': 'add'}
    )
    graph.add_node(reduce_node)
    
    merged_path_id = reduce_id
    
    # === Phase 2: Bottlenecks (기존 C2f 구조) ===
    concat_inputs = [merged_path_id]
    bottleneck_input = merged_path_id
    
    for i in range(n_bottleneck):
        bn_conv1_id = f"{node_id_prefix}_bn{i}_conv1"
        bn_conv1 = ComputeNode(
            node_id=bn_conv1_id,
            node_type="conv",
            device_type=device_type,
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
            device_type=device_type,
            weight_tiles=[],
            input_nodes=[bn_conv1_id],
            input_shape=(mid_ch, H, W),
            output_shape=(mid_ch, H, W),
            metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
        )
        graph.add_node(bn_conv2)
        
        concat_inputs.append(bn_conv2_id)
        bottleneck_input = bn_conv2_id
    
    # === Phase 3: Final (partial + reduce 또는 concat + final_conv) ===
    # use_fused=True 방식: partial conv + reduce
    prev_reduce = None
    
    for i, source_id in enumerate(concat_inputs):
        partial_inputs = [source_id]
        if prev_reduce is not None:
            partial_inputs.append(prev_reduce)
        
        partial_id = f"{node_id_prefix}_final_partial{i}"
        partial_conv = ComputeNode(
            node_id=partial_id,
            node_type="conv",
            device_type=device_type,
            weight_tiles=[],
            input_nodes=partial_inputs,
            input_shape=(mid_ch, H, W),
            output_shape=(out_ch, H, W),
            metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
        )
        graph.add_node(partial_conv)
        
        if i == 0:
            prev_reduce = partial_id
        else:
            reduce_id = f"{node_id_prefix}_final_reduce{i-1}"
            reduce_node = ComputeNode(
                node_id=reduce_id,
                node_type="reduce",
                device_type=device_type,
                weight_tiles=[],
                input_nodes=[prev_reduce, partial_id],
                input_shape=(out_ch, H, W),
                output_shape=(out_ch, H, W),
                metadata={'operation': 'add'}
            )
            graph.add_node(reduce_node)
            prev_reduce = reduce_id
    
    return prev_reduce


def create_fused_up_concat_c2f_module(graph, node_id_prefix, up_input_id, residual_input_id,
                                      up_in_ch, residual_in_ch, out_ch, 
                                      H, W, n_bottleneck=1, n_splits=4, device_type="eflash"):
    """
    Upsample + Concat + C2f를 메모리 최적화된 형태로 구현 (C2f 구조 완전 유지)
    
    수학적 원리 (Weight Splitting):
    원래: W * concat(residual, upsample(up_input))
    = [W_res | W_up] * [residual; upsample(up_input)]
    = W_res * residual + W_up * upsample(up_input)
    
    W_up를 channel 방향으로 분할:
    W_up = [W_up0 | W_up1 | ... | W_upN]
    up_input = [split0 | split1 | ... | splitN]
    
    W_up * upsample(up_input) = Σ W_upi * upsample(spliti)
    
    최종: W_res*residual + Σ(W_upi*upsample(spliti))  [모두 ADD]
    
    구조:
    1. residual → conv → mid_ch (W_res 사용)
    2. up_input → split → upsample → conv → mid_ch (각각 W_upi 사용)
    3. 모두 ADD로 reduce → merged (mid_ch)
    4. Bottleneck + final
    
    Args:
        n_splits: upsample path를 몇 개로 분할할지 (기본 4)
        n_bottleneck: bottleneck 개수 (기본 1)
    """
    H_in = H // 2  # upsample 전 크기
    W_in = W // 2
    mid_ch = out_ch // 2
    
    # === Phase 1: Weight Splitting with ADD ===
    # 각 path는 mid_ch를 전부 출력하고 ADD로 합침
    
    # 1. Residual path: conv (W_residual part)
    residual_conv_id = f"{node_id_prefix}_residual_conv"
    residual_conv = ComputeNode(
        node_id=residual_conv_id,
        node_type="conv",
        device_type=device_type,
        weight_tiles=[],
        input_nodes=[residual_input_id],
        input_shape=(residual_in_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(residual_conv)
    
    # 2. Upsample path를 n_splits로 분할 (메모리 절약)
    split_ch = up_in_ch // n_splits
    up_conv_ids = []
    
    for i in range(n_splits):
        # 2-1. Split (채널 분할)
        split_id = f"{node_id_prefix}_up_split{i}"
        split_node = ComputeNode(
            node_id=split_id,
            node_type="split",
            device_type=device_type,
            weight_tiles=[],
            input_nodes=[up_input_id],
            input_shape=(up_in_ch, H_in, W_in),
            output_shape=(split_ch, H_in, W_in),
            metadata={'split_dim': 0, 'split_idx': i, 'split_total': n_splits}
        )
        graph.add_node(split_node)
        
        # 2-2. Upsample (2x)
        up_id = f"{node_id_prefix}_up{i}"
        up_node = ComputeNode(
            node_id=up_id,
            node_type="upsample",
            device_type=device_type,
            weight_tiles=[],
            input_nodes=[split_id],
            input_shape=(split_ch, H_in, W_in),
            output_shape=(split_ch, H, W),
            metadata={'scale_factor': 2}
        )
        graph.add_node(up_node)
        
        # 2-3. Conv (W_upi part - 각각 mid_ch 출력)
        conv_id = f"{node_id_prefix}_up_conv{i}"
        conv_node = ComputeNode(
            node_id=conv_id,
            node_type="conv",
            device_type=device_type,
            weight_tiles=[],
            input_nodes=[up_id],
            input_shape=(split_ch, H, W),
            output_shape=(mid_ch, H, W),
            metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
        )
        graph.add_node(conv_node)
        up_conv_ids.append(conv_id)
    
    # 3. Reduce (ADD): residual + up0 + up1 + ... → merged
    # 순차적으로 ADD
    prev_reduce = residual_conv_id
    for i, up_conv_id in enumerate(up_conv_ids):
        reduce_id = f"{node_id_prefix}_path_reduce{i}"
        reduce_node = ComputeNode(
            node_id=reduce_id,
            node_type="reduce",
            device_type=device_type,
            weight_tiles=[],
            input_nodes=[prev_reduce, up_conv_id],
            input_shape=(mid_ch, H, W),
            output_shape=(mid_ch, H, W),
            metadata={'operation': 'add'}
        )
        graph.add_node(reduce_node)
        prev_reduce = reduce_id
    
    merged_path_id = prev_reduce
    
    # === Phase 2: Bottlenecks (기존 C2f 구조) ===
    concat_inputs = [merged_path_id]
    bottleneck_input = merged_path_id
    
    for i in range(n_bottleneck):
        bn_conv1_id = f"{node_id_prefix}_bn{i}_conv1"
        bn_conv1 = ComputeNode(
            node_id=bn_conv1_id,
            node_type="conv",
            device_type=device_type,
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
            device_type=device_type,
            weight_tiles=[],
            input_nodes=[bn_conv1_id],
            input_shape=(mid_ch, H, W),
            output_shape=(mid_ch, H, W),
            metadata={'kernel_size': 3, 'stride': 1, 'padding': 1}
        )
        graph.add_node(bn_conv2)
        
        concat_inputs.append(bn_conv2_id)
        bottleneck_input = bn_conv2_id
    
    # === Phase 3: Final (partial + reduce) ===
    prev_reduce = None
    
    for i, source_id in enumerate(concat_inputs):
        partial_inputs = [source_id]
        if prev_reduce is not None:
            partial_inputs.append(prev_reduce)
        
        partial_id = f"{node_id_prefix}_final_partial{i}"
        partial_conv = ComputeNode(
            node_id=partial_id,
            node_type="conv",
            device_type=device_type,
            weight_tiles=[],
            input_nodes=partial_inputs,
            input_shape=(mid_ch, H, W),
            output_shape=(out_ch, H, W),
            metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
        )
        graph.add_node(partial_conv)
        
        if i == 0:
            prev_reduce = partial_id
        else:
            reduce_id = f"{node_id_prefix}_final_reduce{i-1}"
            reduce_node = ComputeNode(
                node_id=reduce_id,
                node_type="reduce",
                device_type=device_type,
                weight_tiles=[],
                input_nodes=[prev_reduce, partial_id],
                input_shape=(out_ch, H, W),
                output_shape=(out_ch, H, W),
                metadata={'operation': 'add'}
            )
            graph.add_node(reduce_node)
            prev_reduce = reduce_id
    
    return prev_reduce


def create_sppf_module(graph, node_id_prefix, input_node, in_ch, out_ch, pool_size=5, device_type="eflash"):
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
        weight_tiles=[],
        input_nodes=[pool2_id],
        input_shape=(mid_ch, H, W),
        output_shape=(mid_ch, H, W),
        metadata={'kernel_size': pool_size, 'stride': 1, 'padding': pool_size//2}
    )
    graph.add_node(pool3)
    
    # Concat
    concat_id = f"{node_id_prefix}_concat"
    concat_ch = mid_ch * 4
    concat_node = ComputeNode(
        node_id=concat_id,
        node_type="concat",
        device_type="eflash",
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
        weight_tiles=[],
        input_nodes=[concat_id],
        input_shape=(concat_ch, H, W),
        output_shape=(out_ch, H, W),
        metadata={'kernel_size': 1, 'stride': 1, 'padding': 0}
    )
    graph.add_node(conv2)
    
    return conv2_id
