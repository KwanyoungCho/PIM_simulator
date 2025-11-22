"""
Graph preprocessing utilities.

Conv 노드 중 weight tiling이 적용된 노드를 실행 전에 tile 단위 sub-graph로 확장한다.
- Output tiling → tile Conv 노드 + 최종 concat 노드
- Reduction tiling → tile Conv 노드 + 단일 reduce 노드 (tile 결과를 빠르게 합산)

이 과정을 offline으로 수행하여 scheduler를 단순화한다.
"""

from collections import defaultdict
from typing import Dict, List

from .compute_node import ComputeGraph, ComputeNode


class GraphPreprocessor:
    """Graph preprocessing helper."""

    @staticmethod
    def expand_tiled_nodes(graph: ComputeGraph) -> ComputeGraph:
        """
        Conv 노드의 weight tiles를 기반으로 sub-graph 확장.

        Returns:
            변환된 그래프 (원본 객체를 수정하여 반환)
        """
        print("\n[Graph Preprocessing] Expanding tiled nodes to sub-graph...")

        tiled_nodes = [
            node for node in graph.get_all_nodes()
            if node.node_type == "conv" and node.device_type == "eflash" and node.weight_tiles and len(node.weight_tiles) > 1
        ]
        print(f"  Found {len(tiled_nodes)} tiled nodes (eFlash only)")

        for parent in tiled_nodes:
            tile_groups = defaultdict(list)  # out_idx -> List[(red_idx, tile_idx, WeightTile)]
            for tile_idx, tile in enumerate(parent.weight_tiles):
                metadata = tile.metadata or {}
                out_idx = metadata.get("output_tile_idx", 0)
                red_idx = metadata.get("reduction_tile_idx", 0)
                tile_groups[out_idx].append((red_idx, tile_idx, tile))

            if not tile_groups:
                continue

            last_array_id = parent.array_id
            last_area_id = parent.area_id
            group_outputs: Dict[int, str] = {}

            for out_idx in sorted(tile_groups.keys()):
                tiles = sorted(tile_groups[out_idx], key=lambda x: x[0])
                tile_node_ids: List[str] = []

                for red_idx, tile_idx, tile in tiles:
                    tile_metadata = tile.metadata or {}
                    tile_array_id = tile_metadata.get("array_id", parent.array_id)
                    tile_area_id = tile_metadata.get("area_id", parent.area_id)
                    last_array_id = tile_array_id
                    last_area_id = tile_area_id

                    tile_id = f"{parent.node_id}_tile{tile_idx}"
                    tile_shape = tile.get_shape()
                    tile_channels = tile_shape[0]

                    if parent.output_shape and len(parent.output_shape) >= 1:
                        spatial = parent.output_shape[1:]
                        tile_output_shape = (tile_channels, *spatial)
                    else:
                        tile_output_shape = parent.output_shape

                    tile_node = ComputeNode(
                        node_id=tile_id,
                        node_type=parent.node_type,
                        device_type=parent.device_type,
                        array_id=tile_array_id,
                        area_id=tile_area_id,
                        weight_tiles=[tile],
                        input_nodes=list(parent.input_nodes),
                        input_shape=parent.input_shape,
                        output_shape=tile_output_shape,
                        metadata={
                            **parent.metadata,
                            "parent_node": parent.node_id,
                            "tile_idx": tile_idx,
                            "output_tile_idx": out_idx,
                            "reduction_tile_idx": red_idx,
                            "is_tile_node": True,
                            "generated_by_preprocessor": True,
                            "preprocessed_type": "tile",
                        },
                    )
                    graph.add_node(tile_node)
                    tile_node_ids.append(tile_id)

                if len(tile_node_ids) == 1:
                    group_outputs[out_idx] = tile_node_ids[0]
                else:
                    reduce_id = f"{parent.node_id}_reduce_o{out_idx}"
                    reduce_node = ComputeNode(
                        node_id=reduce_id,
                        node_type="reduce",
                        device_type=parent.device_type,
                        array_id=last_array_id,
                        area_id=last_area_id,
                        weight_tiles=[],
                        input_nodes=tile_node_ids,
                        input_shape=tile_output_shape,
                        output_shape=tile_output_shape,
                        metadata={
                            "parent_node": parent.node_id,
                            "is_reduction_merge": True,
                            "output_tile_idx": out_idx,
                            "generated_by_preprocessor": True,
                            "preprocessed_type": "reduce",
                        },
                    )
                    graph.add_node(reduce_node)
                    group_outputs[out_idx] = reduce_id

            # Output dimension concat (필요 시)
            sorted_outputs = [group_outputs[idx] for idx in sorted(group_outputs.keys())]
            if len(sorted_outputs) > 1:
                concat_id = f"{parent.node_id}_concat"
                concat_node = ComputeNode(
                    node_id=concat_id,
                    node_type="concat",
                    device_type=parent.device_type,
                    array_id=last_array_id,
                    area_id=last_area_id,
                    weight_tiles=[],
                    input_nodes=sorted_outputs,
                    input_shape=parent.output_shape,
                    output_shape=parent.output_shape,
                    metadata={
                        "parent_node": parent.node_id,
                        "is_output_concat": True,
                        "generated_by_preprocessor": True,
                        "preprocessed_type": "concat",
                    },
                )
                graph.add_node(concat_node)
                final_output_id = concat_id
            else:
                final_output_id = sorted_outputs[0]

            # 부모 노드를 참조하던 consumer를 새로운 output으로 연결
            for consumer in graph.get_all_nodes():
                consumer.input_nodes = [
                    final_output_id if inp == parent.node_id else inp
                    for inp in consumer.input_nodes
                ]

            # 원본 노드 제거
            if parent.node_id in graph.nodes:
                del graph.nodes[parent.node_id]

        print(f"[Graph Preprocessing] Complete! Graph now has {len(graph.get_all_nodes())} nodes")
        return graph

