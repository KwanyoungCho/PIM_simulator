"""
PIM Simulator Package

서브패키지:
- hardware: HW 리소스 및 PIM 시뮬레이터
- graph: ComputeGraph 및 전처리/검증
- scheduler: 이벤트 스케줄러, 컨텍스트
"""

from .hardware import (
    WeightTile,
    Area,
    SRAMBuffer,
    eFlashArray,
    NPU,
    PIMSimulator,
)
from .scheduler import (
    Event,
    EventType,
    ActivationBuffer,
    ActivationManager,
    InferenceScheduler,
    InferenceContext,
)
from .graph import (
    ComputeNode,
    ComputeGraph,
    GraphPreprocessor,
    GraphValidator,
)

__all__ = [
    'WeightTile',
    'Area',
    'SRAMBuffer',
    'eFlashArray',
    'NPU',
    'PIMSimulator',
    'Event',
    'EventType',
    'ActivationBuffer',
    'ActivationManager',
    'ComputeNode',
    'ComputeGraph',
    'GraphPreprocessor',
    'GraphValidator',
    'InferenceScheduler',
    'InferenceContext'
]

__version__ = '0.2.0'
