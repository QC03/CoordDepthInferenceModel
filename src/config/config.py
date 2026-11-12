from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class JointMapping:
    """Explicit MediaPipe/H36M correspondence. Indices not listed are invalid."""

    mediapipe_to_h36m: Dict[int, int]
    mediapipe_names: Tuple[str, ...]
    h36m_names: Tuple[str, ...]
    left_hip: int
    right_hip: int
    excluded: Tuple[int, ...] = ()

    @property
    def num_joints(self) -> int:
        return len(self.mediapipe_names)

    @classmethod
    def h36m25_default(cls) -> "JointMapping":
        # MediaPipe indices follow the 33-landmark BlazePose convention.
        names = tuple(f"mp_{i}" for i in range(33))
        h36m = tuple(f"h36m_{i}" for i in range(25))
        mapping = {
            0: 9, 11: 11, 12: 14, 13: 12, 14: 15, 15: 13, 16: 16,
            23: 4, 24: 1, 25: 5, 26: 2, 27: 6, 28: 3,
            31: 17, 32: 18,
        }
        return cls(mapping, names, h36m, left_hip=4, right_hip=1,
                   excluded=tuple(i for i in range(33) if i not in mapping))


@dataclass(frozen=True)
class PoseConfig:
    image_size: Tuple[int, int] = (224, 224)
    num_joints: int = 15
    feature_channels: int = 512
    embedding_dim: int = 32
    hidden_dim: int = 256
    backbone: str = "resnet18_layer4"
    align_corners: bool = False
    depth_axis: int = 2
    depth_unit: str = "meter"
    mapping: JointMapping = field(default_factory=JointMapping.h36m25_default)
    left_hip_index: int = 7
    right_hip_index: int = 8
