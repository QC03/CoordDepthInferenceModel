from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class JointMapping:
    """Explicit rtmlib/H36M correspondence. Unlisted indices are invalid."""

    rtmlib_to_h36m: Dict[int, int]
    rtmlib_names: Tuple[str, ...]
    h36m_names: Tuple[str, ...]
    left_hip: int
    right_hip: int
    excluded: Tuple[int, ...] = ()

    @property
    def num_joints(self) -> int:
        return len(self.rtmlib_names)

    @classmethod
    def h36m25_default(cls) -> "JointMapping":
        names = tuple(f"rtmlib_{i}" for i in range(26))
        h36m = tuple(f"h36m_{i}" for i in range(25))
        mapping = {
            0: 9, 5: 11, 6: 14, 7: 12, 8: 15, 9: 13, 10: 16,
            11: 4, 12: 1, 13: 5, 14: 2, 15: 6, 16: 3,
            20: 17, 23: 18,
        }
        return cls(mapping, names, h36m, left_hip=4, right_hip=1,
               excluded=tuple(i for i in range(26) if i not in mapping))


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
