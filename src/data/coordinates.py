from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass(frozen=True)
class CropTransform:
    """Mapping from source pixels to a letterboxed square crop."""

    x0: float
    y0: float
    width: float
    height: float
    output_width: int = 224
    output_height: int = 224
    source_width: float = 1.0
    source_height: float = 1.0

    def source_normalized_to_crop_normalized(self, xy: torch.Tensor) -> torch.Tensor:
        """Convert source-image normalized xy (x,y) to crop normalized xy."""
        px = xy[..., 0] * self.source_width
        py = xy[..., 1] * self.source_height
        return torch.stack(((px - self.x0) / self.width,
                            (py - self.y0) / self.height), dim=-1)

    def crop_normalized_to_grid(self, xy: torch.Tensor,
                                align_corners: bool = False) -> torch.Tensor:
        """Convert [0,1] crop xy to grid_sample [-1,1] in x,y order."""
        if align_corners:
            return xy * 2.0 - 1.0
        return xy * 2.0 - 1.0

    def crop_normalized_to_pixels(self, xy: torch.Tensor) -> torch.Tensor:
        return xy * torch.tensor((self.output_width, self.output_height),
                                  dtype=xy.dtype, device=xy.device)


def root_relative_xy(keypoints_2d: torch.Tensor, left_hip: int,
                     right_hip: int) -> torch.Tensor:
    root = (keypoints_2d[:, left_hip] + keypoints_2d[:, right_hip]) / 2
    return keypoints_2d - root[:, None]


def root_relative_depth(depth: torch.Tensor, left_hip: int,
                        right_hip: int) -> torch.Tensor:
    root = (depth[:, left_hip] + depth[:, right_hip]) / 2
    return depth - root[:, None]


def h36m_root_relative(points: torch.Tensor, left_hip: int,
                       right_hip: int) -> torch.Tensor:
    root = (points[:, left_hip] + points[:, right_hip]) / 2
    return points - root[:, None, :]


def crop_box_from_bbox(xyxy: Tuple[float, float, float, float],
                       output_size: Tuple[int, int] = (224, 224),
                       padding: float = 0.0) -> CropTransform:
    x0, y0, x1, y1 = xyxy
    w, h = x1 - x0, y1 - y0
    return CropTransform(x0 - padding * w, y0 - padding * h,
                         w * (1 + 2 * padding), h * (1 + 2 * padding),
                         output_size[0], output_size[1], 1.0, 1.0)
