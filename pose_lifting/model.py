from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from .config import PoseConfig
from .coordinates import root_relative_depth, root_relative_xy


class ResNet18Layer4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise ImportError("torchvision is required for ResNet18Layer4") from exc
        backbone = resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width),
                                  nn.GELU(), nn.Linear(width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DepthLiftingModel(nn.Module):
    """Image + absolute crop-normalized 2D points -> root-relative depth."""

    def __init__(self, config: Optional[PoseConfig] = None,
                 backbone: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.config = config or PoseConfig()
        self.backbone = backbone or ResNet18Layer4()
        self.xy_embedding = nn.Sequential(nn.Linear(2, 32), nn.GELU(),
                                          nn.Linear(32, self.config.embedding_dim))
        self.visual_embedding = nn.Sequential(
            nn.Linear(self.config.feature_channels, 64), nn.GELU(),
            nn.Linear(64, self.config.embedding_dim))
        width = self.config.embedding_dim * 2
        self.head = nn.Sequential(nn.Linear(self.config.num_joints * width,
                                            self.config.hidden_dim), nn.GELU(),
                                  ResidualBlock(self.config.hidden_dim),
                                  ResidualBlock(self.config.hidden_dim),
                                  nn.Linear(self.config.hidden_dim,
                                            self.config.num_joints))

    def sample_features(self, feature_map: torch.Tensor,
                        keypoints_2d: torch.Tensor) -> torch.Tensor:
        grid = keypoints_2d * 2.0 - 1.0
        sampled = F.grid_sample(feature_map, grid[:, :, None, :],
                                mode="bilinear", padding_mode="zeros",
                                align_corners=self.config.align_corners)
        return sampled[:, :, :, 0].transpose(1, 2)

    def forward(self, images: torch.Tensor, keypoints_2d: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if images.ndim != 4 or keypoints_2d.ndim != 3:
            raise ValueError("images must be [B,3,H,W], keypoints_2d [B,J,2]")
        if keypoints_2d.shape[1] != self.config.num_joints:
            raise ValueError("keypoint joint count does not match config")
        xy = root_relative_xy(keypoints_2d, self.config.left_hip_index,
                              self.config.right_hip_index)
        visual = self.sample_features(self.backbone(images), keypoints_2d)
        fused = torch.cat((self.xy_embedding(xy), self.visual_embedding(visual)),
                          dim=-1)
        depth = self.head(fused.flatten(1)).unsqueeze(-1)
        left = depth[:, self.config.left_hip_index]
        right = depth[:, self.config.right_hip_index]
        depth = depth - (left + right).div(2).unsqueeze(1)
        if valid_mask is not None:
            depth = depth * valid_mask.unsqueeze(-1).to(depth.dtype)
        return depth
