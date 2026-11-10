import pytest
import torch
from torch import nn
from pose_lifting.config import PoseConfig
from pose_lifting.model import DepthLiftingModel


class PatternBackbone(nn.Module):
    def forward(self, x):
        # channel 0 is x coordinate and channel 1 is y coordinate.
        h, w = 7, 7
        xx = torch.arange(w, device=x.device).view(1, 1, 1, w).expand(
            x.shape[0], 1, h, w)
        yy = torch.arange(h, device=x.device).view(1, 1, h, 1).expand(
            x.shape[0], 1, h, w)
        return torch.cat((xx, yy, torch.zeros(x.shape[0], 510, h, w,
                                               device=x.device)), dim=1)


def test_sampling_known_pattern():
    config = PoseConfig(num_joints=15, feature_channels=512)
    model = DepthLiftingModel(config, backbone=PatternBackbone())
    fmap = PatternBackbone()(torch.zeros(1, 3, 224, 224))
    points = torch.tensor([[[0., 0.], [.5, .5]] + [[.5, .5]] * 13])
    sampled = model.sample_features(fmap, points)
    assert torch.allclose(sampled[0, 0, :2], torch.tensor([0., 0.]))
    assert torch.allclose(sampled[0, 1, :2], torch.tensor([3., 3.]))


@pytest.mark.skipif(not hasattr(torch, "onnx"), reason="PyTorch ONNX unavailable")
def test_model_root_output_shape():
    model = DepthLiftingModel(PoseConfig(), backbone=PatternBackbone()).eval()
    output = model(torch.randn(1, 3, 224, 224), torch.rand(1, 15, 2))
    assert output.shape == (1, 15, 1)
    assert torch.allclose((output[:, 7] + output[:, 8]) / 2,
                          torch.zeros(1, 1), atol=1e-6)
