import torch
from pose_lifting.coordinates import CropTransform, root_relative_depth, root_relative_xy


def test_root_relative_and_depth_condition():
    xy = torch.tensor([[[1., 2.], [3., 4.], [5., 8.]]])
    assert torch.equal(root_relative_xy(xy, 0, 1),
                       torch.tensor([[[-1., -1.], [1., 1.], [3., 5.]]]))
    depth = torch.tensor([[2., 6., 10.]])
    assert torch.equal(root_relative_depth(depth, 0, 1),
                       torch.tensor([[-2., 2., 6.]]))


def test_crop_normalized_and_grid_conversion():
    transform = CropTransform(100, 50, 200, 100, 224, 224,
                              source_width=400, source_height=200)
    xy = torch.tensor([[.5, .5]])
    assert torch.allclose(transform.source_normalized_to_crop_normalized(xy),
                          torch.tensor([[.5, .5]]))
    assert torch.allclose(transform.crop_normalized_to_grid(xy),
                          torch.tensor([[0., 0.]]))
