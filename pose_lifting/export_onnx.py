import argparse
import torch

from .config import PoseConfig
from .model import DepthLiftingModel


def export_model(model: DepthLiftingModel, output: str, opset: int = 18) -> None:
    model.eval()
    torch.manual_seed(0)
    images = torch.randn(1, 3, 224, 224)
    keypoints = torch.rand(1, 15, 2)
    torch.onnx.export(
        model, (images, keypoints), output,
        input_names=["images", "keypoints_2d"],
        output_names=["root_relative_depth"],
        opset_version=opset, dynamic_axes=None, dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="depth_lifting.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    model = DepthLiftingModel(PoseConfig()).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                         weights_only=True))
    export_model(model, args.output, args.opset)
    print(f"exported {args.output}; verify with compare_onnx.py")


if __name__ == "__main__":
    main()
