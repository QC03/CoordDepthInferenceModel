import argparse
import numpy as np
import torch

from .config import PoseConfig
from .model import DepthLiftingModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    import onnxruntime as ort
    torch.manual_seed(0)
    torch_model = DepthLiftingModel(PoseConfig()).eval()
    if args.checkpoint:
        torch_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                               weights_only=True))
    images = torch.randn(1, 3, 224, 224)
    keypoints = torch.rand(1, 15, 2)
    with torch.no_grad():
        expected = torch_model(images, keypoints).numpy()
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    actual = session.run(["root_relative_depth"],
                         {"images": images.numpy(),
                          "keypoints_2d": keypoints.numpy()})[0]
    print(f"max_abs_error={np.max(np.abs(expected - actual)):.8g}")


if __name__ == "__main__":
    main()
