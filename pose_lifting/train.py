import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PoseConfig
from .dataset import Human36MDataset, VideoHuman36MDataset
from .losses import MaskedSmoothL1Loss
from .metrics import count_parameters, root_relative_depth_mae_mm
from .model import DepthLiftingModel
from .export_onnx import export_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--data-root", default="train")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--pose-model")
    parser.add_argument("--checkpoint-output", default="model.pth")
    parser.add_argument("--onnx-output", default="depth_lifting.onnx")
    args = parser.parse_args()
    config = PoseConfig()
    if args.manifest:
        dataset = Human36MDataset(args.manifest, args.subjects, 7, 8,
                                  h36m_indices=list(range(15)))
    else:
        dataset = VideoHuman36MDataset(
            args.data_root, args.subjects, 7, 8,
            frame_stride=args.frame_stride, max_samples=args.max_samples,
            pose_model=args.pose_model)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = DepthLiftingModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = MaskedSmoothL1Loss()
    print(f"parameters={count_parameters(model)}")
    total_steps = args.epochs * len(loader)
    progress = tqdm(total=total_steps, desc="training", unit="step")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            prediction = model(batch["images"], batch["keypoints_2d"])
            loss = criterion(prediction, batch["target_depth"],
                             batch["valid_mask"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress.update(1)
            progress.set_postfix(epoch=epoch + 1,
                                 loss=f"{loss.item():.5f}")
        print(f"epoch {epoch + 1}/{args.epochs} "
              f"loss={epoch_loss / max(len(loader), 1):.5f}")
    progress.close()
    checkpoint_path = Path(args.checkpoint_output)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path}")
    export_model(model, args.onnx_output)
    print(f"saved ONNX model: {args.onnx_output}")
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        for batch in loader:
            prediction = model(batch["images"], batch["keypoints_2d"])
        elapsed = (time.perf_counter() - start) / max(len(loader), 1)
        mae = root_relative_depth_mae_mm(prediction, batch["target_depth"],
                                         batch["valid_mask"])
    print(f"last_batch_mae_mm={mae.item():.3f} inference_ms={elapsed * 1000:.3f}")


if __name__ == "__main__":
    main()
