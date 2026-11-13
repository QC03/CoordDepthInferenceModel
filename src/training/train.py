import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..config.config import PoseConfig
from ..data.dataset import (
    H36M_TO_MODEL_INDICES,
    Human36MDataset,
    VideoHuman36MDataset,
)
from .losses import MaskedSmoothL1Loss
from .metrics import count_parameters, root_relative_depth_mae_mm
from ..model.model import DepthLiftingModel
from ..inference.export_onnx import export_model


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        cuda_build = torch.version.cuda or "none (CPU-only build)"
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            f"Installed PyTorch CUDA build: {cuda_build}. "
            "Install the NVIDIA build with: "
            "python -m pip install -r requirements-gpu.txt"
        )
    device_name = (
        "cuda" if requested == "auto" and torch.cuda.is_available()
        else "cpu" if requested == "auto"
        else requested
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    elif requested == "auto" and torch.version.cuda is None:
        print(
            "CUDA unavailable: the installed PyTorch is CPU-only. "
            "Install requirements-gpu.txt to enable NVIDIA GPU training."
        )
    return device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--data-root", default="train")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--eval-max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pose-mode", choices=("lightweight", "balanced",
                                                 "performance"),
                        default="lightweight")
    parser.add_argument("--pose-device", choices=("auto", "cuda", "cpu"),
                        default="auto")
    parser.add_argument("--checkpoint-output", default="model.pth")
    parser.add_argument("--onnx-output", default="depth_lifting.onnx")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"),
                        default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.eval_max_samples < 1:
        parser.error("--eval-max-samples must be at least 1")
    device = _select_device(args.device)
    print(f"device={device}")
    config = PoseConfig()
    if args.manifest:
        dataset = Human36MDataset(
            args.manifest, args.subjects, 7, 8,
            h36m_indices=list(H36M_TO_MODEL_INDICES),
        )
    else:
        dataset = VideoHuman36MDataset(
            args.data_root, args.subjects, 7, 8,
            frame_stride=args.frame_stride, max_samples=args.max_samples,
            pose_mode=args.pose_mode, pose_device=args.pose_device)
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(dataset, **loader_options)
    print(f"samples={len(dataset)} epochs={args.epochs} "
          f"updates={args.epochs * len(loader)} batch_size={args.batch_size} "
          f"workers={args.num_workers}")
    model = DepthLiftingModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = MaskedSmoothL1Loss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"parameters={count_parameters(model)}")
    total_steps = args.epochs * len(loader)
    progress = tqdm(total=total_steps, desc="training", unit="step")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(batch["images"], batch["keypoints_2d"])
                loss = criterion(prediction, batch["target_depth"],
                                 batch["valid_mask"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress.update(1)
            progress.set_postfix(epoch=epoch + 1,
                                 loss=f"{loss.item():.5f}")
        print(f"epoch {epoch + 1}/{args.epochs} "
              f"loss={epoch_loss / max(len(loader), 1):.5f}")
    progress.close()
    checkpoint_path = Path(args.checkpoint_output)
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        checkpoint_path,
    )
    print(f"saved checkpoint: {checkpoint_path}")
    export_model(model, args.onnx_output)
    print(f"saved ONNX model: {args.onnx_output}")
    model.eval()
    eval_size = min(args.eval_max_samples, len(dataset))
    eval_dataset = Subset(dataset, range(eval_size))
    eval_options = dict(loader_options)
    eval_options["shuffle"] = False
    eval_loader = DataLoader(eval_dataset, **eval_options)
    print(f"evaluating samples={eval_size}")
    total_mae = 0.0
    evaluated_batches = 0
    with torch.no_grad():
        start = time.perf_counter()
        eval_progress = tqdm(
            eval_loader, total=len(eval_loader), desc="evaluation",
            unit="batch"
        )
        for batch in eval_progress:
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(batch["images"], batch["keypoints_2d"])
            total_mae += root_relative_depth_mae_mm(
                prediction, batch["target_depth"], batch["valid_mask"]
            ).item()
            evaluated_batches += 1
        eval_progress.close()
        elapsed = (time.perf_counter() - start) / max(len(eval_loader), 1)
    mae = total_mae / max(evaluated_batches, 1)
    print(f"mean_mae_mm={mae:.3f} inference_ms={elapsed * 1000:.3f}")


if __name__ == "__main__":
    main()
