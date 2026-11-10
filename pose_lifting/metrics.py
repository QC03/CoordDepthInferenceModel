import torch


def root_relative_depth_mae_mm(prediction: torch.Tensor,
                               target: torch.Tensor,
                               valid_mask: torch.Tensor) -> torch.Tensor:
    error = (prediction.squeeze(-1) - target.squeeze(-1)).abs() * 1000
    return (error * valid_mask).sum() / valid_mask.sum().clamp_min(1)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)
