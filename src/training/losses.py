import torch
from torch import nn


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 0.01) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        loss = nn.functional.smooth_l1_loss(prediction, target, beta=self.beta,
                                             reduction="none")
        mask = valid_mask.to(loss.dtype).unsqueeze(-1)
        denom = mask.sum().clamp_min(1)
        return (loss * mask).sum() / denom

