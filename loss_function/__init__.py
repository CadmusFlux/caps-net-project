from typing import Optional

import torch
import torch.nn.functional as F

from .helper import reduce


class MarginLoss(torch.nn.Module):
    def __init__(self, pos_weight: Optional[float] = 0.9, lambda_: Optional[float] = 0.5,
                 reduction: Optional[str] = 'mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.lambda_ = lambda_

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = F.one_hot(targets, outputs.size(-1))

        loss_pos = targets * torch.relu(self.pos_weight - outputs) ** 2
        loss_neg = (1 - targets) * torch.relu(outputs - self.pos_weight + 1) ** 2

        loss = (loss_pos + self.lambda_ * loss_neg).sum(-1)

        return reduce(loss, self.reduction)
