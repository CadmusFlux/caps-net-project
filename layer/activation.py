from typing import Optional

import torch


class Squash(torch.nn.Module):
    def __init__(self, kind: Optional[str] = 'identity', p: Optional[int] = 2, dim: Optional[int] = -1):
        super().__init__()
        self.kind = kind
        self.p = p
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.kind == 'dr':
            return squash_dr(inputs, self.p, self.dim)

        if self.kind == 'sa':
            return squash_sa(inputs, self.p, self.dim)

        return inputs


def squash_dr(inputs: torch.Tensor, p: Optional[int] = 2, dim: Optional[int] = -1) -> torch.Tensor:
    norm = inputs.norm(p=p, dim=dim, keepdim=True)
    return inputs * norm / (1 + norm ** 2)


def squash_sa(inputs: torch.Tensor, p: Optional[int] = 2, dim: Optional[int] = -1) -> torch.Tensor:
    norm = inputs.norm(p=p, dim=dim, keepdim=True)
    return inputs * norm / (1 + norm ** 2)
