from typing import Optional

import torch


class Squash(torch.nn.Module):
    def __init__(self, p: Optional[int] = 2, dim: Optional[int] = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def __repr__(self):
        return f'{type(self).__name__}(p={self.p}, dim={self.dim})'

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.norm(p=self.p, dim=self.dim, keepdim=True)


class SquashDR(Squash):
    def __init__(self, p: Optional[int] = 2, dim: Optional[int] = -1):
        super().__init__(p, dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        norm = super().forward(inputs)
        return inputs * norm / (1 + norm ** 2)


class SquashSA(Squash):
    def __init__(self, p: Optional[int] = 2, dim: Optional[int] = -1):
        super().__init__(p, dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        norm = super().forward(inputs)
        return inputs / norm * (1 - 1 / norm.exp())


if __name__ == '__main__':
    print(SquashDR())
    print(SquashSA())
