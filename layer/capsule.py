from typing import Optional, Union, Tuple

import torch
from einops import rearrange, einsum


class PrimaryCapsule(torch.nn.Module):
    def __init__(self, in_channels: int, out_capsules: int, out_features: int,
                 kernel_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]] = 1,
                 depthwise: Optional[bool] = False):
        super().__init__()
        self.out_capsules = out_capsules
        self.out_features = out_features
        self.convolve = torch.nn.Conv2d(in_channels, out_capsules * out_features, kernel_size, stride,
                                        groups=out_capsules * out_features if depthwise else 1)
        self.pattern = '... (n d) h w -> ... (n h w) d'

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.convolve(inputs)
        return rearrange(outputs, self.pattern, n=self.out_capsules, d=self.out_features)


class CapsuleTransform(torch.nn.Module):
    def __init__(self, in_capsules: int, in_features: int, out_capsules: int, out_features: int):
        super().__init__()
        self.pattern = '... n_in d_in, n_in d_in n_out d_out -> ... n_in n_out d_out'
        self.weight = torch.nn.Parameter(torch.empty(in_capsules, in_features, out_capsules, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight.data)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return einsum(inputs, self.weight, self.pattern)


if __name__ == '__main__':
    inputs = torch.rand(128, 1, 28, 28)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 256, 9),
        torch.nn.ReLU(),
        PrimaryCapsule(256, 32, 8, 9, 2),
        CapsuleTransform(32 * 6 * 6, 8, 10, 16)
    )

    print(model(inputs).size())
    print(f'{sum(p.numel() for p in model.parameters()):,}')
