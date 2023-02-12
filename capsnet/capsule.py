from typing import Optional

import torch
from einops import einsum, rearrange


class PrimaryCapsule(torch.nn.Module):
    def __init__(self, in_channels: int, out_features: int, num_capsules: int, kernel_size: int,
                 stride: Optional[int] = 1, depthwise: Optional[bool] = False,
                 activation: Optional[torch.nn.Module] = torch.nn.Identity):
        super().__init__()
        self.num_capsules = num_capsules
        self.out_features = out_features
        self.conv = torch.nn.Conv2d(in_channels, out_features * num_capsules, kernel_size, stride,
                                    groups=out_features * num_capsules if depthwise else 1)
        self.pattern = '... (n d) h w -> ... (h w n) d'
        self.activation = activation()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        outputs = rearrange(outputs, self.pattern, n=self.num_capsules, d=self.out_features)
        return self.activation(outputs)


class CapsuleTransform(torch.nn.Module):
    def __init__(self, in_features, in_capsules, out_features, out_capsules):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_features, in_capsules, out_features, out_capsules))
        self.pattern = '... n_in d_in, d_in n_in d_out n_out -> ... n_in n_out d_out'
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1 / self.weight.data.size(0) ** 0.5
        self.weight.data.uniform_(-std, std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return einsum(inputs, self.weight, self.pattern)


if __name__ == '__main__':
    inputs = torch.rand(64, 256, 9, 9)
    prim = PrimaryCapsule(256, 8, 32, 9, depthwise=True)
    tran = CapsuleTransform(8, 32, 16, 10)
    print(prim(inputs).size())
    print(tran(prim(inputs)).size())
