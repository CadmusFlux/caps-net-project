from typing import Optional

import torch

from .activation import squash_dr


class DynamicRouter(torch.nn.Module):
    def __init__(self, num_capsules: int, num_features: int, num_routing: Optional[int] = 3):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(num_capsules, num_features))
        self.num_routing = num_routing

    def reset_parameters(self):
        self.bias.data.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        priors = torch.zeros_like(inputs)
        for i in range(self.num_routing):
            agreements = torch.softmax(priors, -2)
            outputs = torch.sum(inputs * agreements, dim=-3, keepdim=True)
            outputs = squash_dr(outputs + self.bias)
            if i < self.num_routing - 1:
                priors = priors + (inputs * outputs).sum(dim=-1, keepdim=True)
        return outputs[:, 0]


if __name__ == '__main__':
    inputs = torch.rand(128, 1152, 10, 16)
    print(DynamicRouter(10, 16)(inputs).size())
