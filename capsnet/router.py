from typing import Optional

import torch
from einops import einsum

__all__ = ['AgreementRouter', 'SelfAttentionRouter']


class Router(torch.nn.Module):
    def __init__(self, num_features: int, num_capsules: int,
                 activation: Optional[torch.nn.Module] = torch.nn.Identity):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(num_capsules, num_features))
        self.activation = activation()

    def _args_str(self) -> str:
        args = ''
        args += f'num_capsules={self.bias.data.size(0)},'
        args += f' num_features={self.bias.data.size(1)},'
        args += f' activation={self.activation}'
        return args

    def __repr__(self) -> str:
        args = self._args_str()
        return f'{type(self).__name__}({args})'

    def reset_parameters(self) -> None:
        self.bias.data.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.bias)


class AgreementRouter(Router):
    def __init__(self, num_features: int, num_capsules: int, num_routing: Optional[int] = 3,
                 activation: Optional[torch.nn.Module] = torch.nn.Identity):
        super().__init__(num_features, num_capsules, activation)
        self.num_routing = num_routing if num_routing > 0 else 1

    def _args_str(self) -> str:
        args = f'num_capsules={self.bias.data.size(0)},'
        args += f' num_capsules={self.bias.data.size(1)},'
        args += f' num_routing={self.num_routing},'
        args += f' activation={self.activation}'
        return args

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        priors = torch.zeros_like(inputs)

        for i in range(self.num_routing):
            agreements = torch.softmax(priors, dim=-2)
            outputs = (inputs * agreements).sum(dim=1, keepdim=True)
            outputs = outputs + self.bias
            outputs = self.activation(outputs)
            if i < self.num_routing - 1:
                priors = priors + (inputs * outputs).sum(dim=-1, keepdim=True)
        return outputs.squeeze(dim=1)


class SelfAttentionRouter(Router):
    def __init__(self, num_features: int, num_capsules: int, activation: Optional[torch.nn.Module] = torch.nn.Identity):
        super().__init__(num_features, num_capsules, activation)
        self.pattern = '... i j, ... k j -> ... i'
        self.num_features = num_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = einsum(inputs, inputs, self.pattern)
        outputs = outputs.unsqueeze(-1)
        outputs = outputs / self.num_features ** 0.5
        outputs = torch.softmax(outputs, dim=-2)
        outputs = outputs + self.bias
        outputs = (inputs * outputs).sum(dim=1)
        return self.activation(outputs)


if __name__ == '__main__':
    router1 = AgreementRouter(16, 10)
    router2 = SelfAttentionRouter(16, 10)
    tensor = torch.rand(64, 32, 10, 16)
    print(router1)
    print(router2)
    print(router1(tensor).size())
    print(router2(tensor).size())
