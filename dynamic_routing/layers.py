import torch
import einops

__all__ = ['InputStem', 'PrimaryCaps', 'DigitCaps']


class InputStem(torch.nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=9, stride=1):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            torch.nn.ReLU()
        )

    def forward(self, inputs):
        return self.stem(inputs)


class Router:
    def route(self, inputs):
        raise NotImplementedError


class PrimaryCaps(torch.nn.Module):
    def __init__(self, in_channels, out_channels=8, kernel_size=9, n_capsules=32, stride=2, depthwise=False):
        super().__init__()
        self.n_capsules = n_capsules
        self.out_channels = out_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels * n_capsules, kernel_size, stride,
                                    groups=out_channels * n_capsules if depthwise else 1)

    def to_capsule(self, inputs):
        args = dict(n=self.n_capsules, d=self.out_channels)
        return einops.rearrange(inputs, 'b ... (n d) h w -> b ... (h w n) d', **args)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.to_capsule(outputs)
        return outputs


class CapsuleTransform(torch.nn.Module):
    def __init__(self, in_features, in_capsules, out_features, out_capsules):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_features, in_capsules, out_features, out_capsules))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1 / self.weight.data.size(0) ** 0.5
        self.weight.data.uniform_(-std, std)

    def forward(self, inputs):
        return einops.einsum(inputs, self.weight, 'b ... n_in d_in, d_in n_in d_out n_out -> b ... n_in d_out n_out')


class AgreementRouter(torch.nn.Module, Router):
    def __init__(self, num_features, num_capsule):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(num_features, num_capsule))
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()


if __name__ == '__main__':
    inputs = torch.rand(64, 1, 28, 28)

    stem = InputStem(1)
    primary_caps = PrimaryCaps(256)
    digit_caps = CapsuleTransform(8, 32 * 6 * 6, 10, 16)

    print(sum(p.numel() for p in stem.parameters()) + sum(p.numel() for p in primary_caps.parameters()) + sum(
        p.numel() for p in digit_caps.parameters()))

    torch.empty(16, 10).zero_()

    print(digit_caps(primary_caps(stem(inputs))).size())
