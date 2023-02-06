import torch

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


class Capsule:
    def __init__(self, num_routing=-1):
        self.num_routing = num_routing

    @staticmethod
    def squash(inputs):
        norm = inputs.norm(p=2, dim=-1, keepdim=True)
        return inputs * norm / (1 + norm ** 2)

    def route(self, inputs):
        if self.num_routing < 0:
            raise ValueError("Cannot do routing with num_routing < 0!")

        routers = torch.zeros_like(inputs)
        for i in range(self.num_routing):
            agreements = torch.softmax(routers, -1)
            outputs = self.squash(torch.sum(inputs * agreements, 1, keepdim=True))

            if i < self.num_routing - 1:
                routers = routers + torch.sum(inputs * outputs, -1, keepdim=True)
        return outputs


class PrimaryCaps(torch.nn.Module, Capsule):
    def __init__(self, in_channels, out_channels=8, kernel_size=9, n_capsules=32, stride=2):
        super().__init__()
        self.n_capsules = n_capsules
        self.capsule = torch.nn.Conv2d(in_channels, out_channels * n_capsules, kernel_size, stride)

    def forward(self, inputs):
        outputs = self.capsule(inputs)
        batch_size = outputs.size(0)
        return outputs.view(batch_size, -1, self.capsule.out_channels // self.n_capsules)


class DigitCaps(torch.nn.Module, Capsule):
    def __init__(self, in_features, out_features=10, in_capsules=8, out_capsules=16, num_routing=3):
        torch.nn.Module.__init__(self)
        Capsule.__init__(self, num_routing)
        self.out_features = out_features
        self.out_capsules = out_capsules
        self.transform = torch.nn.Linear(in_features * in_capsules, out_features * out_capsules)

    def forward(self, inputs):
        inputs = torch.flatten(inputs, 1)
        outputs = self.route(inputs)
        return outputs.view(-1, self.out_features, self.out_capsules).norm(p=2, dim=-1)
