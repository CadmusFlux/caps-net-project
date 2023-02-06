import torch


class MarginLoss(torch.nn.Module):
    def __init__(self, pos_weight=0.9, lambda_=0.5, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.lambda_ = lambda_

    def forward(self, outputs, targets):
        targets = F.one_hot(targets, outputs.size(-1))

        loss_pos = targets * torch.relu(self.pos_weight - outputs) ** 2
        loss_neg = (1 - targets) * torch.relu(outputs - self.pos_weight + 1) ** 2

        loss = (loss_pos + self.lambda_ * loss_neg).sum(-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
