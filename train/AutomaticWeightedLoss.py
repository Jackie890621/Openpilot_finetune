import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2, sigma_clamp: float = 1e-3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.sigma_clamp = sigma_clamp

    def forward(self, *x):
        loss_sum = 0.0
        for i, loss in enumerate(x):
            sigma = torch.clamp(self.params[i], min=self.sigma_clamp)
            loss_sum += loss * torch.exp(-sigma)  + sigma
            # print(loss * torch.exp(-sigma)  + sigma)
        return loss_sum

if __name__ == '__main__':
    loss1 = 250
    loss2 = 120
    loss3 = -2
    awl = AutomaticWeightedLoss(3)
    loss_sum = awl(loss1, loss2, loss3)