import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def to_categorical(ytrue):
    input_shape = ytrue.size()
    n = ytrue.size(0)
    categorical = torch.zeros(n, 10)
    categorical[torch.arange(n), ytrue] = 1
    output_shape = input_shape + (10,)
    categorical = torch.reshape(categorical, output_shape)
    return categorical


class FocalLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=0.01, gamma=1):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, ypred, ytrue):
        logpt = F.log_softmax(ypred, 1)
        pt = Variable(torch.exp(logpt))
        ytrue = to_categorical(ytrue)
        pt_prime = 1 - pt
        focalloss = -self.alpha * (pt_prime) ** self.gamma * ytrue * logpt
        focalloss = torch.sum(focalloss, 1)
        if (self.reduction == 'sum'):
            return focalloss.sum()
        else:
            return focalloss.mean()


class FocalLoss2(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2.75, weight=None):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


class FocalLoss3(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss3, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        nn.CrossEntropyLoss()

        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

if __name__ == '__main__':
    loss = FocalLoss(reduction="mean", alpha=1, gamma=2)
    loss2 = FocalLoss2()
    loss3 = FocalLoss3()
    pre = torch.randn(3, 10)
    # print(pre)
    target = torch.LongTensor([3, 5, 8])
    loss_out = loss(pre, target)
    loss_out2 = loss2(pre, target)
    loss_out3 = loss3(pre, target)
    print(loss_out)
    print(loss_out2)
    print(loss_out3)
