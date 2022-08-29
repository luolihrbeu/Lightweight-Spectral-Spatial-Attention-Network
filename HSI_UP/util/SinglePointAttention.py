import torch
import torch.nn as nn


class SingPointAttention(nn.Module):
    def __init__(self):
        super(SingPointAttention, self).__init__()

    def forward(self, x):
        bs, c, h, w = x.size()
        center = x[:, :, h // 2:h // 2 + 1, w // 2:w // 2 + 1]
        similarity = (center * x).sum(dim=1).unsqueeze(1)
        out = similarity * x
        return out


if __name__ == '__main__':
    temp = torch.rand(2, 3, 5, 5)
    model = SingPointAttention()
    out = model(temp)
    print(out.shape)
