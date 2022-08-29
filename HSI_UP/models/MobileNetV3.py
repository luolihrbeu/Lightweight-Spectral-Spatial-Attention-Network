import torch
import torch.nn as nn
import torch.nn.functional as F
from dcn_v2 import DCNv2
from util.utils import eca_layer

__all__ = ['MobileNetV3', 'mobilenetv3']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # 此处的激活函数待定
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c).contiguous()
        y = self.fc(y).view(b, c, 1, 1).contiguous()
        return x * y.expand_as(x)


class SingPointAttention(nn.Module):
    def __init__(self, channel):
        super(SingPointAttention, self).__init__()

    def forward(self, x):
        bs, c, h, w = x.size()
        center = x[:, :, h // 2:h // 2 + 1, w // 2:w // 2 + 1]
        similarity = torch.sigmoid(center * x.sum(dim=1).unsqueeze(1))
        # similarity = center * x.sum(dim=1).unsqueeze(1)
        out = similarity * x
        return out


class SpaceAttention(nn.Module):
    def __init__(self, channel):
        super(SpaceAttention, self).__init__()

    def forward(self, cube):
        bs, c, h, w = cube.size()
        cube_key = cube.view(bs, c, -1).permute(0, 2, 1)
        cube_center = cube[:, :, h // 2:h // 2 + 1, w // 2:w // 2 + 1].squeeze(-1)
        similarity = torch.matmul(cube_key, cube_center).permute(0, 2, 1).view(bs, 1, h, w).sigmoid()
        out = cube * similarity

        return out


class DCN(nn.Module):
    def __init__(self, in_channel, out_channel, kernal_size=3, stride=1, padding=1):
        super(DCN, self).__init__()
        self.conv_offset = nn.Conv2d(in_channel, 2 * kernal_size * kernal_size, kernel_size=kernal_size,
                                     stride=stride, padding=padding, bias=True)
        self.conv_mask = nn.Conv2d(in_channel, kernal_size * kernal_size, kernel_size=kernal_size,
                                   stride=stride, padding=padding, bias=True)
        self.dcn = DCNv2(in_channel, out_channel, kernel_size=kernal_size, stride=stride,
                         padding=padding, dilation=1, deformable_groups=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = Mish()

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = self.conv_mask(x)
        mask = torch.sigmoid(mask)
        out = self.dcn(x, offset, mask)
        out = self.bn(out)
        out = self.activation(out)
        return out


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, core="se", nl='RE', dcn=False):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d

        if nl == 'RE':
            nlin_layer = nn.ReLU6  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError

        if core == "se":
            CoreLayer = SEModule
        elif core == "spa":
            CoreLayer = SpaceAttention
        elif core == "eca":
            CoreLayer = eca_layer
        else:
            CoreLayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            # conv_layer(exp, exp, kernel, stride, padding, bias=False),
            norm_layer(exp),
            CoreLayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_class=9, num_bands=16, dropout=0.2, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 128
        last_channel = n_class
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,   nl,  s,
                [3, 16, 16, "spa", 'RE', 1],
                [3, 64, 24, "", 'RE', 2],
                [3, 72, 24, "", 'RE', 1],
                [5, 72, 40, "se", 'RE', 2],
                [5, 120, 40, "se", 'RE', 1],
                [5, 120, 40, "se", 'RE', 1],
                [3, 240, 80, "", 'HS', 2],
                [3, 200, 80, "", 'HS', 1],
                [3, 184, 80, "", 'HS', 1],
                [3, 184, 80, "", 'HS', 1],
                [3, 480, 112, "se", 'HS', 1],
                [3, 672, 112, "se", 'HS', 1],
                [5, 672, 160, "se", 'HS', 2],
                [5, 960, 160, "se", 'HS', 1],
                [5, 960, 160, "se", 'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,  nl,  s,
                # [3, 128, 128, "", "RE", 1],
                # [3, 128, 128, "se", "RE", 1],
                [3, 128, 64, "se", "RE", 1],
                [3, 64, 32, "spa", "RE", 1],
                [3, 64, 32, "se", "RE", 1],
                [3, 32, 32, "se", "RE", 1],
                # [3,  32,  32, "spa", "RE", 1],
                # [3,  32,  32, "se", "RE", 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        # assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(num_bands, input_channel, 1, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(32 * width_mult)
            # self.features.append(SpaceAttention(None))
            # self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv, reduction=2))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(p=dropout),  # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.mean(3).mean(2)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=True)
        # raise NotImplementedError
    return model


if __name__ == '__main__':
    net = mobilenetv3()
    print(net)
    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    from thop import profile

    flops, params = profile(net, inputs=(torch.randn(1, 16, 9, 9),))
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))
    # print('mobilenetv3:\n', net)
    # print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    # input_size = (10, 128, 9, 9)
    # x = torch.randn(input_size)
    # out = net(x)
    # print(out.size())
