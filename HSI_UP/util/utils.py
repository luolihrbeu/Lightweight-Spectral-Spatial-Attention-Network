import spectral
import numpy as np
import scipy.io as sio
from torch import nn
import os
import matplotlib.pyplot as plt


def map_result(data_class, data, out_dir, label, predicted, idx):
    if data_class == "IP":
        colors = np.array([[255, 255, 255],
                           [255, 218, 185],
                           [150, 205, 205],
                           [0, 229, 238],
                           [0, 139, 139],
                           [0, 0, 205],
                           [0, 255, 0],
                           [255, 255, 0],
                           [255, 105, 106],
                           [255, 69, 0],
                           [255, 0, 0],
                           [205, 38, 38],
                           [205, 0, 205],
                           [139, 0, 139],
                           [105, 105, 105]])
    elif data_class == "PU":
        colors = np.array([[255, 255, 255],
                           [255, 218, 185],
                           [150, 205, 205],
                           [0, 229, 238],
                           [0, 139, 139],
                           [0, 0, 205],
                           [0, 255, 0],
                           [255, 255, 0],
                           [255, 105, 106],
                           [255, 69, 0]])
    elif data_class == "SV":
        colors = np.array([[255, 255, 255],
                           [255, 218, 185],
                           [150, 205, 205],
                           [0, 229, 238],
                           [0, 139, 139],
                           [0, 0, 205],
                           [0, 255, 0],
                           [255, 255, 0],
                           [255, 105, 106],
                           [255, 69, 0],
                           [255, 0, 0],
                           [205, 38, 38],
                           [205, 0, 205],
                           [139, 0, 139],
                           [105, 105, 105]])
    else:
        colors = np.array([[255, 255, 255],
                           [255, 218, 185],
                           [150, 205, 205],
                           [0, 229, 238],
                           [0, 139, 139],
                           [0, 0, 205],
                           [0, 255, 0],
                           [255, 255, 0],
                           [255, 105, 106],
                           [255, 69, 0],
                           [255, 0, 0],
                           [205, 38, 38],
                           [205, 0, 205],
                           [139, 0, 139]])
    img = spectral.imshow(classes=data.astype(int), figsize=(9, 9), colors=colors)
    path_cam_labels = os.path.join(out_dir, "cam_lab-%s_pre-%s_idx-%s.jpg" % (label, predicted, idx))
    plt.savefig(path_cam_labels, dpi=100)
    plt.cla()
    plt.close()
    # plt.show()


def loadData(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'IP':
        labels = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
        return labels
    elif name == 'SV':
        labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
        return labels
    elif name == 'UP':
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
        return labels
    elif name == 'KSC':
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
        return labels
    else:
        print("NO DATASET")
        exit()


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
