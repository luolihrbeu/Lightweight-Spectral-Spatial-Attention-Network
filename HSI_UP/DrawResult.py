import argparse
from hyper_pytorch import *
import torch
import torch.nn.parallel
from models.MobileNetV3 import MobileNetV3
import os
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import spectral
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def load_hyper(args):
    data, labels, num_class = loadData(args.dataset, num_components=args.components)
    pixels, labels = createImageCubes(data, labels, windowSize=args.spatialsize, removeZeroLabels=False)
    bands = pixels.shape[-1]
    map_shape = data.shape[0:2]
    all_hyper = HyperData((np.transpose(pixels, (0, 3, 1, 2)).astype("float32"), labels), None)
    del pixels, labels
    kwargs = {'num_workers': 0, 'pin_memory': True}
    all_loader = torch.utils.data.DataLoader(all_hyper, batch_size=128, shuffle=False, **kwargs)
    return all_loader, num_class, bands, map_shape


def loadData(name, num_components=None):
    data_path = "data"
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SV':
        data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
    elif name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    else:
        print("NO DATASET")
        exit()

    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    if num_components != None:
        data = PCA(n_components=num_components).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components
    data = StandardScaler().fit_transform(data.astype(float))
    data = data.reshape(shapeor)
    num_class = len(np.unique(labels)) - 1
    return data, labels, num_class


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode="constant")

    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]

            # import matplotlib.pyplot as plt
            # plt.imshow(patch[:, :, 100])
            # plt.show()

            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels.astype("int")


def predict(all_loader, model):
    model.eval()
    predicted = []
    for batch_idx, (inputs, targets) in enumerate(all_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        [predicted.append(a) for a in model(inputs).data.cpu().numpy()]
    return np.array(predicted)


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth[1] * 2.0 / dpi, ground_truth[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 218, 185]) / 255.
        if item == 1:
            y[index] = np.array([150, 205, 205]) / 255.
        if item == 2:
            y[index] = np.array([0, 229, 238]) / 255.
        if item == 3:
            y[index] = np.array([0, 139, 139]) / 255.
        if item == 4:
            y[index] = np.array([0, 0, 205]) / 255.
        if item == 5:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 6:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 7:
            y[index] = np.array([255, 106, 106]) / 255.
        if item == 8:
            y[index] = np.array([255, 69, 0]) / 255.
        if item == 9:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 10:
            y[index] = np.array([205, 38, 38]) / 255.
        if item == 11:
            y[index] = np.array([205, 0, 205]) / 255.
        if item == 12:
            y[index] = np.array([139, 0, 139]) / 255.
        if item == 13:
            y[index] = np.array([105, 105, 105]) / 255.
        if item == 14:
            y[index] = np.array([79, 79, 79]) / 255.
        if item == 15:
            y[index] = np.array([54, 54, 54]) / 255.
        if item == 16:
            y[index] = np.array([255, 255, 255]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y


def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--components', default=16, type=int, help='dimensionality reduction')
    parser.add_argument('--dataset', default='UP', type=str, help='dataset (options: IP, UP, SV, KSC)')
    parser.add_argument('--spatialsize', dest='spatialsize', default=9, type=int,
                        help='spatial-spectral patch dimension')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    args = parser.parse_args()

    all_loader, num_classes, n_bands, map_shape = load_hyper(args)
    print("已加载全部数据集")
    model = MobileNetV3(n_class=num_classes, num_bands=n_bands)
    print("已加载模型")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    checkpoint = torch.load("imp_temp_best_model_UP_1.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    predicted = np.argmax(predict(all_loader, model), axis=1)
    print("得到预测结果")
    predicted = list_to_colormap(predicted)
    predicted = np.reshape(predicted, (map_shape[0], map_shape[1], 3))
    classification_map(predicted, map_shape, 300, "./Proposed_UP.png")


if __name__ == '__main__':
    main()