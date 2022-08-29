import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv


def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    # 随机生成长度为len(a)的序列，将a和b按同样的随机顺序重新排列
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def split_data(pixels, labels, percent, splitdset="custom", rand_state=69):
    splitdset = "sklearn"
    if splitdset == "sklearn":
        # train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
        return train_test_split(pixels, labels, test_size=(1 - percent), stratify=labels, random_state=rand_state)
    elif splitdset == "custom":
        if (len(pixels.shape) > 3):
            # np.unique:该函数是去除数组中的重复数字，并进行排序之后输出。return_counts = true,返回个数（用于统计各个元素出现的次数）
            # pixels_number:各类标签的个数，按顺序记录在列表中。
            pixels_number = np.unique(labels, return_counts=True)[1]
            # 设置各类地物训练集的大小
            train_set_size = np.ones(len(pixels_number)) * percent
            # 训练集总大小
            tr_size = int(sum(train_set_size))
            # 测试集总大小
            te_size = int(sum(pixels_number)) - int(sum(train_set_size))
            # list(pixels.shape)[1:]表示每一个Cube的大小
            sizetr = np.array([tr_size] + list(pixels.shape[1:]))
            sizete = np.array([te_size] + list(pixels.shape[1:]))
            train_x = np.empty((sizetr))
            train_y = np.empty((tr_size))
            test_x = np.empty((sizete))
            test_y = np.empty((te_size))
            trcont = 0
            tecont = 0
            for cl in np.unique(labels):
                pixels_cl = pixels[labels == cl]
                labels_cl = labels[labels == cl]
                # 将pixels_cl, labels_cl随机重新排序
                pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
                for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
                    if cont < train_set_size[cl]:
                        train_x[trcont, :, :, :] = a
                        train_y[trcont] = b
                        trcont += 1
                    else:
                        test_x[tecont, :, :, :] = a
                        test_y[tecont] = b
                        tecont += 1
            train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
            test_x, test_y = random_unison(test_x, test_y, rstate=rand_state)
            return train_x, test_x, train_y, test_y

        else:
            # pixels_number:各类标签的个数，按顺序记录在列表中。
            pixels_number = np.unique(labels, return_counts=True)[1]
            # 设置各类地物训练集的大小
            # train_set_size = np.ones(len(pixels_number)) * percent
            train_set_size = np.ones(len(pixels_number)) * percent
            # 训练集总大小
            tr_size = int(sum(train_set_size))
            # 测试集总大小
            te_size = int(sum(pixels_number)) - int(sum(train_set_size))
            sizetr = np.array([tr_size])
            sizete = np.array([te_size])
            train_loc = np.empty((np.append(sizetr, [2])))
            train_y = np.empty((tr_size))
            test_loc = np.empty((np.append(sizete, [2])))
            test_y = np.empty((te_size))
            trcont = 0
            tecont = 0
            for cl in np.unique(labels):
                pixels_cl = pixels[labels == cl]
                labels_cl = labels[labels == cl]
                pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
                for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
                    if cont < train_set_size[cl]:
                        train_loc[trcont] = a
                        train_y[trcont] = b
                        trcont += 1
                    else:
                        test_loc[tecont] = a
                        test_y[tecont] = b
                        tecont += 1
            train_loc, train_y = random_unison(train_loc, train_y, rstate=rand_state)
            test_loc, test_y = random_unison(test_loc, test_y, rstate=rand_state)
            return train_loc.astype(np.int64), test_loc.astype(np.int64), \
                   train_y.astype(np.int64), test_y.astype(np.int64)

# def split_data(pixels, labels, percent, mode, splitdset="custom", rand_state=69):
#     """
#     :param pixels: len(pixels.shape) >3表示cube，小于则表示location
#     :param labels: 标签
#     :param percent: 训练集的比重，为整数时，表示每一类选取多少个作为训练集（splitdset="custom"时）
#     :param mode: CNN模式划分训练集和测试集，GAN模式只需要训练集
#     :param splitdset: 使用sklearn还是自己设计的划分方式，“sklearn”表示用sklearn，“custom”表示自己的
#     :param rand_state: 保证每次的划分方式相同
#     :return:
#     """
#     if mode == "CNN":
#         if splitdset == "sklearn":
#             # train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
#             return train_test_split(pixels, labels, test_size=(1 - percent), stratify=labels, random_state=rand_state)
#         elif splitdset == "custom":
#             if (len(pixels.shape) > 3):
#                 # np.unique:该函数是去除数组中的重复数字，并进行排序之后输出。return_counts = true,返回个数（用于统计各个元素出现的次数）
#                 # pixels_number:各类标签的个数，按顺序记录在列表中。
#                 pixels_number = np.unique(labels, return_counts=True)[1]
#                 # 设置各类地物训练集的大小
#                 # train_set_size = np.ones(len(pixels_number)) * percent
#                 train_set_size = [9, 285, 166, 47, 96, 146, 5, 95, 4, 194, 491, 118, 41, 253, 77, 18]
#                 # 训练集总大小
#                 tr_size = int(sum(train_set_size))
#                 # 测试集总大小
#                 te_size = int(sum(pixels_number)) - int(sum(train_set_size))
#                 sizetr = np.array([tr_size] + list(pixels.shape[1:]))
#                 sizete = np.array([te_size] + list(pixels.shape[1:]))
#                 train_x = np.empty((sizetr)); train_y = np.empty((tr_size))
#                 test_x = np.empty((sizete)); test_y = np.empty((te_size))
#                 trcont = 0
#                 tecont = 0
#                 for cl in np.unique(labels):
#                     pixels_cl = pixels[labels == cl]
#                     labels_cl = labels[labels == cl]
#                     pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
#                     for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
#                         if cont < train_set_size[cl]:
#                             train_x[trcont, :, :, :] = a
#                             train_y[trcont] = b
#                             trcont += 1
#                         else:
#                             test_x[tecont, :, :, :] = a
#                             test_y[tecont] = b
#                             tecont += 1
#                 train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
#                 test_x, test_y = random_unison(test_x, test_y, rstate=rand_state)
#                 return train_x, test_x, train_y, test_y
#             else:
#                 # np.unique:该函数是去除数组中的重复数字，并进行排序之后输出。return_counts = true,返回个数（用于统计各个元素出现的次数）
#                 # pixels_number:各类标签的个数，按顺序记录在列表中。
#                 pixels_number = np.unique(labels, return_counts=True)[1]
#                 # 设置各类地物训练集的大小
#                 # train_set_size = np.ones(len(pixels_number)) * percent
#                 train_set_size = [9, 285, 166, 47, 96, 146, 5, 95, 4, 194, 491, 118, 41, 253, 77, 18]
#                 # 训练集总大小
#                 tr_size = int(sum(train_set_size))
#                 # 测试集总大小
#                 te_size = int(sum(pixels_number)) - int(sum(train_set_size))
#                 sizetr = np.array([tr_size])
#                 sizete = np.array([te_size])
#                 train_loc = np.empty((np.append(sizetr, [2]))); train_y = np.empty((tr_size))
#                 test_loc = np.empty((np.append(sizete, [2]))); test_y = np.empty((te_size))
#                 trcont = 0
#                 tecont = 0
#                 for cl in np.unique(labels):
#                     pixels_cl = pixels[labels == cl]
#                     labels_cl = labels[labels == cl]
#                     pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
#                     for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
#                         if cont < train_set_size[cl]:
#                             train_loc[trcont] = a
#                             train_y[trcont] = b
#                             trcont += 1
#                         else:
#                             test_loc[tecont] = a
#                             test_y[tecont] = b
#                             tecont += 1
#                 train_loc, train_y = random_unison(train_loc, train_y, rstate=rand_state)
#                 test_loc, test_y = random_unison(test_loc, test_y, rstate=rand_state)
#                 return train_loc.astype(np.int64), test_loc.astype(np.int64), \
# 					   train_y.astype(np.int64), test_y.astype(np.int64)
#
#         elif mode == "GAN":
#             # np.unique:该函数是去除数组中的重复数字，并进行排序之后输出。return_counts = true,返回个数（用于统计各个元素出现的次数）
#             # pixels_number:各类标签的个数，按顺序记录在列表中。
#             pixels_number = np.unique(labels, return_counts=True)[1]
#             # 设置各类地物训练集的大小
#             train_set_size = [int(np.ceil(a * percent)) for a in pixels_number]
#             # 训练集总大小
#             tr_size = int(sum(train_set_size))
#             sizetr = np.array([tr_size] + list(pixels.shape[1:]))
#             train_x = np.empty((sizetr))
#             train_y = np.empty((tr_size))
#             trcont = 0
#             for cl in np.unique(labels):
#                 pixels_cl = pixels[labels == cl]
#                 labels_cl = labels[labels == cl]
#                 pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
#                 for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
#                     if cont < train_set_size[cl]:
#                         train_x[trcont, :, :, :] = a
#                         train_y[trcont] = b
#                         trcont += 1
#                     else:
#                         continue
#             train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
#             return train_x, train_y


def loadData(name, num_components=None):
    data_path = os.path.join(os.getcwd(), 'data')
    # print(data_path)
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
    # data = MinMaxScaler().fit_transform(data)
    # 归一化和标准化数据集
    data = StandardScaler().fit_transform(data)
    data = data.reshape(shapeor)
    num_class = len(np.unique(labels)) - 1
    return data, labels, num_class


# data的h和w进行padding，2倍的margin
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    # patchesData是一个四维矩阵，shape[0]是index,后面的三维是数据
    non_zero_label_num = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i, j] > 0:
                non_zero_label_num += 1
    patchesData = np.zeros((non_zero_label_num, windowSize, windowSize, X.shape[2]))
    # patchesLabels的大小和X.shape[0] * X.shape[1]一致，是一个一维向量。
    patchesLabels = np.zeros((non_zero_label_num))
    patchIndex = 0
    location = np.zeros((non_zero_label_num, 2))
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if removeZeroLabels:
                if y[r - margin, c - margin] > 0:
                    patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                    patchesData[patchIndex, :, :, :] = patch
                    patchesLabels[patchIndex] = y[r - margin, c - margin]
                    location[patchIndex] = [r - margin, c - margin]
                    patchIndex += 1
    patchesLabels -= 1
    return patchesData, patchesLabels.astype("int"), location


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    # list_diag是混淆矩阵的对角线元素，是一个列表
    list_diag = np.diag(confusion_matrix)
    # list_raw_sum是混淆矩阵按行相加，是一个列表
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    # each_acc是每一个对角线元素和行和的商
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test, name):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))
