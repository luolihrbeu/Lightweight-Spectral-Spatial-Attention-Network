import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv


def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def split_data(pixels, labels, percent, splitdset="custom", rand_state=69):
    # splitdset = "sklearn"
    if splitdset == "sklearn":
        return train_test_split(pixels, labels, test_size=(1 - percent), stratify=labels, random_state=rand_state)
    elif splitdset == "custom":
        if (len(pixels.shape) > 3):
            pixels_number = np.unique(labels, return_counts=True)[1]

            # train_set_size = []
            # for num in pixels_number:
            #     if int(np.ceil(num * percent)) < 3:
            #         train_set_size.append(3)
            #     else:
            #         train_set_size.append(int(np.ceil(num * percent)))

            train_set_size = [int(np.floor(a * percent)) for a in pixels_number]
            val_set_size = train_set_size
            tr_size = int(sum(train_set_size))
            val_size = int(sum(val_set_size))
            assert (train_set_size == val_set_size) and (tr_size == val_size)
            te_size = int(sum(pixels_number)) - int(2 * sum(train_set_size))

            sizetr = np.array([tr_size] + list(pixels.shape)[1:])
            sizeval = np.array([val_size] + list(pixels.shape)[1:])
            sizete = np.array([te_size] + list(pixels.shape)[1:])

            train_x = np.empty((sizetr))
            train_y = np.empty((tr_size))
            val_x = np.empty((sizeval))
            val_y = np.empty((val_size))
            test_x = np.empty((sizete))
            test_y = np.empty((te_size))

            trcont = 0
            valcont = 0
            tecont = 0
            for cl in np.unique(labels):
                pixels_cl = pixels[labels == cl]
                labels_cl = labels[labels == cl]
                pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
                for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
                    if cont < train_set_size[cl]:
                        train_x[trcont, :, :, :] = a
                        train_y[trcont] = b
                        trcont += 1
                    elif (train_set_size[cl] <= cont) and (cont < 2 * train_set_size[cl]):
                        val_x[valcont, :, :, :] = a
                        val_y[valcont] = b
                        valcont += 1
                    else:
                        test_x[tecont, :, :, :] = a
                        test_y[tecont] = b
                        tecont += 1
            train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
            val_x, val_y = random_unison(val_x, val_y, rstate=rand_state)
            test_x, test_y = random_unison(test_x, test_y, rstate=rand_state)
            return train_x, val_x, test_x, train_y, val_y, test_y
        else:
            # pixels_number:各类标签的个数，按顺序记录在列表中。
            pixels_number = np.unique(labels, return_counts=True)[1]
            # 设置各类地物训练集的大小
            # train_set_size = np.ones(len(pixels_number)) * percent
            train_set_size = [int(np.floor(a * percent)) * 2 for a in pixels_number]
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


def loadData(name, num_components=None):
    data_path = os.path.join(os.getcwd(), 'data')
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
    # zeroPaddedX1 = padWithZeros(X, margin=margin)
    # print(np.array_equal(zeroPaddedX, zeroPaddedX1))

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


def createImageCubes1(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode="constant")
    # split patches
    if removeZeroLabels:
        non_zero_label_num = (y > 0).sum()
        patchesData = np.zeros((non_zero_label_num, windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((non_zero_label_num))
        patchLocation = np.zeros((non_zero_label_num, 2))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                if y[r - margin, c - margin] > 0:
                    patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                    patchesData[patchIndex, :, :, :] = patch
                    patchesLabels[patchIndex] = y[r - margin, c - margin]
                    patchLocation[patchIndex] = [r - margin, c - margin]
                    patchIndex += 1
        patchesLabels -= 1
        return patchesData, patchesLabels.astype("int"), patchLocation
    else:
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        patchLocation = np.zeros(((X.shape[0] * X.shape[1]), 2))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                # import matplotlib.pyplot as plt
                # plt.imshow(patch[:, :, 100])
                # plt.show()
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r - margin, c - margin]
                patchLocation[patchIndex] = [r - margin, c - margin]
                patchIndex = patchIndex + 1
        return patchesData, patchesLabels.astype("int"), patchLocation


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
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
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
