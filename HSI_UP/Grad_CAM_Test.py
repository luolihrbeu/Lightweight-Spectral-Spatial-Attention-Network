import argparse
import cv2
import os
import numpy as np
import torch
from models.presnet import pResNet
import auxil_cam
from hyper_pytorch import HyperData
from utils import *
import time


def load_hyper(args):
    data, labels, numclass = auxil_cam.loadData(args.dataset, num_components=args.components)
    pixels, labels, location = auxil_cam.createImageCubes(data, labels, windowSize=args.spatialsize,
                                                          removeZeroLabels=True)
    bands = pixels.shape[-1]
    numberofclass = len(np.unique(labels))
    x_train, x_test, y_train, y_test = auxil_cam.split_data(pixels, labels, args.tr_percent)
    x_train_loc, x_test_loc, y_train, y_test = auxil_cam.split_data(location, labels, args.tr_percent)
    map_loc = [x_train_loc, x_test_loc, y_train, y_test]
    del pixels, data, labels
    train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"), y_train), None)
    test_hyper = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"), y_test), None)
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=False, drop_last=False,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    return train_loader, test_loader, numberofclass, bands, map_loc


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir, label, predicted, idx):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, "cam_lab-%s_pre-%s_idx-%s.jpg" % (label, predicted, idx))
    # path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    # cv2.imwrite(path_raw_img, np.uint8(255 * img))


def comp_class_vec(ouput_vec, num_class, index=None):
    """
    ???????????????
    :param ouput_vec: tensor
    :param index: int???????????????
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, num_class).scatter_(1, index, 1)
    one_hot.requires_grad = True
    one_hot = one_hot.cuda()
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    ?????????????????????????????????cam
    :param feature_map: np.array??? in [C, H, W]
    :param grads: np.array??? in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
    weights = np.mean(grads, axis=(1, 2))  #
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (110, 110))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--components', default=128, type=int, help='dimensionality reduction')
    # IP:200   UP:103   SV:204   KSC:176
    parser.add_argument('--dataset', default='IP', type=str, help='dataset (options: IP, UP, SV, KSC)')
    parser.add_argument('--tr_percent', default=0.15, type=float, help='samples of train set')
    parser.add_argument('--tr_bsize', default=100, type=int, help='mini-batch train size (default: 100)')
    parser.add_argument('--te_bsize', default=1, type=int, help='mini-batch test size (default: 1000)')
    parser.add_argument('--depth', default=32, type=int, help='depth of the network (default: 32)')
    parser.add_argument('--alpha', default=48, type=int, help='number of new channel increases per depth (default: 12)')
    parser.add_argument('--inplanes', dest='inplanes', default=16, type=int, help='bands before blocks')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                        help='to use basicblock (default: bottleneck)')
    parser.add_argument('--spatialsize', dest='spatialsize', default=11, type=int,
                        help='spatial-spectral patch dimension')
    parser.set_defaults(bottleneck=True)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = True

    path_net = "best_model_%s.pth.tar" % args.dataset
    cam_dir = './cam_result/test_result/cam_img_%s' % args.dataset
    label_dir = './cam_result/test_result/cam_labels_%s' % args.dataset
    if not os.path.exists(cam_dir):
        os.makedirs(cam_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    fmap_block = list()
    grad_block = list()
    # ?????????????????????????????????
    b_channel = np.zeros((110, 110), dtype=np.uint8)
    g_channel = np.zeros((110, 110), dtype=np.uint8)
    r_channel = np.zeros((110, 110), dtype=np.uint8)
    image = cv2.merge((b_channel, g_channel, r_channel))
    # ??????????????????????????????0
    labels = loadData(args.dataset)
    labels = np.pad(labels, ((5, 5), (5, 5)), 'constant')
    # ?????????????????????????????????????????????
    train_loader, test_loader, num_classes, n_bands, location, = load_hyper(args)



    net = pResNet(args.depth, args.alpha, num_classes, n_bands, 2, args.inplanes, bottleneck=args.bottleneck)

    net = net.cuda()
    net.eval()
    checkpoint = torch.load(path_net)
    net.load_state_dict(checkpoint['state_dict'])

    # ??????hook
    net.layer1[0].conv3.register_forward_hook(farward_hook)
    net.layer1[0].conv3.register_backward_hook(backward_hook)
    # ???????????????????????????????????????
    model_flag = 'test'
    if model_flag is 'train':
        center_location = location[0]
    else:
        center_location = location[1]

    for batch_idx, (x, y) in enumerate(test_loader):

        x = x.cuda()
        output = net(x)
        idx = np.argmax(output.cpu().data.numpy(), axis=1)
        net.zero_grad()
        print("????????????%d" % (y + 1), "???????????????%d" % (idx + 1))
        class_loss = comp_class_vec(output, num_classes)
        class_loss.backward()

        if np.array(y) != idx:

            # ??????cam
            grads_val = grad_block[batch_idx].cpu().data.numpy().squeeze()
            fmap = fmap_block[batch_idx].cpu().data.numpy().squeeze()
            cam = gen_cam(fmap, grads_val)

            show_cam_on_image(image, cam, cam_dir, y.numpy() + 1, idx + 1, (batch_idx + 1) )
            # ?????????????????????????????????????????????????????????????????????0????????????5????????????????????????????????????0
            loc_center = (center_location[(batch_idx + 1) - 1] + 5).astype(np.uint)
            # ????????????????????????????????????????????????????????????
            cam_label = labels[loc_center[0] - 5:loc_center[0] + 6, loc_center[1] - 5:loc_center[1] + 6]
            # ??????????????????????????????????????????10???????????????????????????10???
            cam_label = cam_label.repeat(10, axis=0).repeat(10, axis=1)
            # ??????????????????????????????????????????????????????
            map_result(args.dataset, cam_label, label_dir, y.numpy() + 1, idx + 1, (batch_idx + 1))
            # ??????????????????????????????txt???????????????????????????
            # np.savetxt("cam_lab-%s_pre-%s_idx-%s.txt" % (y[i] + 1, idx[i] + 1, str((batch_idx + 1) * (i + 1))),
            #            cam_label,
            #            fmt='%d', delimiter=' ')

