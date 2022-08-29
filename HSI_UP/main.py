import argparse
import auxil
from hyper_pytorch import *
import torch
import torch.nn as nn
import torch.nn.parallel
from models.MobileNetV3 import MobileNetV3
from util.focall_loss2 import FocalLoss2
from util.decode_map import map_result
import time
import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def load_hyper(args):
    data, labels, num_class = auxil.loadData(args.dataset, num_components=args.components)
    pixels, labels, patchLocation = auxil.createImageCubes1(
        data, labels, windowSize=args.spatialsize, removeZeroLabels=True)
    bands = pixels.shape[-1]
    map_shape = data.shape[0:2]
    x_train, x_val, x_test, y_train, y_val, y_test = auxil.split_data(pixels, labels, args.tr_percent)
    train_loc, test_loc, train_loc_y, test_loc_y = auxil.split_data(patchLocation, labels, args.tr_percent)
    map_loc = [train_loc, test_loc, train_loc_y, test_loc_y]
    # print(np.unique(y_train, return_counts=True)[1])
    # print(np.unique(y_val, return_counts=True)[1])
    # print(np.unique(y_test, return_counts=True)[1])
    del pixels, labels
    train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"), y_train), None)
    val_hyper = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"), y_val), None)
    test_hyper = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"), y_test), None)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_hyper, batch_size=args.tr_bsize, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, num_class, bands, map_loc, map_shape


def train(trainloader, model, criterion, optimizer, use_cuda):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses[batch_idx] = loss.item()
        accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (np.average(losses), np.average(accs))


def tet(testloader, model, criterion):
    # model.eval()
    with torch.no_grad():
        accs = np.ones((len(testloader))) * -1000.0
        losses = np.ones((len(testloader))) * -1000.0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda())
            outputs = model(inputs)
            losses[batch_idx] = criterion(outputs, targets).item()
            accs[batch_idx] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))


def predict(testloader, model):
    model.eval()
    predicted = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        [predicted.append(a) for a in model(inputs).data.cpu().numpy()]
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 100)) * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--flag', default="train", type=str, help='train or test')
    parser.add_argument('--test_times', default=1, type=int, help='Number of tests')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, help='initial learning rate')
    # IP:200   UP:103   SV:204   KSC:176
    parser.add_argument('--components', default=16, type=int, help='dimensionality reduction')
    parser.add_argument('--dataset', default='UP', type=str, help='dataset (options: IP, UP, SV, KSC)')
    parser.add_argument('--tr_percent', default=0.01, type=float, help='samples of train set')
    parser.add_argument('--tr_bsize', default=128, type=int, help='mini-batch train size (default: 100)')
    parser.add_argument('--te_bsize', default=128, type=int, help='mini-batch test size (default: 1000)')
    parser.add_argument('--spatialsize', dest='spatialsize', default=9, type=int,
                        help='spatial-spectral patch dimension')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    args = parser.parse_args()

    train_loader, val_loader, test_loader, num_classes, n_bands, decode_map, map_shape = load_hyper(args)
    # Use CUDA
    model = MobileNetV3(n_class=num_classes, num_bands=n_bands)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    criterion = FocalLoss2(gamma=1)
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=args.weight_decay)

    title = 'HYPER-' + args.dataset
    if args.flag is "test":
        print("Start Testing......")
        checkpoint = torch.load("imp_temp_best_model_UP_1.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        n_result = np.zeros((args.test_times, num_classes + 3))
        map = np.zeros(map_shape)
        for i in range(args.test_times):
            print("第 %d 次测试......" % (i + 1))
            predicted = np.argmax(predict(test_loader, model), axis=1)
            classification, confusion, results = auxil.reports(
                predicted, np.array(test_loader.dataset.__labels__()), args.dataset)
            decode_map[3] = predicted
            for idx in range(len(decode_map[0])):
                map[int(decode_map[0][idx][0])][int(decode_map[0][idx][1])] = decode_map[2][idx] + 1
            for idx in range(len(decode_map[1])):
                map[int(decode_map[1][idx][0])][int(decode_map[1][idx][1])] = decode_map[3][idx] + 1
            map_result(map)
            # print(results)
            results = np.array(results)
            n_result[i] = results
            if i < (args.test_times - 1):
                train_loader, val_loader, test_loader, num_classes, n_bands, decode_map, map_shape = load_hyper(args)
        print(n_result)
        avg = np.mean(n_result, axis=0)
        var = np.std(n_result, axis=0)
        np.set_printoptions(precision=2, suppress=True)
        print("测试平均值为:", avg)
        print("测试标准差为:", var)

    elif args.flag is "train":
        best_acc = -1
        total_train_time = []
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args)
            start_time = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, use_cuda)
            test_loss, test_acc = tet(val_loader, model, criterion)
            end_time = time.time()
            train_time = end_time - start_time
            total_train_time.append(train_time)
            print("EPOCH", epoch, "|\tTRAIN LOSS: %.4f\t" % train_loss, "TRAIN ACCURACY: %.4f\t" % train_acc, end='')
            print("VAl LOSS: %.4f\t" % test_loss, "VAL ACCURACY: %.4f\t" % test_acc,
                  "TIME: %.2f" % train_time)
            # save model
            if test_acc > best_acc and test_acc >= 90:
                state = {
                    # 'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    # 'acc': test_acc,
                    # 'best_acc': best_acc,
                    # 'optimizer': optimizer.state_dict(),
                }
                torch.save(state, "imp_temp_best_model_UP_1.pth.tar")
                best_acc = test_acc
        time1 = time.time()
        pred = np.argmax(predict(test_loader, model), axis=1)
        time2 = time.time()
        label = np.array(test_loader.dataset.__labels__())
        classification, confusion, results = auxil.reports(pred, label, args.dataset)
        print(title, results)
        print(np.array(total_train_time).sum())
        print(time2-time1)
        # ---------------------------------------------------------------------------
    checkpoint = torch.load("imp_temp_best_model_UP_1.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    time1 = time.time()
    pred = np.argmax(predict(test_loader, model), axis=1)
    time2 = time.time()
    label = np.array(test_loader.dataset.__labels__())
    classification, confusion, results = auxil.reports(pred, label, args.dataset)
    print(time2 - time1)

if __name__ == '__main__':
    main()
