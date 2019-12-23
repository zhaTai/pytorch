# -*-coding:utf-8-*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from matplotlib import pyplot as plt
import csv
import math

from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler

from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, Dataset, DataLoader


def squash(inputs, axis=-1):
    """
    胶囊网络非线性激活函数
    输出向量大小在0到1之间

    :param inputs:
    :param axis: the axis to squash
    :return:
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)  # 二范数
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)  # 防止 norm==0
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.
    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)
        x_hat_detached = x_hat.detach()
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()
        assert self.routings > 0, 'The \'routings\' should be > 0.'

        for i in range(self.routings):
            c = F.softmax(b, dim=1)
            if i == self.routings - 1:
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
            else:
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)

class PrimaryCapsule(nn.Module):

    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv2d(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)


class CapsuleNet(nn.Module):
    def __init__(self, input_size, classes, routings=3):
        super(CapsuleNet, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256,
                               kernel_size=9, stride=1, padding=0)

        self.primarycaps = PrimaryCapsule(in_channels=256, out_channels=256, dim_caps=8,
                                          kernel_size=9, stride=2, padding=0)
        self.digitcaps = DenseCapsule(in_num_caps=32 * 6 * 6, in_dim_caps=8, out_num_caps=classes,
                                      out_dim_caps=16, routings=routings)
        self.decoder = nn.Sequential(
            nn.Linear(16 * classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()
    L_recon = nn.MSELoss()(x_recon.float(), x.float())
    return L_margin + lam_recon * L_recon
    # return L_margin


def test(model, test_loader, classes, lam_recon):
    model.eval()
    test_loss = 0
    correct = 0
    label_pred = []

    for x, y in test_loader:
        y = torch.zeros(y.size(0), classes).scatter_(1, y.view(-1, 1).long(), 1.)
        x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())
        y_pred, x_recon = model(x)
        test_loss += caps_loss(y, y_pred, x, x_recon, lam_recon).item() * x.size(0)
        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        correct += y_pred.eq(y_true).cpu().sum()
        label_pred = np.append(label_pred, y_pred.cpu().detach().numpy())

    test_loss /= len(test_loader.dataset)
    return test_loss, correct.item() / len(test_loader.dataset), label_pred


def train(model, train_loader, test_loader, save_dir, lr, lr_decay, epochs, classes, lam_recon):
    print('Begin Training: ')
    from time import time
    import csv
    logfile = open(save_dir + 'log.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    logwriter.writeheader()
    t0 = time()
    optimizer = Adam(model.parameters(), lr=lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        lr_decay.step()
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            y = torch.zeros(y.size(0), classes).scatter_(1, y.view(-1, 1).long(), 1.)
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            y_pred, x_recon = model(x, y)
            loss = caps_loss(y, y_pred, x, x_recon, lam_recon)
            loss.backward()
            training_loss += loss.item() * x.size(0)
            optimizer.step()

        val_loss, val_acc, label_pred = test(model, test_loader, classes, lam_recon)
        np.savetxt("result\\test_data_pred_" + str(epoch) + ".txt", label_pred)
        if epoch % 5 == 4:
            t_loss, t_acc, t_pred = test(model, train_loader, classes, lam_recon)
            print("\t train_acc = %.5f" % t_acc)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.5f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset), val_loss, val_acc, time() - ti))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir + "\\epoch%d.pkl" % epoch)
            print("\t best val_acc increased to %.5f" % best_val_acc)
    logfile.close()
    torch.save(model.state_dict(), save_dir + "\\trained_model.pkl")
    print("Total time = %ds" % (time() - t0))
    print("End training")
    return model


if __name__ == "__main__":
    import os
    import gzip

    epochs = 5
    batch_size = 100
    lr = 0.001
    lr_decay = 0.9
    lam_recon = 0.0005
    routings = 3
    shift_pixels = 2
    data_dir = "./FashionMnist"
    save_dir = "./result"
    weights = "result\\trained_model.pkl"
    classes = 10

    # train_image_path = "FashionMnist\\train-images-idx3-ubyte.gz"
    # train_label_path = "FashionMnist\\train-labels-idx1-ubyte.gz"
    # test_image_path = "FashionMnist\\t10k-images-idx3-ubyte.gz"
    # test_label_path = "FashionMnist\\t10k-labels-idx1-ubyte.gz"
    #
    # train_label = np.frombuffer(gzip.open(train_label_path, "rb").read(), dtype=np.uint8, offset=8)
    # train_image = np.frombuffer(gzip.open(train_image_path, "rb").read(),
    #                             dtype=np.uint8, offset=16).reshape(len(train_label), 28, 28)
    # train_label = np.array(train_label, dtype=np.int)
    # train_image = np.array(train_image, dtype=np.float) / 255.0
    #
    # test_label = np.frombuffer(gzip.open(test_label_path, "rb").read(), dtype=np.uint8, offset=8)
    # test_image = np.frombuffer(gzip.open(test_image_path, "rb").read(),
    #                            dtype=np.uint8, offset=16).reshape(len(test_label), 28, 28)
    # test_label = np.array(test_label, dtype=np.int)
    # test_image = np.array(test_image, dtype=np.float) / 255.0
    #
    # train_image = train_image[:, np.newaxis, :, :]
    # test_image = test_image[:, np.newaxis, :, :]
    #
    # train_dataset = TensorDataset(torch.from_numpy(train_image),
    #                               torch.from_numpy(train_label))
    # test_dataset = TensorDataset(torch.from_numpy(test_image),
    #                              torch.from_numpy(test_label))
    #
    # transform = transforms.Compose([
    #     # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    #     # transforms.RandomRotation((-45, 45)),  # 随机旋转
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.49, 0.48, 0.44), (0, 229, 0.224, 0.225))  # R, G, B 每层的归一化用到的均值和方差
    # ])
    #
    # train_loader = torch.utils.data.DataLoader(train_image, batch_size=32, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_image, batch_size=32, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./FashionMnist/', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./FashionMnist/', train=False, transform=transforms.Compose([
            transforms.RandomRotation((90, 270), expand=True),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=32, shuffle=False)


    for i, (x, y) in enumerate(test_loader):
        x = x.detach().numpy()
        for j in range(16):
            plt.subplot(4, 4, j+1)
            plt.imshow(x[j-1, 0])
        plt.savefig("roate.jpg", dpi=600)
        plt.show()

        break



    model = CapsuleNet(input_size=[1, 28, 28], classes=classes, routings=routings)
    model.cuda()
    print(model)

    if weights is not None:
        model.load_state_dict(torch.load(weights))
    train(model, train_loader, test_loader, save_dir, lr, lr_decay,
          epochs, classes, lam_recon)

    test_loss, test_acc, label_pred = test(model, test_loader, classes, lam_recon)
    print("test accuracy = %.5f, test_loss = %.5f" % (test_acc, test_loss))
