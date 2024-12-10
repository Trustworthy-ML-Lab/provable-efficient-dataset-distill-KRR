# adapted from
# https://github.com/GeorgeCazenavette/mtt-distillation/blob/main/utils.py
# https://github.com/VICO-UoE/DatasetCondensation

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import kornia as K
import tqdm
from functions.sub_functions import set_seed

def get_dataset(dataset, data_path='./data', device='cuda:0', config=None):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

        # return channel, im_size, num_classes, dst_train, dst_test

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if config.data.zca_whiten:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes


    elif dataset.startswith('CIFAR100'):
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if config.data.zca_whiten:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}

    else:
        exit('unknown dataset: %s' % dataset)

    if config.data.zca_whiten:
        # train_loader = DataLoader(dst_train, batch_size=len(dst_train))
        # images, labels = next(iter(train_loader))
        # images = images.to(device)
        images = []
        labels = []
        print("Train ZCA")
        for i in tqdm.tqdm(range(len(dst_train))):
            im, lab = dst_train[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")
        zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        zca.fit(images)
        zca_images = zca(images).to("cpu")
        dst_train = TensorDataset(zca_images, labels)

        images = []
        labels = []
        print("Test ZCA")
        for i in tqdm.tqdm(range(len(dst_test))):
            im, lab = dst_test[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")

        zca_images = zca(images).to("cpu")
        dst_test = TensorDataset(zca_images, labels)

        config.data.zca_trans = zca

        reverse_trans = None
    else:
        def reverse_trans(x):
            if dataset == 'MNIST':
                x = x * torch.tensor(std).cuda() + torch.tensor(mean).cuda()
            else:
                x = x * torch.tensor(std).cuda().view(-1, 1, 1) + torch.tensor(mean).cuda().view(-1, 1, 1)
            return x

    return channel, im_size, num_classes, dst_train, dst_test, class_names, reverse_trans



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


class FullMnist(Dataset):
    def __init__(self, train=True):
        mean = (0.1307,)
        std = (0.3081,)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.mnist = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    def __getitem__(self, index):
        data, target = self.mnist[index]

        return index, data, target

    def __len__(self):
        return len(self.mnist)


class FullCifar10(Dataset):
    def __init__(self, train=True):
        mean = (0.49139968, 0.48215827 ,0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)


    def __getitem__(self, index):
        data, target = self.CIFAR10[index]

        return index, data, target

    def __len__(self):
        return len(self.CIFAR10)


def get_zca_matrix(X, reg_coef=0.1):
    X_flat = X.reshape(X.shape[0], -1)
    cov = (X_flat.T @ X_flat) / X_flat.shape[0]
    reg_amount = reg_coef * torch.trace(cov) / cov.shape[0]
    u, s, _ = torch.svd(cov.cuda() + reg_amount * torch.eye(cov.shape[0]).cuda())
    inv_sqrt_zca_eigs = s ** (-0.5)
    whitening_transform = torch.einsum(
        'ij,j,kj->ik', u, inv_sqrt_zca_eigs, u)

    return whitening_transform.cpu()


def transform_data(X, whitening_transform):
    if len(whitening_transform.shape) == 2:
        X_flat = X.reshape(X.shape[0], -1)
        X_flat = X_flat @ whitening_transform
        return X_flat.view(*X.shape)
    else:
        X_flat = X.reshape(X.shape[0], -1)
        X_flat = torch.einsum('nd, ndi->ni', X_flat, whitening_transform)
        return X_flat.view(*X.shape)
