import os, time
import argparse
import datetime

import torch.nn
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".95"})

from models.model import FCNN
from functions.dataset import get_dataset
from functions.hparams import HParam
from functions.sub_functions import check_dir, set_seed



def cal_nngp_feature(model, data):
    _, feature = model(data)

    return feature.T

def cal_x_feature(model, data):
    X = data.flatten(1)

    return X.T


def cal_random_fourier(A_b, data):
    A, b = A_b[0], A_b[1]
    X = data.flatten(1)
    D = A.size()[0]
    # cos_x = torch.cos((A * X.unsqueeze(1)).sum(dim=-1))
    # sin_x = torch.sin((A * X.unsqueeze(1)).sum(dim=-1))
    # phi_x = torch.hstack([cos_x, sin_x]).clamp(min=-1, max=1)

    # phi_x = torch.sin(A @ X.T).clamp(min=-1, max=1)
    # phi_x /= torch.sqrt(torch.tensor(D))

    phi_x = torch.cos(A @ X.T + b.unsqueeze(-1)).clamp(min=-1, max=1)
    phi_x *= torch.sqrt(2 / torch.tensor(D))

    return phi_x

def cal_random_fourier_sin(A_b, data):
    A, b = A_b[0], A_b[1]
    X = data.flatten(1)
    D = A.size()[0]
    # cos_x = torch.cos((A * X.unsqueeze(1)).sum(dim=-1))
    # sin_x = torch.sin((A * X.unsqueeze(1)).sum(dim=-1))
    # phi_x = torch.hstack([cos_x, sin_x]).clamp(min=-1, max=1)

    phi_x = torch.sin(A @ X.T).clamp(min=-1, max=1)
    phi_x /= torch.sqrt(torch.tensor(D))

    return phi_x


def random_sample(list_to_sample, num_sample):
    # sample without replacement
    size = list_to_sample.size(0)
    ids = torch.randperm(size)[:num_sample]
    return list_to_sample[ids], ids


def train(config, device, save_path):
    # net_width = config.model.net_width
    # net_depth = config.model.net_depth
    seed = config.model.seed
    batch_size = config.model.batch_size
    IPC = config.distill.IPC
    generate_from_real_data = config.distill.generate_from_real_data

    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'none', 'avgpooling'

    # save_name = config.distill.kernel +  str(net_width) + '_' + str(net_depth) + '_' + str(batch_size) + '_' + str(seed)
    save_name = 'KIP_' + config.distill.kernel + '_IPC' + str(IPC) + '_b' + str(batch_size) + '_seed' + str(seed) +  ('_real' if generate_from_real_data else '')
    print(save_name)
    save_path += save_name + '/'
    check_dir(save_path)

    set_seed(seed)
    # channel, im_size, num_classes, train_data, test_data = get_dataset(config.data.dataset, data_path='./data', device='cuda:0', config=config)
    channel, im_size, num_classes, train_data, test_data, class_names, reverse_trans = get_dataset(config.data.dataset, data_path='./data', device='cuda:0', config=config)
    net = None

    if config.data.zca_whiten:
        zca_trans = config.data.zca_trans
    else:
        zca_trans = None

    d = channel * im_size[0] * im_size[1]
    if config.distill.kernel == 'FCNN':
        cal_feature = cal_nngp_feature
        # net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
        #               net_act=net_act,
        #               net_norm=net_norm, net_pooling=net_pooling, im_size=im_size).to(device)
        net = FCNN(input_size=d, output_size=num_classes, width=d, hidden_layer=config.model.hidden_layer, seed=seed).to(device)
        for param in net.parameters():  # freeze the parameters
            param.requires_grad = False
    elif config.distill.kernel == 'identity':
        cal_feature = cal_x_feature
    elif 'random_fourier' in config.distill.kernel:
        cal_feature = cal_random_fourier
        if 'sin' in config.distill.kernel:
            cal_feature = cal_random_fourier_sin
        d = im_size[0] * im_size[1]
        sigma_2 = 1000
        normal_dis = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([0.]).repeat(d), torch.eye(d)/sigma_2)
        p = 784
        p_sqrt = torch.sqrt(torch.tensor(p))
        A = normal_dis.sample(sample_shape=(p, )).to(device)
        b = torch.rand(size=(p,)).to(device) * torch.pi     # uniform from [0, 2*pi]
    else:
        exit('unknown kernel: %s' % config.distill.kernel)

    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    X_batch, Y_batch = next(iter(train_loader))
    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
    Y_batch_copy = Y_batch
    Y_batch = torch.nn.functional.one_hot(Y_batch, num_classes).float().T

    if 'random_fourier' in config.distill.kernel:
        normalize = False
        if normalize == True:
            # TODO: normalize all dataset
            norm_max = torch.norm(X_batch.flatten(1).float(), dim=-1).max()
            x_norm_bound = np.pi * np.sqrt(sigma_2) / 6 / np.sqrt(d)
            norm_scale = x_norm_bound / norm_max
            # X_batch = X_batch * norm_scale
            sigma_lower_bound = 6 * np.sqrt(d) * norm_max / np.pi
            A = A * norm_scale
        phi_batch = cal_feature((A, b), X_batch)
    else:
        phi_batch = cal_feature(net, X_batch)

    m = IPC * num_classes
    Ys_copy = torch.arange(0, num_classes).long().to(device).repeat(IPC)
    Ys = torch.nn.functional.one_hot(Ys_copy, num_classes).float().T


    if generate_from_real_data:
        temp = []
        for i in range(num_classes):
            # random sample IPC data
            sampled_imgs, _ = random_sample(X_batch[Y_batch_copy == i], IPC)
            temp.append(sampled_imgs)
        Xs_init_list = []
        for i in range(IPC):
            for y in range(num_classes):
                Xs_init_list.append(temp[y][i])

        Xs_init = torch.stack(Xs_init_list)
    else:
        Xs_init = torch.randn(size=(m, X_batch.size()[1:]))

    # minimize the difference between phi and phi_s / Ws and W
    Xs_list = Xs_init.clone().detach().requires_grad_(True)
    lr = config.model.lr
    optimizer = torch.optim.Adam([Xs_list], lr)
    mse_loss = nn.MSELoss(reduction='mean')
    # lr_schedule = [iterations // 3, iterations * 2//3]
    lr_schedule = []
    loss_list = []
    for t in tqdm(range(config.model.iterations)):
        phi = cal_feature(net, Xs_list)
        ntk_s = phi.T @ phi
        lambda_s = 1e-6 * torch.trace(ntk_s) / ntk_s.size()[0]
        ntk_s_inverse_reg = torch.inverse(ntk_s + lambda_s * torch.eye(ntk_s.size()[0], device=device))
        Ys = Y_batch @ torch.linalg.pinv(ntk_s_inverse_reg @ phi.T @ phi_batch)
        Ws_ = Ys @ ntk_s_inverse_reg @ phi.T
        phi_batch_t, batch_ids = random_sample(phi_batch.T, batch_size)
        loss = mse_loss(Ws_ @ phi_batch_t.T, Y_batch[:, batch_ids])
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Xs_list.data.clamp_(min=min_data, max=max_data)
        # if t in lr_schedule:
        #     lr *= 0.1
        #     optimizer = torch.optim.SGD([Xs_list], lr)

    plt.plot(loss_list)
    plt.tight_layout()
    plt.savefig(save_path + '/kip_loss.png', bbox_inches='tight')
    plt.show()

    phi = cal_feature(net, Xs_list)
    # Ws_ = Ys @ torch.inverse(phi.T @ phi) @ phi.T
    ntk_s = phi.T @ phi
    lambda_s = 1e-6 * torch.trace(ntk_s) / ntk_s.size()[0]
    Ws_ = Ys @ torch.inverse(ntk_s + lambda_s * torch.eye(ntk_s.size()[0], device=device)) @ phi.T
    predict_train = Ws_ @ phi_batch
    acc_train = torch.mean((predict_train.argmax(dim=0) == Y_batch_copy).float())
    acc_test = test_kernel_machine(Ws_, net, test_loader, cal_feature)
    print("acc_train: ", acc_train, "acc_test: ", acc_test)
    f = open(save_path + "/acc.txt", "w")
    f.write("acc_train: {}, acc_test: {}".format(acc_train, acc_test))
    f.close()


    ########### plot the distilled data #########
    if channel > 1:
        Xs_list = Xs_list.reshape(-1, channel, im_size[0], im_size[1])

    if config.data.zca_whiten:
        Xs_list = zca_trans.inverse_transform(Xs_list.reshape(-1, channel, im_size[0], im_size[1])).clip(min=0., max=1.)


    if generate_from_real_data:
        if channel > 1:
            Xs_init = Xs_init.reshape(-1, channel, im_size[0], im_size[1])

        if config.data.zca_whiten:
            Xs_init = zca_trans.inverse_transform(Xs_init.reshape(-1, channel, im_size[0], im_size[1])).clip(min=0., max=1.)
        else:
            Xs_init = reverse_trans(Xs_init)
        print('difference between Xs and Xs_init: ', torch.nn.MSELoss()(Xs_init.reshape(Xs_list.size()), Xs_list))

        Xs_plot = Xs_list[:10].detach()
        fig, axs = plt.subplots(2, 10, figsize=(5 * 10, 5 * 2))
        for i in range(Xs_plot.size()[0]):
            if channel > 1:
                axs[0, i].imshow(Xs_init[i].cpu().numpy().transpose(1, 2, 0))
                axs[1, i].imshow(Xs_plot[i].cpu().numpy().transpose(1, 2, 0))
            else:
                axs[0, i].imshow(Xs_init[i].cpu().numpy().reshape(im_size))
                axs[1, i].imshow(Xs_plot[i].cpu().numpy().reshape(im_size))
    else:
        Xs_plot = Xs_list[:10].detach()
        fig, axs = plt.subplots(1, 10, figsize=(5 * 10, 5))
        for i in range(Xs_plot.size()[0]):
            if channel > 1:
                axs[i].imshow(Xs_plot[i].cpu().numpy().transpose(1, 2, 0))
            else:
                axs[i].imshow(Xs_plot[i].cpu().numpy().reshape(im_size))
    for ax in axs.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path + '/distill_imgs.png', bbox_inches='tight')
    # plt.show()
    np.savez(save_path + '/distilled_data.npz', distilled_data=Xs_list.detach().cpu().numpy())

    return acc_test


@torch.no_grad()
def test_kernel_machine(W, net, test_loader, cal_feature):
    correct = torch.tensor(0., device=device)
    for batch_id, (X_batch, Y_batch) in enumerate(test_loader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        phi_batch = cal_feature(net, X_batch)
        predict_batch = W @ phi_batch
        correct += torch.sum((predict_batch.argmax(dim=0) == Y_batch).float())

    N_test = len(test_loader.dataset)
    acc_test = correct.item() / N_test
    return acc_test

@torch.no_grad()
def test_nn(model, test_loader, device, loss_fn):
    # test
    correct = torch.tensor(0., device=device)
    loss_test = torch.tensor(0., device=device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    for i, (X_batch, Y_batch) in enumerate(test_loader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        output = model(X_batch)
        correct += torch.sum((output.argmax(dim=1) == Y_batch).float())
        loss_test += loss_fn(output, Y_batch)

    N_test = len(test_loader.dataset)
    acc_test = correct.item() / N_test
    loss_test = loss_test.item() / N_test
    return acc_test, loss_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="config/ntk.yaml")
    parser.add_argument("--config", type=str, default="config/kip.yaml")
    args, unknown = parser.parse_known_args()
    config = HParam(args.config)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.data.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, config.data.dataset)

    today = datetime.date.today().isoformat()
    save_path = 'exper/' + config.data.dataset + '/' + str(today) + '/'

    # train(config, device, save_path)

    for config.distill.IPC in [1, 10, 50]:
        acc_list, time_list = [], []

        for config.model.seed in range(4):
            time_start = time.time()
            acc_test = train(config, device, save_path)
            time_end = time.time()
            acc_list.append(acc_test * 100)
            time_list.append(time_end - time_start)

        acc_mean = np.mean(acc_list)
        acc_std = np.std(acc_list)
        time_mean = np.mean(time_list)
        f = open(save_path + "/KIP_acc_IPC_{}.txt".format(config.distill.IPC), "w")
        f.write("acc_mean: {}, acc_std: {}, time_mean: {}".format(acc_mean, acc_std, time_mean))
        f.close()