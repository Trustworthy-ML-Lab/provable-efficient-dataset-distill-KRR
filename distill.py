import os, time
import argparse
import datetime

import numpy as np
import torch
import torch.nn
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
    p = A.size()[0]

    phi_x = torch.cos(A @ X.T + b.unsqueeze(-1))
    phi_x *= torch.sqrt(2 / torch.tensor(p))

    return phi_x


def compute_KRR(num_data, dim_data, phi_data, Y_data, lambda_reg):
    if lambda_reg == 0:
        W = Y_data @ torch.linalg.pinv(phi_data)
        # W = torch.linalg.lstsq(phi_data @ phi_data.T, phi_data @ Y_data.T).solution.T
    else:
        if num_data <= dim_data:
            K_data = phi_data.T @ phi_data
            W = Y_data @ torch.inverse(K_data + lambda_reg * torch.eye(K_data.size()[0], device=device)) @ phi_data.T
        else:
            # W = Y_data @ phi_data.T @ torch.inverse(phi_data @ phi_data.T + lambda_reg * torch.eye(phi_data.size()[0], device=device))
            # W (phi^T phi + lambda I) = Y phi^T
            # torch.linalg.lstsq has a smaller numerical error than torch.inverse
            W = torch.linalg.lstsq(phi_data @ phi_data.T + lambda_reg * torch.eye(phi_data.size()[0], device=device), phi_data @ Y_data.T).solution.T

    return W

def random_sample(list_to_sample, num_sample):
    # sample without replacement
    size = list_to_sample.size(0)
    ids = torch.randperm(size)[:num_sample]
    return list_to_sample[ids], ids


def train(config, device, save_path):

    seed = config.model.seed
    IPC = config.distill.IPC
    generate_from_real_data = config.distill.generate_from_real_data
    reg_s = config.distill.reg_s

    save_name = config.distill.kernel + '_IPC' + str(IPC) + '_seed' + str(seed) +  ('_real' if generate_from_real_data else '_noise')
    print(save_name)
    save_path += save_name + '/'
    check_dir(save_path)

    set_seed(seed)
    channel, im_size, num_classes, train_data, test_data, class_names, reverse_trans = get_dataset(config.data.dataset, data_path='./data', device='cuda:0', config=config)
    net = None

    if config.data.zca_whiten:
        zca_trans = config.data.zca_trans
    else:
        zca_trans = None

    ### train eval split
    train_set, val_set = torch.utils.data.random_split(train_data, [0.95, 0.05])
    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    d = channel * im_size[0] * im_size[1]
    if config.distill.kernel == 'FCNN':
        cal_feature = cal_nngp_feature
        net = FCNN(input_size=d, output_size=num_classes, width=d, hidden_layer=config.model.hidden_layer, seed=seed).to(device)
    elif config.distill.kernel == 'identity':
        cal_feature = cal_x_feature
    elif 'random_fourier' in config.distill.kernel:
        cal_feature = cal_random_fourier
        sigma_2 = 1000
        normal_dis = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([0.]).repeat(d), torch.eye(d)/sigma_2)
        p = d
        p_sqrt = torch.sqrt(torch.tensor(p))
        A = normal_dis.sample(sample_shape=(p, )).to(device)
        b = torch.rand(size=(p,)).to(device) * torch.pi     # uniform from [0, 2*pi] or [0, pi]
    else:
        exit('unknown kernel: %s' % config.distill.kernel)

    if net is not None:
        for param in net.parameters():  # freeze the parameters
            param.requires_grad = False



    ### Train the original ridge regression model
    X_batch, Y_batch = next(iter(train_loader))
    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
    Y_batch_copy = Y_batch
    Y_batch = torch.nn.functional.one_hot(Y_batch, num_classes).float().T

    if 'random_fourier' in config.distill.kernel:
        phi_batch = cal_feature((A, b), X_batch)
    else:
        phi_batch = cal_feature(net, X_batch)

    max_acc_val = 0.
    W_best = 0.
    lambda_best = 0.
    lambda_list =  10. ** torch.arange(-6, 6, step=0.5).float()
    if config.distill.kernel == 'identity':     # use larger lambada because of the singularity error
        lambda_list =  10. ** torch.arange(-3, 6, step=0.5).float()
    for lambda_ in lambda_list:
        print("lambda_: ", lambda_)
        # compute the analytical solution of W
        W = compute_KRR(phi_batch.size()[-1], phi_batch.size()[0], phi_batch, Y_batch, lambda_)
        predict_train = W @ phi_batch
        acc_train = torch.mean((predict_train.argmax(dim=0) == Y_batch_copy).float())
        if 'random_fourier' in config.distill.kernel:
            acc_val = test_kernel_machine(W, (A, b), val_loader, cal_feature)
        else:
            acc_val = test_kernel_machine(W, net, val_loader, cal_feature)
        print("acc_train before training: ", acc_train, "acc_val before training: ", acc_val)

        if acc_val > max_acc_val:
            max_acc_val = acc_val
            W_best = W
            lambda_best = lambda_

    # del entk_batch, ntk_inverse
    W = W_best
    print('best accuracy: ', max_acc_val, 'lambda: ', lambda_best)
    if 'random_fourier' in config.distill.kernel:
        acc_test = test_kernel_machine(W, (A, b), test_loader, cal_feature)
    else:
        acc_test = test_kernel_machine(W, net, test_loader, cal_feature)
    print('test accuracy: ', acc_test)

    if reg_s:
        lambda_s = lambda_best
    else:
        lambda_s = 0.

    print('Images/class: ', IPC)
    m = IPC * num_classes
    p = W.size()[-1]
    Ys_copy = torch.arange(0, num_classes).long().to(device).repeat(IPC)
    Ys = torch.nn.functional.one_hot(Ys_copy, num_classes).float().T
    Ys_psdinv = torch.linalg.pinv(Ys)
    # if not reg_s and IPC == 1:
    #     phi_s = torch.linalg.pinv(W) @ Ys
    #     ntk_s = phi_s.T @ phi_s
    #     Ws = Ys @ torch.inverse(ntk_s) @ phi_s.T
    # else:

    # for _ in range(100):  # sample a Z either from real data or from random noise

    if generate_from_real_data: # compute Z such that the distilled data are close to some real data

        temp = []
        for i in range(num_classes):
            # temp.append(X_batch[Y_batch_copy == i][:IPC])
            sampled_imgs, _ = random_sample(X_batch[Y_batch_copy == i], IPC)
            temp.append(sampled_imgs)
        Xs_init = []
        for i in range(IPC):
            for y in range(num_classes):
                Xs_init.append(temp[y][i])
        Xs_init = torch.stack(Xs_init).flatten(1)
        if 'random_fourier' in config.distill.kernel:
            phi_s_init = cal_feature((A, b), Xs_init)
        else:
            phi_s_init = cal_feature(net, Xs_init)

        if reg_s:
            U_hat, S_hat, Vh_hat = torch.linalg.svd(phi_s_init, full_matrices=False)
            S_hat_reg = torch.diag(S_hat / (S_hat ** 2 + lambda_s))
            Ys_init = W @ U_hat @ torch.inverse(S_hat_reg) @ Vh_hat
            if torch.linalg.matrix_rank(Ys_init) == num_classes:
                Ys = Ys_init
                Ys_psdinv = torch.linalg.pinv(Ys)
            Z = (torch.eye(m, device=device) - Ys_psdinv @ Ys) @ (Vh_hat.T @ S_hat_reg @ U_hat.T - Ys_psdinv @ W)
        else:
            if torch.linalg.matrix_rank(W @ phi_s_init) == num_classes:
                Ys = W @ phi_s_init
                Ys_psdinv = torch.linalg.pinv(Ys)
            Z = (torch.eye(m, device=device) - Ys_psdinv @ Ys) @ (torch.linalg.pinv(phi_s_init) - Ys_psdinv @ W)

    else:   # sample Z as random noise
        Z = torch.randn(size=(m, p), device=device) * torch.std(Ys_psdinv @ W) * 0.1
        # Z = torch.randn(size=(m, d), device=device) * torch.std(Ys_psdinv @ W)
    if IPC==1:
        D = Ys_psdinv @ W
    else:
        D = Ys_psdinv @ W + (torch.eye(m, device=device) - Ys_psdinv @ Ys) @ Z

    #     if torch.linalg.matrix_rank(D) == min(m, p) or reg_s:
    #         break
    # else:   # if there is no regularization and we didn't sample a full rank D
    #     print('Rank of D: ', torch.linalg.matrix_rank(D))
    #     raise ValueError('D is singular and there is no regularization')

    if reg_s:
        V, S, Uh = torch.linalg.svd(D, full_matrices=False)
        lambda_s_max = 1. / (4. * S[0] ** 2)
        if lambda_s > lambda_s_max:
            lambda_s = lambda_s_max
        print('lambda_s: ', lambda_s)
        Sigma = S.clone()
        Sigma[S != 0] = (1. + torch.sqrt(1 - 4 * lambda_s * S[S != 0] ** 2)) / (2. * S[S != 0])
        phi_s = Uh.T @ torch.diag(Sigma) @ V.T

    else:
        phi_s = torch.linalg.pinv(D)
        Ws = Ys @ D     # Ws = Ys @ phi_s^+ = Ys @ D

    Ws = compute_KRR(m, p, phi_s, Ys, lambda_s)

    predict_train = Ws @  phi_batch
    acc_train = torch.mean((predict_train.argmax(dim=0) == Y_batch_copy).float())
    if 'random_fourier' in config.distill.kernel:
        acc_test = test_kernel_machine(Ws, (A, b), test_loader, cal_feature)
    else:
        acc_test = test_kernel_machine(Ws, net, test_loader, cal_feature)
    print("acc_train: ", acc_train, "acc_test: ", acc_test)

    cosine = torch.nn.CosineSimilarity(dim=0)(W.flatten(), Ws.flatten())
    print('Cosine Similarity: ', cosine)
    if cosine < 0.99:
        print('Cosine Similarity: ', cosine)
        raise ValueError('does not recover W')


    print('-------------start recovering the distilled dataset from the feature-------------')
    if config.distill.kernel == 'identity':
        Xs_list = phi_s.T
    elif config.distill.kernel == 'FCNN':
        Xs_list = net.inverse(phi_s).T
        phi_s_recover = cal_feature(net, Xs_list)
        phi_error_recover = torch.nn.MSELoss()(phi_s_recover, phi_s)
        print('phi_error_recover: ', phi_error_recover)

        # train and prediction on Xs
        print('--------before learn labels----------')
        Ws_recover = compute_KRR(m, d, phi_s_recover, Ys, lambda_s)
        predict_train = Ws_recover @ phi_batch
        acc_train = torch.mean((predict_train.argmax(dim=0) == Y_batch_copy).float())
        acc_test = test_kernel_machine(Ws_recover, net, test_loader, cal_feature)
        print("acc_train: ", acc_train, "acc_test: ", acc_test)
        cosine = torch.nn.CosineSimilarity(dim=0)(W.flatten(), Ws_recover.flatten())
        print('Cosine Similarity: ', cosine)

        # print('--------after learn labels----------')
        # if m <= d:
        #     Ys_learned = W @ torch.linalg.pinv(torch.inverse(phi_s_recover.T @ phi_s_recover + lambda_s * torch.eye(m, device=device)) @ phi_s_recover.T)
        # else:
        #     Ys_learned = W @ torch.linalg.pinv(phi_s_recover.T @ torch.inverse(phi_s_recover @ phi_s_recover.T + lambda_s * torch.eye(d, device=device)))
        # Ws_recover = compute_KRR(m, d, phi_s_recover, Ys_learned, lambda_s)
        #
        # predict_train = Ws_recover @ phi_batch
        # acc_train = torch.mean((predict_train.argmax(dim=0) == Y_batch_copy).float())
        # acc_test = test_kernel_machine(Ws_recover, net, test_loader, cal_feature)
        # print("acc_train: ", acc_train, "acc_test: ", acc_test)
        # cosine = torch.nn.CosineSimilarity(dim=0)(W.flatten(), Ws_recover.flatten())
        # print('Cosine Similarity: ', cosine)

    elif 'random_fourier' in config.distill.kernel:

        # cos_feature = (phi_s * p_sqrt / np.sqrt(2.)).clamp(min=-1, max=1)
        cos_feature = (phi_s * np.sqrt(p / 2.))
        cos_feature = cos_feature / max(cos_feature.max(), cos_feature.min().abs())
        cos_feature = cos_feature.clamp(min=-1, max=1)
        # Xs = torch.linalg.pinv(A) @ (torch.arccos(cos_feature) - b.unsqueeze(-1))
        Xs = torch.linalg.lstsq(A, torch.arccos(cos_feature) - b.unsqueeze(-1)).solution
        phi_s_recover = cal_feature((A, b), Xs.T)

        phi_error_recover = torch.nn.MSELoss()(phi_s_recover, phi_s)
        print('phi_error_recover: ', phi_error_recover)

        # train and prediction on Xs
        Ws_recover = compute_KRR(m, d, phi_s_recover, Ys, lambda_s)
        predict_train = Ws_recover @ phi_batch
        acc_train = torch.mean((predict_train.argmax(dim=0) == Y_batch_copy).float())
        acc_test = test_kernel_machine(Ws_recover, (A, b), test_loader, cal_feature)
        print("acc_train: ", acc_train, "acc_test: ", acc_test)
        cosine = torch.nn.CosineSimilarity(dim=0)(W.flatten(), Ws_recover.flatten())
        print('Cosine Similarity: ', cosine)

        Xs_list = Xs.T
    else:
        exit('unknown kernel: %s' % config.distill.kernel)


    ########### plot and save the distilled data #########
    if channel > 1:
        Xs_list = Xs_list.reshape(-1, channel, im_size[0], im_size[1])

    if config.data.zca_whiten:  # either use whiten or use transform
        Xs_list = zca_trans.inverse_transform(Xs_list.reshape(-1, channel, im_size[0], im_size[1])).clip(min=0., max=1.)
    else:
        Xs_list = reverse_trans(Xs_list)


    if generate_from_real_data:
        if channel > 1:
            Xs_init = Xs_init.reshape(-1, channel, im_size[0], im_size[1])

        if config.data.zca_whiten:
            Xs_init = zca_trans.inverse_transform(Xs_init.reshape(-1, channel, im_size[0], im_size[1])).clip(min=0., max=1.)
        else:
            Xs_init = reverse_trans(Xs_init)
        print('difference between Xs and Xs_init: ', torch.nn.MSELoss()(Xs_init.reshape(Xs_list.size()), Xs_list))
        np.savez(save_path + '/init_data.npz', init_data=Xs_init.cpu().numpy())

        Xs_plot = Xs_list[:10].detach()
        fig, axs = plt.subplots(2, 10, figsize=(5 * 10, 6 * 2))
        for i in range(Xs_plot.size()[0]):
            if channel > 1:
                axs[0, i].imshow(Xs_init[i].cpu().numpy().transpose(1, 2, 0))
                axs[1, i].imshow(Xs_plot[i].cpu().numpy().transpose(1, 2, 0))
            else:
                axs[0, i].imshow(Xs_init[i].cpu().numpy().reshape(im_size))
                axs[1, i].imshow(Xs_plot[i].cpu().numpy().reshape(im_size))
            axs[0, i].set_title(class_names[i], fontsize=40)
    else:
        Xs_plot = Xs_list[:10].detach()
        fig, axs = plt.subplots(1, 10, figsize=(5 * 10, 6))
        for i in range(Xs_plot.size()[0]):
            if channel > 1:
                axs[i].imshow(Xs_plot[i].cpu().numpy().transpose(1, 2, 0))
            else:
                axs[i].imshow(Xs_plot[i].cpu().numpy().reshape(im_size))
            axs[i].set_title(class_names[i], fontsize=40)
    for ax in axs.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path + '/distill_imgs.png', bbox_inches='tight')
    plt.show()
    np.savez(save_path + '/distilled_data.npz', distilled_data=Xs_list.cpu().numpy())

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/distill.yaml")
    args, unknown = parser.parse_known_args()
    config = HParam(args.config)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.data.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, config.data.dataset)

    today = datetime.date.today().isoformat()
    save_path = 'exper/' + config.data.dataset + '/' + str(today) + '/'

    train(config, device, save_path)
