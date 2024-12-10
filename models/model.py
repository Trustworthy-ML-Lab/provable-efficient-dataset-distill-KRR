import random
import numpy as np
import torch
from torch import nn


def _initialize_weights(model, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    for i, m in enumerate(model):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 1.0 / torch.sqrt(torch.tensor(m.weight.shape[-1])))


def make_layers(hidden_layer=1, width=512, input_size=3 * 32 * 32, output_size=10, linear=False):
    if linear:
        layers = [nn.Linear(input_size, width, bias=False)]
    else:
        layers = [nn.Linear(input_size, width, bias=False), nn.LeakyReLU(negative_slope=0.1)]

    if hidden_layer >= 2:
        for i in range(hidden_layer-1):
            if linear:
                layers += [nn.Linear(width, width, bias=False)]
            else:
                layers += [nn.Linear(width, width, bias=False), nn.LeakyReLU(negative_slope=0.1)]

    layers += [nn.Linear(width, output_size, bias=False)]
    return nn.Sequential(*layers)


class FCNN(nn.Module):
    def __init__(self, input_size=28 * 28, output_size=1, width=100, hidden_layer=1, linear=False, seed=0):
        super(FCNN, self).__init__()
        self.classifier = make_layers(hidden_layer=hidden_layer, width=width, input_size=input_size, output_size=output_size, linear=linear)
        _initialize_weights(self.classifier, seed=seed)
        self.inverse_LeakyReLU = nn.LeakyReLU(negative_slope=1 / 0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        for i, m in enumerate(self.classifier):
            if i == len(self.classifier)-2:
                x = m(x).flatten(1)
                feature = x
            else:
                x = m(x)
        return x, feature


    def inverse(self, x):
        for i, m in reversed(list(enumerate(self.classifier[:-1]))):
            if isinstance(m, nn.Linear):
                # solving using torch.linalg.pinv has large error, use torch.linalg.lstsq instead
                # x = torch.linalg.pinv(m.weight) @ x
                x = torch.linalg.lstsq(m.weight, x).solution
            else:
                # inverse_LeakyReLU = nn.LeakyReLU(negative_slope=1 / m.negative_slope)
                x = self.inverse_LeakyReLU(x)

        return x

