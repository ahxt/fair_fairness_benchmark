from torch.autograd import Variable, grad
import numpy as np


import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA


class gap_reg(torch.nn.Module):
    def __init__(self, mode="dp"):
        super(gap_reg, self).__init__()
        self.mode = mode

    def forward(self, y_pred, s, y_gt):
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        y_gt = y_gt.reshape(-1)

        if self.mode == "eo":
            y_pred = y_pred[y_gt == 1]
            s = s[y_gt == 1]
        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]
        reg_loss = torch.abs(torch.mean(y0) - torch.mean(y1))
        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


# Loss  PRLoss


class HSIC(torch.nn.Module):  # using linear
    def __init__(self, s_x=1, s_y=1):
        super(HSIC, self).__init__()
        self.s_x = s_x
        self.s_y = s_y

    def forward(self, x, y, y_gt):
        m, _ = x.shape  # batch size
        K = GaussianKernelMatrix(x, self.s_x).cuda()
        L = GaussianKernelMatrix(y, self.s_y).cuda()
        H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
        # H = H.double().cuda()
        H = H.cuda()
        HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
        return HSIC, torch.Tensor([0])


# Loss  PRLoss
class PRLoss(torch.nn.Module):  # using linear
    def __init__(self, eta=1.0):
        super(PRLoss, self).__init__()
        self.eta = eta

    # def forward(self, output_f, output_m):

    def forward(self, y_pred, s, y_gt):
        output_f = y_pred[s == 0]
        output_m = y_pred[s == 1]

        # For the mutual information,
        # Pr[y|s] = sum{(xi,si),si=s} sigma(xi,si) / #D[xs]
        # D[xs]
        N_female = torch.tensor(output_f.shape[0]).to("cuda")
        N_male = torch.tensor(output_m.shape[0]).to("cuda")
        # male sample, #female sample
        Dxisi = torch.stack((N_male, N_female), axis=0)
        # Pr[y|s]
        y_pred_female = torch.sum(output_f)
        y_pred_male = torch.sum(output_m)
        P_ys = torch.stack((y_pred_male, y_pred_female), axis=0) / Dxisi
        # Pr[y]
        P = torch.cat((output_f, output_m), 0)
        P_y = torch.sum(P) / y_pred.shape[0]
        # P(siyi)
        P_s1y1 = torch.log(P_ys[1]) - torch.log(P_y)
        P_s1y0 = torch.log(1 - P_ys[1]) - torch.log(1 - P_y)
        P_s0y1 = torch.log(P_ys[0]) - torch.log(P_y)
        P_s0y0 = torch.log(1 - P_ys[0]) - torch.log(1 - P_y)
        # PI
        PI_s1y1 = output_f * P_s1y1
        PI_s1y0 = (1 - output_f) * P_s1y0
        PI_s0y1 = output_m * P_s0y1
        PI_s0y0 = (1 - output_m) * P_s0y0
        PI = (
            torch.sum(PI_s1y1)
            + torch.sum(PI_s1y0)
            + torch.sum(PI_s0y1)
            + torch.sum(PI_s0y0)
        )
        # PI = self.eta * PI
        return PI, torch.Tensor([0])


def sigma_estimation(X, Y):
    """sigma from median distance"""
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1e-2:
        med = 1e-2
    return med


def distmat(X):
    """distance matrix"""
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def kernelmat(X, sigma):
    """kernel matrix baker"""
    m = int(X.size()[0])
    dim = int(X.size()[1]) * 1.0
    H = torch.eye(m) - (1.0 / m) * torch.ones([m, m])
    Dxx = distmat(X)

    if sigma:
        variance = 2.0 * sigma * sigma * X.size()[1]
        # kernel matrices
        Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)
        # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
    else:
        try:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2.0 * sx * sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError(
                "Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)
                )
            )

    Kxc = torch.mm(Kx, H)

    return Kxc


def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp(-X / (2.0 * sigma * sigma))
    return torch.mean(X)


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    m = int(x.size()[0])
    H = torch.eye(m) - (1.0 / m) * torch.ones([m, m])
    # H = Variable(H)
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx = torch.exp(-Dxx / (2.0 * sigma * sigma))  # kernel matrices
        Ky = torch.exp(-Dyy / (2.0 * sigma * sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x, x)
        sy = sigma_estimation(y, y)
        sxy = sigma_estimation(x, y)
        Kx = torch.exp(-Dxx / (2.0 * sx * sx))
        Ky = torch.exp(-Dyy / (2.0 * sy * sy))
    # Kxc = torch.mm(Kx,H)            # centered kernel matrices
    # Kyc = torch.mm(Ky,H)
    Dxy = distmat(torch.cat([x, y]))
    Dxy = Dxy[: x.size()[0], x.size()[0] :]
    Kxy = torch.exp(-Dxy / (1.0 * sxy * sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2 * torch.mean(Kxy)

    return mmdval


def mmd_pxpy_pxy(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """ """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
    m = int(x.size()[0])

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx = torch.exp(-Dxx / (2.0 * sigma * sigma))  # kernel matrices
        Ky = torch.exp(-Dyy / (2.0 * sigma * sigma))
    else:
        sx = sigma_estimation(x, x)
        sy = sigma_estimation(y, y)
        sxy = sigma_estimation(x, y)
        Kx = torch.exp(-Dxx / (2.0 * sx * sx))
        Ky = torch.exp(-Dyy / (2.0 * sy * sy))
    A = torch.mean(Kx * Ky)
    B = torch.mean(torch.mean(Kx, dim=0) * torch.mean(Ky, dim=0))
    C = torch.mean(Kx) * torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2 * B + C
    return mmd_pxpy_pxy_val


def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """ """
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy


def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=True):
    """ """
    m = int(x.size()[0])
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    thehsic = Pxy / (Px * Py)
    return thehsic


def hsic_normalized_cca(x, y, sigma, use_cuda=True, to_numpy=True):
    """ """
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma)

    epsilon = 1e-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = Kxc.mm(Kxc_i)
    Ry = Kyc.mm(Kyc_i)
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy
