import numpy as np
from torch import nn
import torch
from torch.distributions import Normal
import scipy
import tools

class autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, z_size):
        super(autoencoder, self).__init__()
        self.z_size = z_size
        self.prior_encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, z_size),
            nn.Softplus(),
        ).cuda()
        # 求均值
        self.prior_encoder_mu = nn.Sequential(
            nn.BatchNorm1d(z_size),
            nn.Linear(z_size, z_size),
            nn.GELU(),
            nn.Linear(z_size, z_size),
            nn.Softplus(),
        ).cuda()
        # 求方差
        self.prior_encoder_sig = nn.Sequential(
            nn.BatchNorm1d(z_size),
            nn.Linear(z_size, z_size),
            nn.Softplus(),
        ).cuda()
        self.LN = nn.Sequential(
            nn.BatchNorm1d(z_size),
            nn.Linear(z_size, z_size),
            nn.Softplus(),
        ).cuda()

    def forward(self, x):
        # 计算历史滑动窗口内隐变量分布
        list = []
        list_mu = []
        list_cov = []
        for i in range(0, x.shape[1]):
            X = x[:, i, :]
            ebd = self.prior_encoder(X)
            # 求均值
            mu = self.prior_encoder_mu(ebd)
            # 求协方差矩阵 E * E^t
            E = self.LN(self.prior_encoder_sig(ebd) - mu).unsqueeze(2)
            E_T = E.permute(0, 2, 1)

            cov = torch.bmm(E, E_T)
            # cov.add_(torch.eye(self.z_size).cuda())
            # ch_cov = torch.linalg.cholesky(cov, upper=True)
            list_mu.append(mu)
            list_cov.append(cov)
        # ( batch_size, seq_len, emding_size)
        return list_mu, list_cov
