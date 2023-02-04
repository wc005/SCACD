from torch import nn
from torch.distributions import Normal
import utils
import torch
import tools


class Decoder(nn.Module):
    def __init__(self, z_size, hidden_size, z_samples, s_samples, window):
        super().__init__()
        self.hid_dim = hidden_size
        self.z_samples = z_samples
        self.s_samples = s_samples
        self.z_size = z_size
        self.window = window

        # 计算 S 均值
        self.dec_mean = nn.Sequential(
            nn.BatchNorm1d(z_size),
            nn.Linear(z_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, window),
        ).cuda()
        # 计算 S 方差
        self.dec_std = nn.Sequential(
            nn.BatchNorm1d(z_size),
            nn.Linear(z_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, window),
        ).cuda()
        dim = 1
        self.self_att = utils.SelfAttention(dim_in=dim, dim_k=dim, dim_v=dim).cuda()
        self.pe = utils.PositionalEncoding(d_model=1, dropout=0, max_len=window)
        self.gen = nn.Sequential(
            nn.Linear(window, window),
            nn.Linear(window, window),
            nn.GELU(),
            nn.Linear(window, window),
            nn.GELU(),
            nn.Linear(window, window),

        ).cuda()


    def generat_z(self, mu_n, cov_n, device):
        '''
        生成 隐变量
        :param mu_n:
        :param cov_n:
        :param device:
        :return:
        '''
        mu = mu_n.squeeze()
        # cov * cov^T
        cov_n = cov_n.squeeze()
        cov_n_t = cov_n.permute(0, 2, 1)
        # 生成预测隐变量 Z 的分布
        cov_d = torch.bmm(cov_n, cov_n_t)
        cov_d.add_(torch.eye(self.z_size).cuda())
        ch_cov = torch.linalg.cholesky(cov_d, upper=False)
        # 重参数采样获取 Z
        Z = 0
        for i in range(self.z_samples):
            Z = Z + utils.Z_Repasampling(mu, ch_cov, device)
        return Z / self.z_samples


    def generat_s(self, Z, device):
        '''
        生成 序列
        :param Z:
        :param device:
        :return:
        '''
        mu = self.dec_mean(Z)
        # 首先特征映射
        # E = self.dec_std(Z).unsqueeze(2)
        E = (self.dec_std(Z) - mu).unsqueeze(2)

        E_T = E.permute(0, 2, 1)
        cov_s = torch.bmm(E, E_T)
        # 保证协方差矩阵正定
        cov_s.add_(torch.eye(self.window).cuda())
        # cholesky 分解
        ch_cov = torch.linalg.cholesky(cov_s, upper=False)
        # 重参数采样获取 Z
        S = 0
        for i in range(self.s_samples):
            S = S + utils.S_Repasampling(mu, ch_cov, device)
        set = S / self.s_samples
        set = set.unsqueeze(2)
        result = self.self_att(set).squeeze()
        return result


    def forward(self, mu_n, cov_n, device):
        # 生成隐变量 Z
        Z = self.generat_z(mu_n, cov_n, device)
        # 生成序列 S
        S = self.generat_s(Z, device)

        # 位置编码
        # position_emd = self.pe.pe[:, :].cuda()
        last_x = S.unsqueeze(1).permute(0, 2, 1)
        # input = (last_x + position_emd * torch.mean(last_x)).permute(0, 2, 1)
        att = self.self_att(last_x).squeeze()
        result = self.gen(att)
        return result

