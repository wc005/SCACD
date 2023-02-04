from torch import nn
import torch
import numpy as np
from torch.distributions import Normal


class Post(nn.Module):
    def __init__(self, input_size, hidden_size, z_size, n_layers, dropout):
        super(Post, self).__init__()
        self.z_size = z_size
        self.hidden_size = hidden_size
        self.rnn_mu = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout).cuda()
        self.rnn_std = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout).cuda()
        self.get_post_mu = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, z_size),
            nn.Softplus()
        ).cuda()
        self.get_post_std = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, z_size),
            nn.Softplus(),
        ).cuda()


    def forward(self, history, future):
        # history: [batch size, sequence len,  z_size]
        # future: [batch size, z_size]
        future = torch.as_tensor(future, dtype=torch.float32).squeeze().cuda()
        # future = torch.tensor(future).squeeze().to(torch.float32).cuda()
        history = torch.as_tensor(history, dtype=torch.float32).squeeze().cuda()
        # history = torch.tensor(history).squeeze().to(torch.float32).cuda()
        post_list = []

        for i in range(history.shape[1]):
            # print("history.shape {},i:{}".format(history.shape, i))
            h = history[:, i, :]
            Z = torch.stack((h, future), dim=1)
            # 计算均值
            outputs, hidden = self.rnn_mu(Z)
            post_mu = self.get_post_mu(outputs[:, -1:].squeeze())
            # 计算方差
            outputs, hidden = self.rnn_std(Z)
            post_std = self.get_post_std(outputs[:, -1:].squeeze())
            post_list.append(Normal(post_mu, post_std))
        return post_list