from torch import nn
import torch


class CausaConv(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.conv_mu = nn.Conv1d(in_channels=input_size,
                                 out_channels=input_size,
                                 kernel_size=2,
                                 stride=1,
                                 bias=True).cuda()
        self.conv_cov = nn.Conv3d(in_channels=1,
                                  out_channels=1,
                                  kernel_size=(2, 1, 1),
                                  stride=1,
                                  padding=0,
                                  dilation=1,
                                  groups=1,
                                  bias=True).cuda()


    def forward(self, list_mu, list_cov):
        mu = torch.stack(list_mu).permute(1, 2, 0).to(torch.float32)
        # cov: [batch, channels, length, width, hight]
        cov = torch.stack(list_cov).permute(1, 0, 2, 3).unsqueeze(1)
        mu_n = self.conv_mu(mu)
        # s: [num_layers, batch size, hid dim]
        cov_n = self.conv_cov(cov)
        return mu_n, cov_n

