import torch
from torch import nn
import utils


class SCACD(nn.Module):
    def __init__(self, prior, shift, decoder):
        super().__init__()
        self.prior = prior
        self.shift = shift
        self.decoder = decoder


    def forward(self, x_data, device):
        list_mu, list_cov = self.prior(x_data)
        next_mu, next_cov = self.shift(list_mu, list_cov)
        S = self.decoder(next_mu, next_cov, device)
        return S