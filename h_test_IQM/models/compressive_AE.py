import torch
from torch import nn
import torch.nn.functional as F

from h_test_IQM.models.gdn import GDN


# GDN from https://interdigitalinc.github.io/CompressAI/_modules/compressai/layers/gdn.html
class EntropyLimitedModel(nn.Module):
    """Autoencoder that quantises to centers and can be
    optimised for only distortion term, with theoretical limit
    on the entropy."""
    def __init__(self, sigma=1, N=128, M=64, sigmoid=False, centers=None):
        super().__init__()
        self.entropy_bottleneck = None
        self.sigma = sigma
        if centers == 1:
            cent = torch.Tensor([1])
        elif centers == 2:
            cent = torch.Tensor([-1, 1])
        elif centers == 5:
            cent = torch.Tensor([-2, -1, 0, 1, 2])
        else:
            cent = None
        self.register_buffer('centers', cent)
        #self.centers = centers
        padding = 3 // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, N, kernel_size=3, stride=2, padding=padding),
            GDN(N),
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=padding),
            GDN(N),
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=padding),
            GDN(N),
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=padding),
            GDN(N),
            nn.Conv2d(N, M, kernel_size=3, stride=1, padding=padding))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(M, N, kernel_size=3, stride=1, padding=padding),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=3, stride=2, padding=padding,
                              output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=3, stride=2, padding=padding,
                              output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=3, stride=2, padding=padding,
                              output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, kernel_size=3, stride=2, padding=padding,
                              output_padding=1))

        if sigmoid:
            self.decoder.add_module('sig', nn.Sigmoid())

    def encode(self, x):
        y = self.encoder(x)
        return y

    def quantise(self, y):
        if self.centers is None:
            return y
        y_flat = y.reshape(y.size(0), y.size(1), y.size(2)*y.size(3), 1)
        dist = torch.abs((y_flat - self.centers))**2
        # if self.train:
        if self.training:
            phi = F.softmax(-self.sigma * dist, dim=-1)
        else:
            phi = F.softmax(-1e7 * dist, dim=-1)
            symbols_hard = torch.argmax(phi, axis=-1)
            phi = F.one_hot(symbols_hard, num_classes=self.centers.size(0))
        inner_product = phi * self.centers
        y_hat = torch.sum(inner_product, axis=-1)
        y_hat = y_hat.reshape(y.shape)
        return y_hat

    def decode(self, y):
        x = self.decoder(y)
        return x

    def forward(self, x):
        y = self.encode(x)
        y_hat = self.quantise(y)
        x_hat = self.decode(y_hat)

        return {
            'x_hat': x_hat,
            'likelihoods': {
                'y': torch.zeros_like(y),
            }
        }
