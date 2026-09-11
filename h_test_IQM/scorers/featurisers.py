'''
Featurisers for exp8's ladder -- vector scores that know progressively more about images.

Each maps an image batch to a feature vector per image, with the same (im_size, device) call and
preprocessing as every other torch scorer (base_scorer_torch: [0, 1] input, channel-first,
resized to im_size). Where a featuriser is random, the seed fixes it, built under a forked RNG
so it never disturbs anyone else's torch random state.

    colour_moments_model    per-channel RGB mean and std              6-D   knows nothing
    random_pixels_model     seeded Gaussian projection of raw pixels  64-D  no convolution
    resnet18_model          global-average-pooled ResNet-18 features  512-D untrained or ImageNet
'''
import torch
import torch.nn as nn

from h_test_IQM.scorers.torch_scorers import base_scorer_torch

# the statistics torchvision's ImageNet weights were trained with
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class colour_moments_model(base_scorer_torch):
    '''(mean_R, mean_G, mean_B, std_R, std_G, std_B) over each image's pixels'''

    def setup(self):
        pass

    def get_score(self, x):
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        return torch.cat([flat.mean(dim=2), flat.std(dim=2)], dim=1).cpu().numpy()


class random_pixels_model(base_scorer_torch):
    '''
    A fixed random linear map from the raw pixels to `dim` features: no convolution, no
    locality, no nonlinearity. Separates "a random projection" from "a random CONVOLUTIONAL
    projection" -- the untrained AE encoder is the latter.
    '''

    def setup(self, dim=64, seed=0):
        n_in = 3 * self.im_size[0] * self.im_size[1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            w = torch.randn(n_in, dim) / n_in ** 0.5
        self.w = w.to(self.device)

    def get_score(self, x):
        return (x.reshape(x.shape[0], -1) @ self.w).cpu().numpy()


class resnet18_model(base_scorer_torch):
    '''
    ResNet-18 up to and including global average pooling -> 512-D. pretrained=True loads the
    torchvision ImageNet weights; pretrained=False keeps the seeded random initialisation, the
    same null exp7 used for the autoencoder, now on a different architecture.
    '''

    def setup(self, pretrained=True, seed=0):
        from torchvision.models import resnet18, ResNet18_Weights
        if pretrained:
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                net = resnet18(weights=None)
        net.fc = nn.Identity()
        self.net = net.eval().to(self.device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def get_score(self, x):
        with torch.no_grad():
            return self.net((x - self.mean) / self.std).cpu().numpy()
