'''
generate noise on a unit sphere for 'barely visible' noise
'''
from abc import ABC, abstractmethod
import numpy as np
import torch
import warnings

# Suppress the specific DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message="__array_wrap__ must accept context and return_scalar arguments")


class base_noise(ABC):
    def __init__(self, acceptable_percent=0.9, max_iter=50, reject_low_noise=False, **kwargs):
        self.setup_args(**kwargs)
        self.acceptable_percent = acceptable_percent
        self.max_iter = max_iter
        self.reject_low_noise = reject_low_noise
        self.num_rejected = 0

    def setup_args(self, **kwargs):
        pass

    @abstractmethod
    def _make_noisey_image(self, img, lib=np):
        # uses any args on self for parameters
        # must return x_noisey, additive_noise
        pass

    def __call__(self, img, epsilon=None):
        '''
        will return a noisy image with noise level epsilon
        None will be returned if noise level is too low acccording to acceptable_percent
        '''
        if not isinstance(img, np.ndarray):
            # torch batch
            device = img.device
            img = img.cpu()
            if len(img.shape) == 4:
                noisy_imgs = []
                for i in range(img.shape[0]):
                    noisy_img = self._call_single(img[i], lib=torch)
                    if noisy_img is not None:
                        noisy_imgs.append(noisy_img)
                if len(noisy_imgs) == 0:
                    return None

                return torch.stack(noisy_imgs).to(device)
            else:
                # torch single
                noisey = self._call_single(img, lib=torch)
                if noisey is None:
                    return None
                else:
                    return noisey.to(device)
        else:
            # numpy array
            return self._call_single(img, lib=np)

    def _call_single(self, img, lib=np):
        for _ in range(self.max_iter):
            x_noisey, additive_noise = self._make_noisey_image(img, lib=lib)
            # check noise level and actual noise level reduced after clipping
            expected_noise = np.sqrt(np.mean(np.square(additive_noise)))
            if self.reject_low_noise == False:
                return x_noisey
            else:
                actual_noise = lib.sqrt(lib.mean(lib.square(x_noisey - img)))
                # option to reject and redo the noise if too low (recursion)
                if actual_noise >= self.acceptable_percent * expected_noise:
                    return x_noisey
                else:
                    self.num_rejected += 1
                    return None
                

class epsilon_noise(base_noise):
    def setup_args(self, epsilon=1):
        self.epsilon = epsilon

    def _make_noisey_image(self, img, lib=np):
        noise = np.random.randn(*img.shape)
        noise_norm = noise / np.linalg.norm(noise.reshape(-1))
        additive_noise = self.epsilon * noise_norm
        if lib == torch:
            unclipped = img + torch.Tensor(additive_noise)
        else:
            unclipped = img + additive_noise
        x_noisey = lib.clip(unclipped, 0, 1)
        return x_noisey, additive_noise


class Gaussian_noise(base_noise):
    def setup_args(self, std=0.1):
        self.std = std

    def _make_noisey_image(self, img, lib=np):
        additive_noise = np.random.normal(
            loc=0, scale=self.std, size=img.shape)
        if lib == torch:
            unclipped = img + torch.Tensor(additive_noise)
        else:
            unclipped = img + additive_noise
        x_noisey = lib.clip(unclipped, 0, 1)
        return x_noisey, additive_noise


if __name__ == '__main__':
    # test for both numpy and torch images
    import torch
    import matplotlib.pyplot as plt
    noise = epsilon_noise()

    # numpy
    img = np.random.rand(32, 32, 3)
    noisy_img = noise(img)
    print(noisy_img.shape)
    # now plot the image
    # plt.imshow(noisy_img)
    # plt.show()

    # torch single
    img = torch.rand(3, 32, 32)
    noisy_img = noise(img)
    print(noisy_img.shape)

    # now plot the image
    # convert torch to numpy for plotting 
    noisy_img = noisy_img.permute(1, 2, 0).numpy()
    # plt.imshow(noisy_img)
    # plt.show()

    # torch batch 
    img = torch.rand(5, 3, 32, 32)
    noisy_img = noise(img)
    print(noisy_img.shape)
    # noisy_img = noisy_img[0].permute(1, 2, 0).numpy()