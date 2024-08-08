'''
generate noise on a unit sphere for 'barely visible' noise
'''

import numpy as np


class epislon_noise:
    def __init__(self, epsilon=1, acceptable_percent=0.9, max_iter=10):
        self.epsilon = epsilon
        self.acceptable_percent = acceptable_percent
        self.max_iter = max_iter

    @staticmethod
    def _make_noisey_image(img, epsilon):
        noise = np.random.randn(*img.shape)
        noise_norm = noise / np.linalg.norm(noise.reshape(-1))
        additive_noise = epsilon * noise_norm
        x_noisey = np.clip(img + additive_noise, 0, 1)
        return x_noisey, additive_noise


    def __call__(self, img, epsilon=None):
        '''
        will return a noisy image with noise level epsilon
        None will be returned if noise level is too low acccording to acceptable_percent
        '''
        if epsilon is None:
            epsilon = self.epsilon
        for _ in range(self.max_iter):
            x_noisey, additive_noise = self._make_noisey_image(img, epsilon)
            # check noise level and actual noise level reduced after clipping
            expected_noise = np.sqrt(np.mean(np.square(additive_noise)))
            actual_noise = np.sqrt(np.mean(np.square(x_noisey - img)))
            # option to reject and redo the noise if too low (recursion)
            if actual_noise >= self.acceptable_percent * expected_noise:
                return x_noisey
        return None

