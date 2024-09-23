from .additive_noise import epsilon_noise, Gaussian_noise


def init_epsilon_noise():
    return epsilon_noise(epsilon=1, acceptable_percent=0.9, max_iter=50, reject_low_noise=True)

def init_gaussian_noise():
    return Gaussian_noise(std=0.1)

def init_no_transform():
    return None


TRANSFORMS = {
    'epsilon_noise': init_epsilon_noise,
    'gaussian_noise': init_gaussian_noise,
    None: init_no_transform
}



