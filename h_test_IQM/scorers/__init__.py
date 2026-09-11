from .entropy_AE import entropy_encoder_model
from .brisque_scorer import brisque_model
from .baseline_scorers import (pixel_std_model, pixel_entropy_model,
                               jpeg_bytes_model)
from .featurisers import colour_moments_model, random_pixels_model, resnet18_model


class init_entropy_encoder():
    def __init__(self,
                 metric='mse',
                 dist='natural',
                 centers=2,
                 spacial=False,
                 seed=0,
                 code='occupancy',
                 ):
        self.metric = metric
        self.dist = dist
        self.centers = centers
        # spacial=True keeps one +1-ratio per latent channel -> a 64-D score per image
        # instead of a scalar. Only the multivariate experiments use it.
        self.spacial = spacial
        # only read for dist='random', where it fixes the untrained weights
        self.seed = seed
        # 'activation' = the unquantised per-channel mean, for exp8's quantiser question
        self.code = code

    def __call__(self, im_size, device):
        return entropy_encoder_model(metric=self.metric,
                               dist=self.dist,
                               centers=self.centers,
                               spacial=self.spacial,
                               seed=self.seed,
                               code=self.code,
                               im_size=im_size,
                               device=device)


class init_torch_scorer():
    '''wraps a base_scorer_torch subclass with fixed setup kwargs'''
    def __init__(self, model_class, **setup_kwargs):
        self.model_class = model_class
        self.setup_kwargs = setup_kwargs

    def __call__(self, im_size, device):
        return self.model_class(im_size=im_size, device=device, **self.setup_kwargs)
    

class init_numpy_scorer():
    '''wraps a base_scorer_numpy subclass so it takes the same (im_size, device) call as
    the torch scorers. device is accepted and ignored -- these all run on CPU.'''
    def __init__(self, model_class, **setup_kwargs):
        self.model_class = model_class
        self.setup_kwargs = setup_kwargs

    def __call__(self, im_size=None, device=None):
        return self.model_class(im_size=im_size, **self.setup_kwargs)


SCORERS = {
    # the perceptual autoencoder -- {mse,ssim,nlpd} x {2,5} centres x {natural,uniform}
    # are all available in models/save_nets, only these are wired up so far
    'entropy-2-mse': init_entropy_encoder(metric='mse', dist='natural', centers=2),
    'entropy-2-ssim': init_entropy_encoder(metric='ssim', dist='natural', centers=2),
    'entropy-2-nlpd': init_entropy_encoder(metric='nlpd', dist='natural', centers=2),
    'entropy-2-mse-u': init_entropy_encoder(metric='mse', dist='uniform', centers=2),
    'entropy-2-ssim-u': init_entropy_encoder(metric='ssim', dist='uniform', centers=2),
    'entropy-2-nlpd-u': init_entropy_encoder(metric='nlpd', dist='uniform', centers=2),
    # multivariate variant: a 64-vector of per-channel +1 ratios rather than one scalar.
    # Needs a multivariate two-sample test -- see pipeline/multivariate.py.
    'entropy-2-mse-64d': init_entropy_encoder(metric='mse', dist='natural', centers=2,
                                              spacial=True),
    # exp7: the 64-D code for every usable checkpoint, plus the same architecture UNTRAINED
    # at three seeds -- the null for "does the code's structure come from natural images?"
    'entropy-2-ssim-64d': init_entropy_encoder(metric='ssim', dist='natural', centers=2,
                                               spacial=True),
    'entropy-2-nlpd-64d': init_entropy_encoder(metric='nlpd', dist='natural', centers=2,
                                               spacial=True),
    'entropy-2-mse-u-64d': init_entropy_encoder(metric='mse', dist='uniform', centers=2,
                                                spacial=True),
    'entropy-2-ssim-u-64d': init_entropy_encoder(metric='ssim', dist='uniform', centers=2,
                                                 spacial=True),
    'entropy-2-nlpd-u-64d': init_entropy_encoder(metric='nlpd', dist='uniform', centers=2,
                                                 spacial=True),
    'entropy-2-random-s0-64d': init_entropy_encoder(dist='random', centers=2,
                                                    spacial=True, seed=0),
    'entropy-2-random-s1-64d': init_entropy_encoder(dist='random', centers=2,
                                                    spacial=True, seed=1),
    'entropy-2-random-s2-64d': init_entropy_encoder(dist='random', centers=2,
                                                    spacial=True, seed=2),
    # exp8: the featuriser ladder -- see featurisers.py and PREREGISTRATION.md
    'colour-moments': init_torch_scorer(colour_moments_model),
    'random-pixels-s0-64d': init_torch_scorer(random_pixels_model, dim=64, seed=0),
    'random-pixels-s1-64d': init_torch_scorer(random_pixels_model, dim=64, seed=1),
    'random-pixels-s2-64d': init_torch_scorer(random_pixels_model, dim=64, seed=2),
    'entropy-2-random-s0-act64d': init_entropy_encoder(dist='random', spacial=True, seed=0,
                                                       code='activation'),
    'entropy-2-random-s1-act64d': init_entropy_encoder(dist='random', spacial=True, seed=1,
                                                       code='activation'),
    'entropy-2-random-s2-act64d': init_entropy_encoder(dist='random', spacial=True, seed=2,
                                                       code='activation'),
    'entropy-2-mse-act64d': init_entropy_encoder(metric='mse', dist='natural', spacial=True,
                                                 code='activation'),
    'resnet18-imagenet': init_torch_scorer(resnet18_model, pretrained=True),
    'resnet18-random-s0': init_torch_scorer(resnet18_model, pretrained=False, seed=0),
    'resnet18-random-s1': init_torch_scorer(resnet18_model, pretrained=False, seed=1),
    'resnet18-random-s2': init_torch_scorer(resnet18_model, pretrained=False, seed=2),
    # baselines / controls -- see baseline_scorers.py
    'BRISQUE': init_numpy_scorer(brisque_model),
    'pixel_std': init_numpy_scorer(pixel_std_model),
    'pixel_entropy': init_numpy_scorer(pixel_entropy_model),
    'jpeg_bytes': init_numpy_scorer(jpeg_bytes_model),
}