from .entropy_AE import entropy_encoder_model
from .brisque_scorer import brisque_model
from .baseline_scorers import (pixel_std_model, pixel_entropy_model,
                               jpeg_bytes_model)


class init_entropy_encoder():
    def __init__(self,
                 metric='mse',
                 dist='natural',
                 centers=2,
                 ):
        self.metric = metric
        self.dist = dist
        self.centers = centers

    def __call__(self, im_size, device):
        return entropy_encoder_model(metric=self.metric,
                               dist=self.dist,
                               centers=self.centers,
                               im_size=im_size, 
                               device=device)
    

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
    'entropy-2-nlpd-u': init_entropy_encoder(metric='nlpd', dist='uniform', centers=2),
    # baselines / controls -- see baseline_scorers.py
    'BRISQUE': init_numpy_scorer(brisque_model),
    'pixel_std': init_numpy_scorer(pixel_std_model),
    'pixel_entropy': init_numpy_scorer(pixel_entropy_model),
    'jpeg_bytes': init_numpy_scorer(jpeg_bytes_model),
}