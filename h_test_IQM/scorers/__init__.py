from .entropy_AE import entropy_encoder_model
from .brisque_scorer import brisque_model


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
    

def init_BRISQUE(*args, **kwargs):
    return brisque_model()
    

SCORERS = {
    'entropy-2-mse': init_entropy_encoder(metric='mse', dist='natural', centers=2),
    'BRISQUE': init_BRISQUE,
}   