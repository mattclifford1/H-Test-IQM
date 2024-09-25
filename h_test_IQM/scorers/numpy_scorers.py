'''
Base class for getting a score from a numpy model, outputs a numpy score
'''
from abc import ABC, abstractmethod
import torch
import numpy as np
from torchvision.transforms.v2 import Resize as Resize_v2


class base_scorer_numpy(ABC):
    def __init__(self, im_size=None, **kwargs):
        # resizer for images
        self.im_size = im_size
        if self.im_size is not None:
            if not isinstance(im_size, tuple):
                raise ValueError('im_size should be a tuple or None')
            else:
                self.resizer = Resize_v2(im_size)

        # setup for child class
        self.setup(**kwargs)

    @abstractmethod
    def setup(self, **kwargs):
        pass

    def __call__(self, x):
        x = self._preprocess_image(x)

        # get scores
        scores = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            scores[i] = self.get_score(x[i, :, :, :])

        return scores

    @abstractmethod
    def get_score(self, x):
        pass

    def _preprocess_image(self, x):
        '''
        we are expecting a torch array usually - so do everything in torch then covert
        to numpy at the end
        '''
        # convert to pytorch tensor if numpy array input
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        # check if image is grayscale
        if len(x.shape) == 2:
            x = x.unsqueeze(2).expand(x.shape[0], x.shape[1], 3)

        # expand dimensions if needed for batch size 1
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # change to channel first if channels are last
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        # resize image to the correct size
        if self.im_size is not None and x.shape[2:] != self.im_size:
            x = self.resizer(x)

        # now convert back to numpy format
        x = x.permute(0, 2, 3, 1)
        return x.cpu().numpy()


'''
testing
'''

class test_model(base_scorer_numpy):
    def setup(self, **kwargs):
        pass

    def get_score(self, x):
        print(x.shape)
        return np.mean(x)

if __name__ == '__main__':
    # create random image
    x = np.random.rand(5, 32, 32, 3).astype(np.float32)

    scores = test_model()(x)
    print(scores)
    print(type(scores))
