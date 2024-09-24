'''
Base class for getting a score from a torch model as input, outputs a numpy score
'''
from abc import ABC, abstractmethod
import torch
from torchvision.transforms.v2 import Resize as Resize_v2


class base_scorer_torch(ABC):
    def __init__(self, im_size=(128, 128), device='cpu', **kwargs):
        # resizer for images
        self.im_size = im_size
        self.resizer = Resize_v2(im_size)

        # device
        if device == 'cuda':
            if not torch.cuda.is_available():
                print(
                    'Cuda not available even though requested for model, running on CPU instead')
                device = 'cpu'
        self.device = device

        # setup for child class
        self.setup(**kwargs)

    @abstractmethod
    def setup(self, **kwargs):
        pass

    def __call__(self, x):
        x = self._preprocess_image(x)
        x = x.to(self.device)

        # get score
        with torch.no_grad():
            score = self.get_score(x)

        if isinstance(score, torch.Tensor):
            score =  score.cpu().numpy()
        return score

    @abstractmethod
    def get_score(self, x):
        pass

    def _preprocess_image(self, x):
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
        if x.shape[2:] != self.im_size:
            x = self.resizer(x)
        return x

