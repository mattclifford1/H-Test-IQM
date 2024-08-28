import os
import torch
from torchvision.transforms import Resize
from torchvision.transforms.v2 import Resize as Resize_v2

from h_test_IQM.models.compressive_AE import EntropyLimitedModel


class entropy_model:
    def __init__(self, metric='mse', dist='natural', centers=2, device='cpu'):
        # checks    
        if metric not in ['mse', 'ssim', 'nlpd']:
            raise ValueError('Invalid metric, needs to be one of: mse, ssim, nlpd')
        if dist not in ['natural', 'uniform']:
            raise ValueError('Invalid training distribution, needs to be one of: natural, uniform')
        if centers not in [2, 5]:
            raise ValueError('Invalid number of centers, needs to be one of: 2, 5')

        self.model = EntropyLimitedModel(
            N=128, M=64, sigmoid=True, centers=centers)

        # get weights path
        current_file = os.path.abspath(__file__)
        name = f'{metric}-{centers}'
        if dist == 'uniform':
            name += '-u'
        model_path = os.path.join(os.path.dirname(current_file), 'save_nets', f'{name}.pth')

        # load weights
        self.device = device
        # load model from checkpoint - https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # puts into exact rounding for the embedding code vectors

        # extras
        # self.resizer = Resize_v2((256, 256))
        self.resizer = Resize_v2((128, 128))
        # self.resizer = Resize_v2((64, 64))
        # self.resizer = Resize_v2((32, 32))
        # self.resizer = Resize_v2((16, 16))

    def __call__(self, x, flat=True):
        x = self._preprocess_image(x)
        # encode
        x = x.to(self.device)
        with torch.no_grad():
            y = self.model.encode(x)
            y_hat = self.model.quantise(y)
        y_hat = y_hat.cpu().numpy()
        # return flattened tensor if needed
        if flat == True:
            y_hat = y_hat.flatten()
        return y_hat
    
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
        size = 128
        if x.shape[1:] != (3, size, size):
            # x = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(size, size), mode='bilinear').permute(0, 2, 3, 1)
            x = self.resizer(x)
        return x
        

if __name__ == '__main__':
    # test
    import numpy as np
    import matplotlib.pyplot as plt

    # create model
    model = entropy_model(metric='mse', dist='natural', centers=2)

    # create random image
    x = np.random.rand(32, 32, 3).astype(np.float32)
    x = np.expand_dims(x, axis=0)

    # encode
    y = model(x, flat=False)
    print(y.shape)
    # print(y.shape)

    # plot
    # plt.imshow(x.squeeze())
    # plt.show()