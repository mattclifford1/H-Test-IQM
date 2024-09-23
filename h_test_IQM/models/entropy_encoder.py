'''
Get scoring from entropy model
Currently only supports entropy model with 2 centers and we take the ratio of -1, 1 either spacially or flattened
'''
import os
import torch
from torchvision.transforms import Resize
from torchvision.transforms.v2 import Resize as Resize_v2

from h_test_IQM.models.compressive_AE_model import EntropyLimitedModel


class entropy_model:
    def __init__(self, im_size=(128, 128), metric='mse', dist='natural', centers=2, device='cpu', spacial=False):
        # checks    
        if metric not in ['mse', 'ssim', 'nlpd']:
            raise ValueError('Invalid metric, needs to be one of: mse, ssim, nlpd')
        if dist not in ['natural', 'uniform']:
            raise ValueError('Invalid training distribution, needs to be one of: natural, uniform')
        if centers not in [2, 5]:
            raise ValueError('Invalid number of centers, needs to be one of: 2, 5')
        self.centers = centers
        self.spacial = spacial
        self.model = EntropyLimitedModel(
            N=128, M=64, sigmoid=True, centers=self.centers)

        # get weights path
        current_file = os.path.abspath(__file__)
        name = f'{metric}-{self.centers}'
        if dist == 'uniform':
            name += '-u'
        model_path = os.path.join(os.path.dirname(current_file), 'save_nets', f'{name}.pth')

        # load weights
        if device == 'cuda':
            if not torch.cuda.is_available():
                print('Cuda not available even though requested for model, running on CPU instead')
                device = 'cpu'
        self.device = device
        # load model from checkpoint - https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # puts into exact rounding for the embedding code vectors
        self.model.to(device)

        # extras
        # self.resizer = Resize_v2((256, 256))
        # self.resizer = Resize_v2((128, 128))
        # self.resizer = Resize_v2((64, 64))
        # self.resizer = Resize_v2((32, 32))
        # self.resizer = Resize_v2((16, 16))
        self.resizer = Resize_v2(im_size)

    def embeddings(self, x):
        x = self._preprocess_image(x)
        # encode
        x = x.to(self.device)
        with torch.no_grad():
            y = self.model.encode(x)
            y_hat = self.model.quantise(y)
        y_hat = y_hat.cpu().numpy()
        return y_hat
    
    def _preprocess_image(self, x):
        # convert to pytorch tensor if numpy array input
        # print(x.shape)
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
    
    def counts_per_emb_feature_flat(self, x):
        # get the counts of values in the embeddings, we ignore all spacial information
        embs = self.embeddings(x)
        # flatten the embeddings
        embs = embs.reshape((embs.shape[0], -1))
        if self.centers == 2:  # can do the ratio of -1, 1
            ones_emb = (embs==1)
            counts = ones_emb.sum(axis=1)/ones_emb.shape[1]

        # TODO: center 5
        elif self.centers == 5:  # return all counts -2, -1, 0, 1, 2
            counts = embs

        return counts
    
    def counts_per_emb_feature_spacial(self, x):
        # get the counts of values in the embeddings, using all spacial information
        embs = self.embeddings(x)
        # flatten the embeddings
        embs = embs.reshape((embs.shape[0], embs.shape[1], -1))
        if self.centers == 2:  # can do the ratio of -1, 1
            ones_emb = (embs==1)
            counts = ones_emb.sum(axis=2)/ones_emb.shape[2]

        # TODO: center 5
        elif self.centers == 5:  # return all counts -2, -1, 0, 1, 2
            counts = embs

        return counts
    
    def __call__(self, x):
        if self.spacial == False:
            return self.counts_per_emb_feature_flat(x)
        else:
            return self.counts_per_emb_feature_spacial(x)

        

if __name__ == '__main__':
    # test
    import numpy as np
    import matplotlib.pyplot as plt

    # create model
    model = entropy_model(metric='mse', dist='natural', centers=2, im_size=(256, 256), device='cuda')

    # create random image
    x = np.random.rand(32, 32, 3).astype(np.float32)
    x = np.expand_dims(x, axis=0)

    # encode
    y = model.embeddings(x)
    print('embeddings:', y.shape)
    c = model.counts_per_emb_feature_flat(x)
    print('counts flat:', c.shape)
    c = model.counts_per_emb_feature_spacial(x)
    print('counts spac:', c.shape)


    ## plot
    # plt.imshow(x.squeeze())
    # plt.show()