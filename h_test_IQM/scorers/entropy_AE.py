'''
Get scoring from entropy model
Currently only supports entropy model with 2 centers and we take the ratio of -1, 1 either spacially or flattened
'''
import os
import torch

from h_test_IQM.models.compressive_AE_model import EntropyLimitedModel
from h_test_IQM.scorers.torch_scorers import base_scorer_torch


class entropy_encoder_model(base_scorer_torch):
    def setup(self, metric='mse', dist='natural', centers=2, spacial=False):
       # checks
        if metric not in ['mse', 'ssim', 'nlpd']:
            raise ValueError(
                'Invalid metric, needs to be one of: mse, ssim, nlpd')
        if dist not in ['natural', 'uniform']:
            raise ValueError(
                'Invalid training distribution, needs to be one of: natural, uniform')
        if centers not in [2, 5]:
            raise ValueError(
                'Invalid number of centers, needs to be one of: 2, 5')
        self.centers = centers
        self.spacial = spacial
        self.model = EntropyLimitedModel(
            N=128, M=64, sigmoid=True, centers=self.centers)

        # get weights path
        name = f'{metric}-{self.centers}'
        if dist == 'uniform':
            name += '-u'
        model_path = os.path.join(self.model.checkpoint_dir, f'{name}.pth')

        # load model from checkpoint - https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # puts into exact rounding for the embedding code vectors
        self.model.to(self.device)

    def get_score(self, x):
        if self.spacial == False:
            return self.counts_per_emb_feature_flat(x)
        else:
            return self.counts_per_emb_feature_spacial(x)

    def counts_per_emb_feature_flat(self, x):
        # get the counts of values in the embeddings, we ignore all spacial information
        embs = self._get_embeddings(x)
        # flatten the embeddings
        embs = embs.reshape((embs.shape[0], -1))
        if self.centers == 2:  # can do the ratio of -1, 1
            ones_emb = (embs == 1)
            counts = ones_emb.sum(axis=1)/ones_emb.shape[1]

        # TODO: center 5
        elif self.centers == 5:  # return all counts -2, -1, 0, 1, 2
            counts = embs

        return counts

    def counts_per_emb_feature_spacial(self, x):
        # get the counts of values in the embeddings, using all spacial information
        embs = self._get_embeddings(x)
        # flatten the embeddings
        embs = embs.reshape((embs.shape[0], embs.shape[1], -1))
        if self.centers == 2:  # can do the ratio of -1, 1
            ones_emb = (embs == 1)
            counts = ones_emb.sum(axis=2)/ones_emb.shape[2]

        # TODO: center 5
        elif self.centers == 5:  # return all counts -2, -1, 0, 1, 2
            counts = embs

        return counts

    def _get_embeddings(self, x, im_pre_processed=True):
        if im_pre_processed == False:
            x = self._preprocess_image(x)
            x = x.to(self.device)

        with torch.no_grad():
            y = self.model.encode(x)
            y_hat = self.model.quantise(y)
        return y_hat.cpu().numpy()


if __name__ == '__main__':
    # test
    import numpy as np
    import matplotlib.pyplot as plt


    # create random image
    x = np.random.rand(32, 32, 3).astype(np.float32)
    x = np.expand_dims(x, axis=0)

    # create model
    model_spacial = entropy_encoder_model(metric='mse', dist='natural',
                                  centers=2, im_size=(128, 128), device='cuda', spacial=True)
    model = entropy_encoder_model(metric='mse', dist='natural',
                                  centers=2, im_size=(128, 128), device='cuda', spacial=False)
    
    # scores
    print('embeddings:', model._get_embeddings(x, im_pre_processed=False).shape)
    print('counts flat:', model(x).shape)
    print('counts spac:', model_spacial(x).shape)

    # plot
    # plt.imshow(x.squeeze())
    # plt.show()
