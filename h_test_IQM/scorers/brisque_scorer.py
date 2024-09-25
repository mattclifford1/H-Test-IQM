'''
Get scoring from entropy model
Currently only supports entropy model with 2 centers and we take the ratio of -1, 1 either spacially or flattened
'''
from h_test_IQM.metrics.IQMs import BRISQUE
from h_test_IQM.scorers.numpy_scorers import base_scorer_numpy


class brisque_model(base_scorer_numpy):
    def setup(self, **kwargs):
        self.model = BRISQUE()

    def get_score(self, x):
        return self.model(x)


if __name__ == '__main__':
    # test
    import numpy as np

    # create random image
    x = np.random.rand(5, 32, 32, 3).astype(np.float32)

    scores = brisque_model()(x)
    print(scores)
    print(type(scores))
