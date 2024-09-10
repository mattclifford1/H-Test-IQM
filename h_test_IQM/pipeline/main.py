import torch
import numpy as np
from tqdm import tqdm

from h_test_IQM.datasets.torch_loaders import CIFAR10_loader
from h_test_IQM.datasets.numpy_loaders import kodak_loader
from h_test_IQM.distortions.noise_sphere import epsilon_noise
from h_test_IQM.models.entropy_encoder import entropy_model


def get_scores(dataset_target='CIFAR-10',
               dataset_test='CIFAR-10',
               transform=None,
               scorer='entropy-2-mse',
               test='plot',
               device='cuda',
               dev=False):
    
    print(f'Running with: {dataset_target}, {dataset_test}, {transform}, {scorer}, {test}, {device}')

    if dev == True:
        dataset_proportion = 0.01
    else:
        dataset_proportion = 1

    # check if cuda is available
    if device == 'cuda':
        if not torch.cuda.is_available():
            print('cuda not available, using cpu')
            device = 'cpu'

    # pre-data loading
    if 'CIFAR' in dataset_target or 'CIFAR' in dataset_test:
        train_CIFAR, val_CIFAR, test_CIFAR, train_total = CIFAR10_loader(
            pre_loaded_images=False, device=device, dataset_proportion=dataset_proportion)

    # DATA TARGET LOADING
    if dataset_target == 'CIFAR-10':        
        target_dataloader = train_CIFAR
    elif dataset_target == 'KODAK':
        target_dataloader = kodak_loader()

    
    # DATA TEST LOADING
    if dataset_test == 'CIFAR-10':
        test_dataloader = test_CIFAR
    elif dataset_test == 'KODAK':
        test_dataloader = kodak_loader()


    # DISTORTION
    if transform == 'epsilon_noise':
        transform_func = epsilon_noise(
            epsilon=1, acceptable_percent=0.9, max_iter=10)
    else:
        transform_func = None
    
    # SCORER
    if scorer == 'entropy-2-mse':
        model = entropy_model(metric='mse', dist='natural',
                              centers=2, im_size=(256, 256))

    # TESTING
    dist_target = get_dist_from_scorer(
        target_dataloader, transform_func, model)
    dist_test = get_dist_from_scorer(
        test_dataloader, transform_func, model)
    
    return dist_target, dist_test


def get_dist_from_scorer(dataset, transform, scorer):
    scores = []
    for batch in tqdm(dataset, desc='scoring', leave=False):
        # get image
        if isinstance(batch, tuple) or isinstance(batch, list):
            img = batch[0]
        else:
            img = batch
        # transform
        if transform is not None:
            img = transform(img)
        # score
        score = scorer(img)
        if len(score.shape) == 2:
            for s in score:
                scores.append(s)
            scores.append(score)
        else:
            scores.append(score)
    # stack all scores into one array
    scores = np.hstack(scores)
    return scores

if __name__ == '__main__':
    get_scores()