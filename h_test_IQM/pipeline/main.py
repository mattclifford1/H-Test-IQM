import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from tqdm import tqdm

from h_test_IQM.datasets.torch_loaders import CIFAR10_loader, IMAGENET64_loader, IMAGENET64VAL_loader
from h_test_IQM.datasets.numpy_loaders import kodak_loader
from h_test_IQM.distortions.noise_sphere import epsilon_noise
from h_test_IQM.models.entropy_encoder import entropy_model


def get_scores(dataset_target='CIFAR-10',
               dataset_test='CIFAR-10',
               transform=None,
               scorer='entropy-2-mse',
               test='plot_hist',
               device='cuda',
               batch_size=32,
               dev=False,
               help=False):
    if help == True:
        print('''
Pipeline to test an image dataset compared to a target distribution.
              
Available params:
    dataset_target: 'CIFAR-10', 'IMAGENET64', 'IMAGENET64VAL', 'KODAK'
    dataset_test: 'CIFAR-10', 'IMAGENET64', 'IMAGENET64VAL', 'KODAK'
    transform: None, 'epsilon_noise'
    scorer: 'entropy-2-mse'
    test: 'plot_hist', 'KL' (can be a list)
    device: 'cuda', 'cpu'
    batch_size: int
    dev: bool
        ''')
        return
    
    print(f'''Pipeline: 
    target_dataset: {dataset_target} 
    test_dataset:   {dataset_test} 
    transform:      {transform} 
    scorer:         {scorer} 
    test type:      {test} 

Extras:
    device:     {device}
    batch size: {batch_size}
    dev mode:   {dev}
''')

    if dev == True:
        dataset_proportion_CIFAR = 0.005
        dataset_proportion_IMAGENET = 0.0002
        dataset_proportion_IMAGENETVAL = 0.0002
    else:
        dataset_proportion_CIFAR = 1
        dataset_proportion_IMAGENET = 1
        dataset_proportion_IMAGENETVAL = 1

    # check if cuda is available
    if device == 'cuda':
        if not torch.cuda.is_available():
            print('cuda not available, using cpu')
            device = 'cpu'

    # pre-data loading
    if 'CIFAR' in dataset_target or 'CIFAR' in dataset_test:
        train_CIFAR, val_CIFAR, test_CIFAR, train_total = CIFAR10_loader(
            pre_loaded_images=False, 
            device=device, 
            dataset_proportion=dataset_proportion_CIFAR,
            batch_size=batch_size)
    if 'IMAGENET64' == dataset_target or 'IMAGENET64' == dataset_test:
        train_IMAGENET, val_IMAGENET, test_IMAGENET, train_total = IMAGENET64_loader(
            pre_loaded_images=False, 
            device=device, 
            dataset_proportion=dataset_proportion_IMAGENET,
            batch_size=batch_size)
    if 'IMAGENET64VAL' == dataset_target or 'IMAGENET64VAL' == dataset_test:
        train_IMAGENETVAL, val_IMAGENETVAL, test_IMAGENETVAL, train_total = IMAGENET64_loader(
            pre_loaded_images=False, 
            device=device, 
            dataset_proportion=dataset_proportion_IMAGENETVAL,
            batch_size=batch_size)

    # DATA TARGET LOADING
    if dataset_target == 'CIFAR-10':        
        target_dataloader = train_CIFAR
    elif dataset_target == 'IMAGENET64':    
        target_dataloader = train_IMAGENET
    elif dataset_target == 'IMAGENET64VAL':        
        target_dataloader = train_IMAGENETVAL
    elif dataset_target == 'KODAK':
        target_dataloader = kodak_loader()
    else:
        raise ValueError(f'{dataset_target} dataset_target not recognised')

    
    # DATA TEST LOADING
    if dataset_test == 'CIFAR-10':
        test_dataloader = train_CIFAR
    elif dataset_test == 'IMAGENET64':  
        test_dataloader = train_IMAGENET
    elif dataset_test == 'IMAGENET64VAL':     
        test_dataloader = train_IMAGENETVAL
    elif dataset_test == 'KODAK':
        test_dataloader = kodak_loader()
    else:
        raise ValueError(f'{dataset_test} dataset_test not recognised')


    # DISTORTION
    if transform == 'epsilon_noise':
        transform_func = epsilon_noise(
            epsilon=1, acceptable_percent=0.9, max_iter=10)
    else:
        transform_func = None
    
    # SCORER
    if scorer == 'entropy-2-mse':
        model = entropy_model(metric='mse', 
                              dist='natural',
                              centers=2, 
                              im_size=(256, 256),
                              device=device)

    # TESTING
    scores_target = get_sample_from_scorer(
        target_dataloader, transform_func, model, name='scoring target')
    scores_test = get_sample_from_scorer(
        test_dataloader, transform_func, model, name='scoring test')
    
    results = {}
    dist_target, dist_test, target_bins, test_bins = samples_to_pdf(
        scores_target, scores_test, num_bins=50)
    if 'KL' in test:
        # use the histogram of score samples to get some sort of "PMF/PDF"
        kl = entropy(pk=dist_target, 
                     qk=dist_test)
        print(f'KL divergence: {kl}')
        results['KL'] = kl

    if 'plot_hist' in test:
        plot_hist(dist_target, target_bins, name='target')
        plot_hist(dist_test, test_bins, name='test')
        plt.legend()
        plt.show()

    
    return scores_target, scores_test, results


def plot_hist(dist, bins, name=''):
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, dist, align='center',
            width=width, label=name, alpha=0.5)


def samples_to_pdf(sample1, sample2, num_bins=10):
    # convert scores into PDF via normalised histograms
    # use two samples for both on the same range
    min_val = min(sample1.min(), sample2.min())
    max_val = max(sample1.max(), sample2.max())
    range_vals = (min_val, max_val)
    dist1, bins1 = np.histogram(
        sample1, bins=num_bins, range=range_vals, density=True)
    dist2, bins2 = np.histogram(
        sample2, bins=num_bins, range=range_vals, density=True)
    
    # numerical stability if any bin is 0
    if np.any(dist1 == 0) or np.any(dist2 == 0):
        dist1 += 1e-6
        dist2 += 1e-6
        # normalise
        dist1 /= dist1.sum()
        dist2 /= dist2.sum()

    return dist1, dist2, bins1, bins2


def get_sample_from_scorer(dataset, transform, scorer, name='scorer'):
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