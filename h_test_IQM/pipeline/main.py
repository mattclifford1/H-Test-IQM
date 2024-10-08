import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from tqdm import tqdm

from h_test_IQM.datasets import DATASET_PROPORTIONS
from h_test_IQM.datasets.torch_loaders import get_preloaded, get_all_loaders
from h_test_IQM.distortions import TRANSFORMS
from h_test_IQM.scorers import SCORERS

def _get_preloaded(dataset='CIFAR_10', device='cpu'):
    if dataset in ['CIFAR_10', 'IMAGENET64_TRAIN', 'IMAGENET64_VAL']:
        return get_preloaded(dataset=dataset, device=device)
    else:
        return False

def fetch_preloaded(dataset_list, device='cpu', dev=False):
    pre_loaded_images = {}
    for dataset in set(dataset_list):
        if dev == True:
            pre_loaded_images[dataset] = {}
        else:
            pre_loaded_images[dataset] = _get_preloaded(dataset=dataset, device=device)
    return pre_loaded_images


def get_scores(dataset_target='CIFAR_10',
               dataset_test='CIFAR_10',
               target_labels='all',
               test_labels='all',
               transform_target=None,
               transform_test=None,
               scorer='entropy-2-mse',
               test='plot_hist',
               device='cuda',
               batch_size=32,
               dataset_proportion_target=1,
               dataset_proportion_test=1,
               dev=False,
               help=False,
               seed=0,
               _print=True,
               preloaded_ims=None):
    if help == True:
        print('''
Pipeline to test an image dataset compared to a target distribution.
              
Available params:
    dataset_target: 'CIFAR_10', 'IMAGENET64_TRAIN', 'IMAGENET64_VAL', 'KODAK'
    dataset_test: 'CIFAR_10', 'IMAGENET64_TRAIN', 'IMAGENET64_VAL', 'KODAK'
    transform_target: None, 'epsilon_noise' 'gaussian_noise', None
    transform_test: None, See above
    scorer: 'entropy-2-mse'
    test: 'plot_hist', 'KL' (can be a list)
    device: 'cuda', 'cpu'
    batch_size: int
    dev: bool
        ''')
        return
    if _print == True:
        print(f'''Pipeline: 
        target_dataset:   {dataset_target} 
        test_dataset:     {dataset_test} 
        transform_target: {transform_target} 
        transform_test:   {transform_test} 
        scorer:           {scorer} 
        test type:        {test} 

    Extras:
        device:     {device}
        batch size: {batch_size}
        dev mode:   {dev}
    ''')

    # change to full dataset size if not in dev mode
    if dev == True:
        dataset_proportion_target = DATASET_PROPORTIONS[dataset_target]
        dataset_proportion_test = DATASET_PROPORTIONS[dataset_test]

    # check if cuda is available
    if device == 'cuda':
        if not torch.cuda.is_available():
            if _print == True:
                print('cuda not available, using cpu')
            device = 'cpu'

    # INIT PRELOADED ########################################################################################
    if preloaded_ims == None:
        preloaded_ims = {}
    if not isinstance(preloaded_ims, dict):
        raise ValueError(f'incorrect preloaded_ims type: {type(preloaded_ims)}, needs to be none of dict')
    # initialise as empty if not already got (when scoring will be cached to preloaded_ims)
    if dataset_target not in preloaded_ims:
        preloaded_ims[dataset_target] = {}
    if dataset_test not in preloaded_ims:
        preloaded_ims[dataset_test] = {}

    # DISTORTIONS ########################################################################################
    if transform_target in TRANSFORMS:
        transform_func_target = TRANSFORMS[transform_target]()
    else:
        raise ValueError(f'{transform_target} transform_target needs to be one of {TRANSFORMS.keys()}')
    
    if transform_test in TRANSFORMS:
        transform_func_test = TRANSFORMS[transform_test]()
    else:
        raise ValueError(f'{transform_test} transform_test needs to be one of {TRANSFORMS.keys()}')
        

    # SCORER ########################################################################################
    if scorer in SCORERS:
        model = SCORERS[scorer](im_size=(256, 256), device=device)
    else: 
        raise ValueError(f'{scorer} scorer needs to be one of {SCORERS.keys()}')


    # TESTING ########################################################################################
    # DATA TARGET LOADING ########################################################################################
    target_dataloader = get_all_loaders(
        device=device,
        batch_size=batch_size,
        pre_loaded_images=preloaded_ims[dataset_target],
        dataset=dataset_target,
        dataset_proportion=dataset_proportion_target,
        seed=seed,
        labels_to_use=target_labels
    )
    if dev == True:
        if _print == True:
            print('scoring target')
    scores_target = get_sample_from_scorer(
        target_dataloader, transform_func_target, model, name='scoring target')
    
    # get any cached images to re use in the test
    preloaded_ims[dataset_target] = target_dataloader.dataset.image_dict

    # DATA TEST LOADING ########################################################################################
    # get test dataset (use cached data from target is possible)
    test_dataloader = get_all_loaders(
        device=device,
        batch_size=batch_size,
        # pre_loaded_images=preloaded_ims[dataset_test],
        pre_loaded_images={},
        dataset=dataset_test,
        dataset_proportion=dataset_proportion_test,
        seed=seed,
        labels_to_use=test_labels)

    if dev == True:
        if _print == True:
            print('scoring test')
    scores_test = get_sample_from_scorer(
        test_dataloader, transform_func_test, model, name='scoring test')

    # get any cached images to return
    preloaded_ims[dataset_test] = test_dataloader.dataset.image_dict
    
    if _print == True:
        print(f'''num target samples: {len(target_dataloader.dataset)
                            }\nnum test samples: {len(test_dataloader.dataset)}\n''')
    
    # MAKE PDFs ########################################################################################
    results = {}
    dist_target, dist_test, target_bins, test_bins = samples_to_pdf(
        scores_target, scores_test, num_bins=50)
    if 'KL' in test:
        # use the histogram of score samples to get some sort of "PMF/PDF"
        kl = entropy(pk=dist_target, 
                     qk=dist_test)
        if _print == True:
            print(f'KL divergence: {kl}')
        results['KL'] = kl

    if 'plot_hist' in test:
        plot_hist(dist_target, target_bins, name='target')
        plot_hist(dist_test, test_bins, name='test')
        plt.legend()
        plt.show()

    return {'scores_target': scores_target, 
            'scores_test': scores_test, 
            'results': results,
            'preloaded_ims': preloaded_ims}


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
            img = batch[0] # just get the image not labels
        else:
            img = batch
        # transform
        if transform is not None:
            img = transform(img)
            if img is None:
                continue
        # score
        score = scorer(img)
        if len(score.shape) == 2:
            for s in score:
                scores.append(s)
            scores.append(score)
        else:
            scores.append(score)

    # show usesr if there were any rejected transformations of images
    if hasattr(transform, 'num_rejected'):
        if transform.num_rejected > 0:
            print(f'{transform.num_rejected} images were rejected in distortion transform')

    # stack all scores into one array
    scores = np.hstack(scores)
    return scores

if __name__ == '__main__':
    get_scores(
        dataset_target='Caltech256',
        dataset_test='Caltech101',
        # test_labels=[0, 1],
        transform_test='gaussian_noise',
        scorer='entropy-2-mse',
        # scorer='BRISQUE',
        test='KL',
        dev=True,
        )