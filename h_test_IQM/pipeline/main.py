import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from h_test_IQM.datasets import DATASET_PROPORTIONS
from h_test_IQM.datasets.torch_loaders import get_preloaded, get_all_loaders
from h_test_IQM.distortions import TRANSFORMS
from h_test_IQM.scorers import SCORERS
from h_test_IQM.pipeline.h_tests import compare, DIVERGENCES, TESTS

# the scorer resizes every image to this before encoding. CIFAR is 32x32, so the default
# upsamples it 8x -- see FINDINGS.md 3.6. Kept as the default so old numbers reproduce.
DEFAULT_IM_SIZE = (256, 256)

# Scorers are stateless at inference but expensive to build (entropy_AE reads a checkpoint off
# disk every time). The repeated-run experiments call get_scores hundreds of times with the same
# scorer, where that rebuild dominates the runtime, so keep one instance per configuration.
_SCORER_CACHE = {}


def _get_scorer(scorer, im_size, device):
    if scorer not in SCORERS:
        raise ValueError(f'{scorer} scorer needs to be one of {list(SCORERS.keys())}')
    key = (scorer, tuple(im_size), device)
    if key not in _SCORER_CACHE:
        _SCORER_CACHE[key] = SCORERS[scorer](im_size=tuple(im_size), device=device)
    return _SCORER_CACHE[key]

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
               preloaded_ims=None,
               shift_seed_test=0,
               im_size=DEFAULT_IM_SIZE,
               num_bins=50,
               alpha=0.5,
               n_permutations=0,
               partition_target=None,
               partition_test=None):
    if help == True:
        print('''
Pipeline to test an image dataset compared to a target distribution.
              
Available params:
                        --- DATASETS ---
    dataset_target: 'CIFAR_10', 'IMAGENET64_TRAIN', 'IMAGENET64_VAL', 'KODAK'
    dataset_test: 'CIFAR_10', 'IMAGENET64_TRAIN', 'IMAGENET64_VAL', 'KODAK'
    dataset_proportion_target: float -- the actual fraction of the dataset (0-1)
    dataset_proportion_test: float
    partition_target: None, 'a', 'b' -- disjoint halves, for a guaranteed-clean control.
    partition_test:   'a' vs 'b' with the same seed share no images at all.

                        --- LABELS ---
    target_labels: 'all', list of labels
    test_labels: 'all', list of labels

                        --- TRANSFORMS ---
    transform_target: None, 'epsilon_noise' 'gaussian_noise', None
    transform_test: None, See above

                        --- SCORERS ---
    scorer: 'entropy-2-mse', 'BRISQUE', 'pixel_std', 'jpeg_bytes'
    im_size: (H, W) every image is resized to this before scoring. Default (256, 256).

                        --- TESTING ---
    test: 'plot_hist', and any of
          divergences (effect size):  'KL', 'JS', 'wasserstein'
          tests (statistic + p-value): 'KS', 'CVM', 'AD'
          or 'all' for every statistic. Can be a list.
    num_bins: int, binning for the binned divergences (KL, JS)
    alpha: float, count smoothing for the binned divergences (0.5 = Krichevsky-Trofimov)
    n_permutations: int, >0 adds a permutation p-value to each divergence

                        --- EXTRAS ---
    device: 'cuda', 'cpu'
    batch_size: int
    seed: int
    dev: bool (If True, the code will run on very small data size)
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
    model = _get_scorer(scorer, im_size, device)


    # TESTING ########################################################################################
    # DATA TARGET LOADING ########################################################################################
    target_dataloader = get_all_loaders(
        device=device,
        batch_size=batch_size,
        pre_loaded_images=preloaded_ims[dataset_target],
        dataset=dataset_target,
        dataset_proportion=dataset_proportion_target,
        seed=seed,
        labels_to_use=target_labels,
        partition=partition_target,
    )
    if dev == True:
        if _print == True:
            print('scoring target')
    scores_target = get_sample_from_scorer(
        target_dataloader, transform_func_target, model, name='scoring target')
    
    # get any cached images to re use in the test
    if hasattr(target_dataloader.dataset, 'image_dict'):
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
        seed=seed+shift_seed_test,
        labels_to_use=test_labels,
        partition=partition_test)

    if dev == True:
        if _print == True:
            print('scoring test')
    scores_test = get_sample_from_scorer(
        test_dataloader, transform_func_test, model, name='scoring test')

    # get any cached images to return
    if hasattr(test_dataloader.dataset, 'image_dict'):
        preloaded_ims[dataset_test] = test_dataloader.dataset.image_dict
    
    if _print == True:
        print(f'''num target samples: {len(target_dataloader.dataset)
                            }\nnum test samples: {len(test_dataloader.dataset)}\n''')
    
    # COMPARE ########################################################################################
    # `test` may be a single name or a list. Everything that isn't 'plot_hist' is a statistic.
    requested = [test] if isinstance(test, str) else list(test)
    stat_names = [t for t in requested if t != 'plot_hist']

    results = {}
    if stat_names:
        if 'all' in stat_names:
            stat_names = list(DIVERGENCES) + list(TESTS)
        unknown = [t for t in stat_names if t not in DIVERGENCES and t not in TESTS]
        if unknown:
            raise ValueError(
                f'unknown test(s) {unknown}, expected any of '
                f'{list(DIVERGENCES) + list(TESTS)} (or "plot_hist")')
        results = compare(scores_target, scores_test, which=stat_names,
                          num_bins=num_bins, alpha=alpha,
                          n_permutations=n_permutations, seed=seed)
        if _print == True:
            for name, value in results.items():
                print(f'{name}: {value:.6g}')

    if 'plot_hist' in requested:
        dist_target, dist_test, target_bins, test_bins = samples_to_pdf(
            scores_target, scores_test, num_bins=num_bins)
        plot_hist(dist_target, target_bins, name='target')
        plot_hist(dist_test, test_bins, name='test')
        plt.xlabel('Score')
        plt.ylabel('Density')
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
    '''
    Two samples -> two density histograms over a shared range, FOR PLOTTING.

    Deliberately unsmoothed: an empty bin should show as empty. Divergences must not be
    computed from these -- use h_tests.samples_to_pmf, which smooths the counts and is
    therefore invariant to the units of the score (see FINDINGS.md 3.1).
    '''
    min_val = min(sample1.min(), sample2.min())
    max_val = max(sample1.max(), sample2.max())
    range_vals = (min_val, max_val)
    dist1, bins1 = np.histogram(
        sample1, bins=num_bins, range=range_vals, density=True)
    dist2, bins2 = np.histogram(
        sample2, bins=num_bins, range=range_vals, density=True)
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
            # multi-feature scorer (centers=5 / spacial=True): one row per image
            for s in score:
                scores.append(s)
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