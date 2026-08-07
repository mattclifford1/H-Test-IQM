from torch.utils.data import DataLoader
from h_test_IQM.datasets import DATA_LOADER, TOTAL_INSTANCES
from h_test_IQM.datasets.utils.proportions import get_indicies

NORMALISE = (0, 1)


def get_preloaded(dataset='CIFAR_10', device='cpu'):
    # load the main dataset images etc.
    loader = DATA_LOADER[dataset](normalise=NORMALISE, 
                                    device=device)
    pre_loaded_images = loader.get_images_dict()
    return pre_loaded_images

def get_classes(dataset='CIFAR_10', numerical=False):
    # load all the classes for the dataset
    loader = DATA_LOADER[dataset]()
    if numerical == True:
        return loader.get_numerical_labels()
    else:
        return loader.get_str_labels()


def get_all_loaders(device='cpu',
                    batch_size=32,
                    pre_loaded_images=None,
                    dataset='CIFAR_10',
                    dataset_proportion=0.5,
                    seed=0,
                    labels_to_use='all',
                    partition=None):
    '''
    get a torch loader for a (subsample of a) dataset
        set pre_loaded_images to True to load all images into RAM/VRAM
        device = 'cpu' or 'cuda'
        dataset_proportion: fraction of the dataset to use, between 0 and 1. This is the
            ACTUAL fraction -- there used to be an unused [0.4, 0.3, 0.3] train/val/test
            split applied first, which silently made dataset_proportion=1 mean 40% of the
            data (see FINDINGS.md 3.4).
        partition: None, 'a' or 'b'. Splits the shuffled indices in half first and draws
            from that half only, so 'a' and 'b' at the SAME seed are guaranteed disjoint.
            This is how to build an honest control -- a different seed only re-shuffles and
            gives chance-level overlap, not disjointness.
        dataset = 'CIFAR_10' or 'IMAGENET64_VAL' or 'IMAGENET64_TRAIN' or 'UNIFORM'(uniform noise data)
    '''
    if dataset.upper() == 'KODAK':
        return DATA_LOADER['KODAK']()

    # get the pool of indices to draw from
    if labels_to_use == 'all':
        total = TOTAL_INSTANCES[dataset]
        all_inds_to_use = list(range(total))
    elif not isinstance(labels_to_use, list):
        raise ValueError(
            f"labels_to_use needs to be a list of labels to use (or 'all'), got: {labels_to_use}")
    else:
        if isinstance(labels_to_use[0], str):
            labels = get_classes(dataset=dataset, numerical=False)
        else:
            labels = get_classes(dataset=dataset, numerical=True)
        all_inds_to_use = []
        # loop over inds and labels
        for i, l in enumerate(labels):
            if l in labels_to_use:
                all_inds_to_use.append(i)

    # shuffle once, then optionally take a disjoint half
    if partition is None:
        train_inds, = get_indicies([1.0], total_instances=all_inds_to_use, seed=seed)
    elif partition in ('a', 'b'):
        half_a, half_b = get_indicies(
            [0.5, 0.5], total_instances=all_inds_to_use, seed=seed)
        train_inds = half_a if partition == 'a' else half_b
    else:
        raise ValueError(f"partition needs to be None, 'a' or 'b', got: {partition}")

    # reduce the amount of data
    if not isinstance(dataset_proportion, str):
        train_total = max(min(int(dataset_proportion*len(train_inds)), len(train_inds)), 1)
        train_inds = train_inds[:train_total]   # inds are shuffled already so we can take a random sample

    # load images into RAM/VRAM
    if pre_loaded_images == True:
        pre_loaded_images = get_preloaded(dataset=dataset, device=device)
    elif pre_loaded_images == None or pre_loaded_images == False:
        pre_loaded_images = {}
    else:
        if not isinstance(pre_loaded_images, dict):
            raise ValueError(f'incorrect pre_loaded_images type: {type(pre_loaded_images)}')

    # get loaders
    if not isinstance(dataset_proportion, str):
        train_loader = DATA_LOADER[dataset](normalise=NORMALISE, indicies_to_use=train_inds, image_dict=pre_loaded_images)
    elif dataset_proportion == 'uniform': # get uniform noise loader
        train_total = 5000
        if dataset == 'CIFAR_10':
            size = (3, 32, 32)
        elif dataset == 'IMAGENET64_VAL' or dataset == 'IMAGENET64_TRAIN':
            size = (3, 64, 64)

        train_loader = DATA_LOADER['UNIFORM'](normalise=NORMALISE, length=train_total, size=size)
    else:
        raise ValueError(f'Cannot use trainset type/size: {dataset_proportion}')
    # get torch loaders
    train_dataloader = DataLoader(train_loader,  # type: ignore
                                batch_size=batch_size,
                                shuffle=True)
    return train_dataloader