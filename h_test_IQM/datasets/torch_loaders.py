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
        return loader.get_labels()


def get_all_loaders(props=[0.4, 0.3, 0.3], 
                    device='cpu', 
                    batch_size=32, 
                    pre_loaded_images=None, 
                    dataset='CIFAR_10',
                    dataset_proportion=0.5,
                    seed=0,
                    labels_to_use='all'):
    '''
    get train, val and test torch loaders from CIFAR or IMAGENET 
        set pre_loaded_images to True to load all images into RAM/VRAM
        device = 'cpu' or 'cuda'
        dataset_proportion between 0 and 1 to reduce the amount of training data
        dataset = 'CIFAR_10' or 'IMAGENET64_VAL' or 'IMAGENET64_TRAIN' or 'UNIFORM'(uniform noise data)
    '''
    if dataset == 'kodak':
        return DATA_LOADER['KODAK']()
    
    # get inds - split into train, val and test
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

    train_inds, val_inds, test_inds = get_indicies(
        props, total_instances=all_inds_to_use, seed=seed)
    # reduce the amount of training data
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
    # val_loader = DATA_LOADER[dataset](normalise=NORMALISE, indicies_to_use=val_inds, image_dict=pre_loaded_images)
    # test_loader = DATA_LOADER[dataset](normalise=NORMALISE, indicies_to_use=test_inds, image_dict=pre_loaded_images)
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
    # val_dataloader = DataLoader(val_loader,  # type: ignore
    #                             batch_size=batch_size, 
    #                             shuffle=False,
    #                             drop_last=False)
    # test_dataloader = DataLoader(test_loader,  # type: ignore
    #                             batch_size=batch_size, 
    #                             shuffle=False,
    #                             drop_last=False)
    return train_dataloader#, val_dataloader, test_dataloader, train_total