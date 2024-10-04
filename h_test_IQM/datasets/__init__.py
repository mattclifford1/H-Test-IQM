from .MNIST.loader import MNIST_LOADER
from .Caltech256.loader import Caltech256_LOADER
from .CIFAR_10.loader import CIFAR_10_LOADER
from .IMAGENET.ImageNet64 import IMAGENET_64_LOADER_VAL, IMAGENET_64_LOADER_TRAIN 
from .uniform import UNIFORM_LOADER, UNIFORM_INSTANCES
from .KODAK.loader import kodak_loader

DATA_LOADER = {
    'MNIST': MNIST_LOADER,
    'Caltech256': Caltech256_LOADER,
    'CIFAR_10': CIFAR_10_LOADER,
    'IMAGENET64_VAL': IMAGENET_64_LOADER_VAL,
    'IMAGENET64_TRAIN': IMAGENET_64_LOADER_TRAIN,
    'UNIFORM': UNIFORM_LOADER,
    'KODAK': kodak_loader
}

TOTAL_INSTANCES = {
    'MNIST': 70000,
    'Caltech256': 30607,
    'CIFAR_10': 60000,
    'IMAGENET64_VAL': 50000,
    'IMAGENET64_TRAIN': 1281167,
    'UNIFORM': UNIFORM_INSTANCES,
}

# Proportions of the dataset to use for dev mode
DATASET_PROPORTIONS = {'CIFAR_10': 0.005,
                       'Caltech256': 0.01,
                       'UNIFORM': 0.005,
                       'MNIST': 0.005,
                       'IMAGENET64_TRAIN': 0.0002,
                       'IMAGENET64_VAL': 0.0002,
                       'KODAK': 1}
