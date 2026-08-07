from .MNIST.loader import MNIST_LOADER
from .Caltech101.loader import Caltech101_LOADER
from .Caltech256.loader import Caltech256_LOADER
from .CIFAR_10.loader import CIFAR_10_LOADER
from .CIFAR_100.loader import CIFAR_100_LOADER
from .DTD.loader import DTD_LOADER
from .IMAGENET.ImageNet64 import IMAGENET_64_LOADER_VAL, IMAGENET_64_LOADER_TRAIN
from .uniform import UNIFORM_LOADER, UNIFORM_INSTANCES
from .KODAK.loader import kodak_loader

DATA_LOADER = {
    'MNIST': MNIST_LOADER,
    'Caltech101': Caltech101_LOADER,
    'Caltech256': Caltech256_LOADER,
    'CIFAR_10': CIFAR_10_LOADER,
    'CIFAR_100': CIFAR_100_LOADER,
    'DTD': DTD_LOADER,
    'IMAGENET64_VAL': IMAGENET_64_LOADER_VAL,
    'IMAGENET64_TRAIN': IMAGENET_64_LOADER_TRAIN,
    'UNIFORM': UNIFORM_LOADER,
    'KODAK': kodak_loader
}

TOTAL_INSTANCES = {
    'MNIST': 70000,
    'Caltech101': 8677,
    'Caltech256': 30607,
    'CIFAR_10': 60000,
    'CIFAR_100': 60000,
    'DTD': 5640,
    'IMAGENET64_VAL': 50000,
    'IMAGENET64_TRAIN': 1281167,
    'UNIFORM': UNIFORM_INSTANCES,
    'KODAK': 24,
}

# Proportions of the dataset to use for dev mode
DATASET_PROPORTIONS = {
    'MNIST': 0.005,
    'Caltech101': 0.05,
    'Caltech256': 0.01,
    'CIFAR_10': 0.005,
    'CIFAR_100': 0.005,
    'DTD': 0.05,
    'UNIFORM': 0.005,
    'IMAGENET64_TRAIN': 0.0002,
    'IMAGENET64_VAL': 0.0002,
    'KODAK': 1
}
