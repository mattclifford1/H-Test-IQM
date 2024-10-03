'''
Abstract dataset to be used with the loading functions required
'''
from abc import ABC, abstractmethod
import torch


class abstract_dataset_torch(ABC):
    def __init__(self, 
                 indicies_to_use='all',
                 image_dict={},
                 cache_data=True,
                 normalise=(0, 1),
                 dtype=torch.float32,
                 device='cpu',
                 **kwargs):
        self.indicies_to_use = indicies_to_use
        self.image_dict = image_dict
        self.cache_data = cache_data
        self.normalise = normalise
        self.dtype = dtype
        self.device = device

        self.setup(**kwargs)

    @abstractmethod
    def setup(self, **kwargs):
        ''' any extras needed for the dataset '''
        pass

    @abstractmethod
    def reduce_data(self):
        ''' reduce the data based on indicies to use -- will be dataset specific '''
        pass

    @abstractmethod
    def get_images_dict(self):
        '''get dict of all pre loaded and processed images - useful to pass to other dataset
        so that you only have to read from file and process once'''
        pass

    @abstractmethod
    def get_numerical_labels(self):
        ''' get numerical labels for the dataset '''
        pass

    @abstractmethod
    def get_str_labels(self):
        ''' get string labels for the dataset '''
        pass

    @abstractmethod
    def __len__(self):
        ''' get length of dataset '''
        pass

    @abstractmethod
    def __getitem__(self, idx):
        ''' get item from dataset '''
        pass    