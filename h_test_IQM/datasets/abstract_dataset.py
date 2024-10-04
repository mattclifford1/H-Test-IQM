'''
Abstract dataset to be used with the loading functions required
'''
from abc import ABC, abstractmethod
import os
from tqdm import tqdm
import pandas as pd
import torch
from torchvision.io import read_image


class abstract_dataset_torch(ABC):
    '''
    Args:
        indicies_to_use: list of in the dataset to use if you require to use a split of the dataset
        image_dict: dictionary of pre loaded images filename as keys and torch tensor as values 
    '''
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


'''
generic torch dataset for loading images in a folder with csv file for filenames and labels
'''


class generic_loader(abstract_dataset_torch):
    '''
    Generic data loader for CIFAR-10
    '''
    # change downloaddata, setvars, _get_image in child class to use this data template
    @abstractmethod
    def download_data(self):
        pass
    
    @abstractmethod
    def set_vars(self):
        # e.g.
        # self.csv_file = 
        # self.image_dir = 
        # self.dataset_name = 
        # self.num_classes = 
        pass
    
    # overwrite this in child if needed
    def _get_image(self, filename):
        # use cache data if loaded already
        if filename in self.image_dict:
            image = self.image_dict[filename]
        else:
            # load if not cached
            path = os.path.join(self.image_dir, filename)
            image = read_image(path)
            image = image.to(self.dtype)
            image = image/255
            image = image*(self.normalise[1]-self.normalise[0])
            image = image + self.normalise[0]
            image = image.to(self.device)
            if self.cache_data == True:
                self.image_dict[filename] = image
        return image

    def setup(self):
        # make sure dataset is downloaded
        self.download_data()

        # set vars
        self.set_vars()

        self.label_cache = {}
        self.meta_data = pd.read_csv(self.csv_file)
        self.filenames = self.meta_data['filename'].to_list()
        self.labels = self.meta_data['label'].to_list()
        self.numerical_label = self.meta_data['numerical_label'].to_list()

        # reduce data if needed
        self.reduce_data()

    def reduce_data(self):
        if self.indicies_to_use == 'all':
            return
        else:
            self.filenames = [self.filenames[i] for i in self.indicies_to_use]
            self.labels = [self.labels[i] for i in self.indicies_to_use]
            self.numerical_label = [self.numerical_label[i]
                                    for i in self.indicies_to_use]

    def get_images_dict(self):
        '''get dict of all pre loaded and processed images - useful to pass to other dataset
        so that you only have to read from file and process once'''
        images = {}
        for ind in tqdm(range(len(self.labels)), desc=f'Loading {self.dataset_name}', leave=False):
            filename = self.filenames[ind]
            # get image
            image = self._get_image(filename)
            images[filename] = image
        return images

    def get_numerical_labels(self):
        return self.numerical_label

    def get_str_labels(self):
        return self.labels

    def _get_labels(self, ind):
        if ind in self.label_cache:
            one_hot = self.label_cache[ind]['one_hot']
            label = self.label_cache[ind]['label']
        else:
            one_hot = torch.zeros(self.num_classes, dtype=self.dtype)
            one_hot[self.numerical_label[ind]] = 1
            # one_hot = one_hot.to(self.device)
            label = torch.tensor(self.numerical_label[ind])
            # label = label.to(self.device)
            if self.cache_data == True:
                self.label_cache[ind] = {'label': {}, 'one_hot': {}}
                self.label_cache[ind]['one_hot'] = one_hot
                self.label_cache[ind]['label'] = label
        return one_hot, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        filename = self.filenames[ind]
        # get image
        image = self._get_image(filename)
        # get labels
        one_hot, label = self._get_labels(ind)
        return image, one_hot, label
