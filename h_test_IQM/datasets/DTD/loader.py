from h_test_IQM.datasets.DTD.VARS import DTD_META_CSV, DTD_IMAGE_DIR
from h_test_IQM.datasets.DTD.downloader import download_DTD
from h_test_IQM.datasets.abstract_dataset import generic_loader


class DTD_LOADER(generic_loader):
    '''
    Generic data loader for CIFAR-10
    Args:
        indicies_to_use: list of in the dataset to use if you require to use a split of the dataset
        image_dict: dictionary of pre loaded images filename as keys and torch tensor as values 
    '''
    # change downloaddata, setvars, _get_image in child class to use this data template

    def download_data(self):
        download_DTD()

    def set_vars(self):
        self.csv_file = DTD_META_CSV
        self.image_dir = DTD_IMAGE_DIR
        self.dataset_name = 'DTD'
        self.num_classes = 47

if __name__ == '__main__':
    from tqdm import tqdm
    a = DTD_LOADER()
    ims = a.get_images_dict()

    b = DTD_LOADER(image_dict=ims)

    # benchmark
    for e in tqdm(range(10)):
        for i in tqdm(b, leave=False):
            pass
