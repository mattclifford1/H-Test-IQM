from h_test_IQM.datasets.CIFAR_100.VARS import CIFAR_100_META_CSV, CIFAR_100_IMAGE_DIR
from h_test_IQM.datasets.CIFAR_100.downloader import download_CIFAR_100
from h_test_IQM.datasets.abstract_dataset import generic_loader


class CIFAR_100_LOADER(generic_loader):
    '''
    Generic data loader for CIFAR-10
    Args:
        indicies_to_use: list of in the dataset to use if you require to use a split of the dataset
        image_dict: dictionary of pre loaded images filename as keys and torch tensor as values 
    '''
    # change downloaddata, setvars, _get_image in child class to use this data template

    def download_data(self):
        download_CIFAR_100()

    def set_vars(self):
        self.csv_file = CIFAR_100_META_CSV
        self.image_dir = CIFAR_100_IMAGE_DIR
        self.dataset_name = 'CIFAR_100'
        self.num_classes = 100

if __name__ == '__main__':
    from tqdm import tqdm
    a = CIFAR_100_LOADER()
    ims = a.get_images_dict()

    b = CIFAR_100_LOADER(image_dict=ims)

    # benchmark
    for e in tqdm(range(10)):
        for i in tqdm(b, leave=False):
            pass
