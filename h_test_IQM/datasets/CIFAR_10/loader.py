from h_test_IQM.datasets.CIFAR_10.VARS import CIFAR_10_META_CSV, CIFAR_10_IMAGE_DIR
from h_test_IQM.datasets.CIFAR_10.downloader import download_CIFAR_10
from h_test_IQM.datasets.abstract_dataset import generic_loader


class CIFAR_10_LOADER(generic_loader):
    '''
    Generic data loader for CIFAR-10
    Args:
        indicies_to_use: list of in the dataset to use if you require to use a split of the dataset
        image_dict: dictionary of pre loaded images filename as keys and torch tensor as values 
    '''
    # change downloaddata, setvars, _get_image in child class to use this data template
    def download_data(self):
        download_CIFAR_10()

    def set_vars(self):
        self.csv_file = CIFAR_10_META_CSV
        self.image_dir = CIFAR_10_IMAGE_DIR
        self.dataset_name = 'CIFAR-10'
        self.num_classes = 10


if __name__ == '__main__':
    from tqdm import tqdm
    a = CIFAR_10_LOADER()
    ims = a.get_images_dict()

    b = CIFAR_10_LOADER(image_dict=ims)

    # benchmark
    for e in tqdm(range(10)):
        for i in tqdm(b, leave=False):
            pass