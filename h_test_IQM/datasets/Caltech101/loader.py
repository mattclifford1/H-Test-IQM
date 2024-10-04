from torchvision.transforms.v2 import Resize as Resize_v2
from h_test_IQM.datasets.Caltech101.VARS import Caltech101_META_CSV, Caltech101_IMAGE_DIR
from h_test_IQM.datasets.Caltech101.downloader import download_Caltech101
from h_test_IQM.datasets.abstract_dataset import generic_loader



class Caltech101_LOADER(generic_loader):
    '''
    Generic data loader for CIFAR-10
    Args:
        indicies_to_use: list of in the dataset to use if you require to use a split of the dataset
        image_dict: dictionary of pre loaded images filename as keys and torch tensor as values 
    '''
    def __init__(self, *args, im_size=(256, 256), **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = Resize_v2(im_size)

    def download_data(self):
        download_Caltech101()

    def set_vars(self):
        self.csv_file = Caltech101_META_CSV
        self.image_dir = Caltech101_IMAGE_DIR
        self.dataset_name = 'Caltech101'
        self.num_classes = 101


if __name__ == '__main__':
    from tqdm import tqdm
    a = Caltech101_LOADER()
    ims = a.get_images_dict()

    b = Caltech101_LOADER(image_dict=ims)

    # benchmark
    for e in tqdm(range(10)):
        for i in tqdm(b, leave=False):
            pass
