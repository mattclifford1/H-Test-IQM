from h_test_IQM.datasets.MNIST.VARS import MNIST_META_CSV, MNIST_IMAGE_DIR
from h_test_IQM.datasets.MNIST.downloader import download_MNIST
from h_test_IQM.datasets.abstract_dataset import generic_loader


class MNIST_LOADER(generic_loader):
    '''
    Generic data loader for CIFAR-10
    Args:
        indicies_to_use: list of in the dataset to use if you require to use a split of the dataset
        image_dict: dictionary of pre loaded images filename as keys and torch tensor as values 
    '''
    # change downloaddata, setvars, _get_image in child class to use this data template

    def download_data(self):
        download_MNIST()

    def set_vars(self):
        self.csv_file = MNIST_META_CSV
        self.image_dir = MNIST_IMAGE_DIR
        self.dataset_name = 'MNIST'
        self.num_classes = 10

    def _get_image(self, filename):
        img = super()._get_image(filename)
        # expand to 3 channels
        return img.expand(3, *img.shape[1:])


if __name__ == '__main__':
    from tqdm import tqdm
    a = MNIST_LOADER()
    ims = a.get_images_dict()

    b = MNIST_LOADER(image_dict=ims)

    # benchmark
    for e in tqdm(range(10)):
        for i in tqdm(b, leave=False):
            pass
