import os

file_path = os.path.dirname(os.path.abspath(__file__))
MNIST_RAW_DATA_DIR = os.path.join(file_path, 'raw_data')
MNIST_META_CSV = os.path.join(MNIST_RAW_DATA_DIR, 'meta_data.csv')
MNIST_LABELS = os.path.join(MNIST_RAW_DATA_DIR, 'unique_labels.json')
MNIST_IMAGE_DIR = os.path.join(MNIST_RAW_DATA_DIR, 'images')
