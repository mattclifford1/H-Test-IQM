import os

file_path = os.path.dirname(os.path.abspath(__file__))
CIFAR_100_RAW_DATA_DIR = os.path.join(file_path, 'raw_data')
CIFAR_100_META_CSV = os.path.join(CIFAR_100_RAW_DATA_DIR, 'meta_data.csv')
CIFAR_100_LABELS = os.path.join(CIFAR_100_RAW_DATA_DIR, 'unique_labels.json')
CIFAR_100_IMAGE_DIR = os.path.join(CIFAR_100_RAW_DATA_DIR, 'images')
