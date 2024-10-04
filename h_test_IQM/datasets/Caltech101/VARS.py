import os

file_path = os.path.dirname(os.path.abspath(__file__))
Caltech101_RAW_DATA_DIR = os.path.join(file_path, 'raw_data')
Caltech101_META_CSV = os.path.join(Caltech101_RAW_DATA_DIR, 'meta_data.csv')
Caltech101_IMAGE_DIR = os.path.join(Caltech101_RAW_DATA_DIR, 'images')
Caltech101_LABELS = os.path.join(Caltech101_RAW_DATA_DIR, 'unique_labels.json')
