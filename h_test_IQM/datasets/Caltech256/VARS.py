import os

file_path = os.path.dirname(os.path.abspath(__file__))
Caltech256_RAW_DATA_DIR = os.path.join(file_path, 'raw_data')
Caltech256_META_CSV = os.path.join(Caltech256_RAW_DATA_DIR, 'meta_data.csv')
Caltech256_IMAGE_DIR = os.path.join(Caltech256_RAW_DATA_DIR, 'images')
Caltech256_LABELS = os.path.join(Caltech256_RAW_DATA_DIR, 'unique_labels.json')
