import os

file_path = os.path.dirname(os.path.abspath(__file__))
DTD_RAW_DATA_DIR = os.path.join(file_path, 'raw_data')
DTD_META_CSV = os.path.join(DTD_RAW_DATA_DIR, 'meta_data.csv')
DTD_LABELS = os.path.join(DTD_RAW_DATA_DIR, 'unique_labels.json')
DTD_IMAGE_DIR = os.path.join(DTD_RAW_DATA_DIR, 'images')
