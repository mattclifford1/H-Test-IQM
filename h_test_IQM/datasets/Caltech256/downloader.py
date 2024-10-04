import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision
from h_test_IQM.datasets.Caltech256.VARS import Caltech256_RAW_DATA_DIR, Caltech256_META_CSV, Caltech256_IMAGE_DIR


def download_Caltech256(redo_download=False):
    if redo_download == True:
        shutil.rmtree(Caltech256_RAW_DATA_DIR)
    
    if not os.path.exists(Caltech256_RAW_DATA_DIR) or not os.path.exists(Caltech256_IMAGE_DIR):
        # use torch vision to download CIFAR-10 and unzip
        dataset = torchvision.datasets.Caltech256(root=Caltech256_RAW_DATA_DIR,
                                            download=True, transform=None)
        # save to png
        save_torch_dataset_as_png([dataset], Caltech256_IMAGE_DIR)

    # delete raw files
    torch_folder = os.path.join(Caltech256_RAW_DATA_DIR, 'Caltech256')
    if os.path.exists(torch_folder):
        shutil.rmtree(torch_folder)


def save_torch_dataset_as_png(datasets, ims_dir):
    file_names = []
    labels = []
    one_hot_labels = []
    os.makedirs(ims_dir, exist_ok=True)
    count = 0
    for dataset in datasets:
        name = 'train' if count == 0 else 'test'
        for i, (im, label) in tqdm(enumerate(dataset), desc=f'Saving {name} Caltech256 to PNG', leave=True, total=len(dataset)): 
            filename = f'{i}.png'
            im.save(os.path.join(ims_dir, filename))

            file_names.append(filename)
            labels.append(str(label))
            one_hot_labels.append(label)
            count += 1

    meta_dict = {'filename': file_names, 
                 'label': labels,
                 'numerical_label': one_hot_labels}
    df = pd.DataFrame.from_dict(meta_dict)
    df.to_csv(Caltech256_META_CSV, index=False, header=list(meta_dict.keys()))


if __name__ == '__main__':
    download_Caltech256(redo_download=True)

