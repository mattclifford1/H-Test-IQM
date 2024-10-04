import os
import shutil
import pandas as pd
from tqdm import tqdm
import torchvision
import json
from h_test_IQM.datasets.Caltech256.VARS import Caltech256_RAW_DATA_DIR, Caltech256_META_CSV, Caltech256_IMAGE_DIR, Caltech256_LABELS


def download_Caltech256(redo_download=False):
    if redo_download == True:
        shutil.rmtree(Caltech256_RAW_DATA_DIR)
    
    if not os.path.exists(Caltech256_RAW_DATA_DIR) or not os.path.exists(Caltech256_IMAGE_DIR):
        # use torch vision to download CIFAR-10 and unzip
        dataset = torchvision.datasets.Caltech256(root=Caltech256_RAW_DATA_DIR,
                                            download=True, transform=None)

        make_meta_data(dataset)

    # delete raw files
    torch_folder = os.path.join(Caltech256_RAW_DATA_DIR, 'caltech256')
    if os.path.exists(torch_folder):
        shutil.rmtree(torch_folder)


def make_meta_data(dataset):
    file_names = []
    labels = []
    one_hot_labels = []
    print(len(dataset))
    for i in tqdm(range(len(dataset)), desc=f'Making Caltech256 to meta data'):
        filename = os.path.join(
            Caltech256_IMAGE_DIR, 
            dataset.categories[dataset.y[i]],
            f"{dataset.y[i] + 1:03d}_{dataset.index[i]:04d}.jpg")
        file_names.append(filename)
        num_label = dataset.y[i]
        one_hot_labels.append(num_label)
        labels.append(dataset.categories[num_label][4:])

    meta_dict = {'filename': file_names, 
                 'label': labels,
                 'numerical_label': one_hot_labels}
    df = pd.DataFrame.from_dict(meta_dict)
    df.to_csv(Caltech256_META_CSV, index=False, header=list(meta_dict.keys()))

    # move image folder
    shutil.move(os.path.join(Caltech256_RAW_DATA_DIR, 'caltech256', '256_ObjectCategories'),
                Caltech256_IMAGE_DIR)
    
    # save unique labels
    unique_labels = list(set(labels))
    unique_labels.sort()
    unique_one_hot_labels = list(set(one_hot_labels))
    unique_one_hot_labels.sort()
    with open(Caltech256_LABELS, 'w') as f:
        json.dump({'labels':unique_labels, 'numerical':unique_one_hot_labels}, f)


if __name__ == '__main__':
    download_Caltech256(redo_download=True)

