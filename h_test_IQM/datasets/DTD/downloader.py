import os
import shutil
import pandas as pd
from tqdm import tqdm
import torchvision
import json
from h_test_IQM.datasets.DTD.VARS import DTD_RAW_DATA_DIR, DTD_META_CSV, DTD_IMAGE_DIR, DTD_LABELS


def download_DTD(redo_download=False):
    if redo_download == True:
        if os.path.exists(DTD_RAW_DATA_DIR):
            shutil.rmtree(DTD_RAW_DATA_DIR)
    
    if not os.path.exists(DTD_RAW_DATA_DIR) or not os.path.exists(DTD_IMAGE_DIR):
        # use torch vision to download CIFAR-10 and unzip
        train = torchvision.datasets.DTD(root=DTD_RAW_DATA_DIR, split='train',
                                            download=True, transform=None)
        val = torchvision.datasets.DTD(root=DTD_RAW_DATA_DIR, split='val',
                                            download=True, transform=None)
        test = torchvision.datasets.DTD(root=DTD_RAW_DATA_DIR, split='test',
                                            download=True, transform=None)

        make_meta_data([train, val, test])

    # delete raw files
    torch_folder = os.path.join(DTD_RAW_DATA_DIR, 'dtd')
    if os.path.exists(torch_folder):
        shutil.rmtree(torch_folder)


def make_meta_data(datasets):
    file_names = []
    labels = []
    one_hot_labels = []
    for dataset in datasets:
        for i in tqdm(range(len(dataset)), desc=f'Making DTD {dataset._split} to meta data'):
            filename = os.path.join(
                DTD_IMAGE_DIR, 
                dataset._image_files[i].parent.name,
                dataset._image_files[i].name)
            file_names.append(filename)
            num_label = dataset._labels[i]
            one_hot_labels.append(num_label)
            labels.append(dataset.classes[num_label])

    meta_dict = {'filename': file_names, 
                 'label': labels,
                 'numerical_label': one_hot_labels}
    df = pd.DataFrame.from_dict(meta_dict)
    df.to_csv(DTD_META_CSV, index=False, header=list(meta_dict.keys()))

    # move image folder
    shutil.move(os.path.join(DTD_RAW_DATA_DIR, 'dtd', 'dtd', 'images'),
                DTD_IMAGE_DIR)
    
    # save unique labels
    unique_labels = list(set(labels))
    unique_labels.sort()
    unique_one_hot_labels = list(set(one_hot_labels))
    unique_one_hot_labels.sort()
    with open(DTD_LABELS, 'w') as f:
        json.dump({'labels':unique_labels, 'numerical':unique_one_hot_labels}, f)
    print(f'Number of classes: {len(unique_labels)}')


if __name__ == '__main__':
    download_DTD(redo_download=True)

