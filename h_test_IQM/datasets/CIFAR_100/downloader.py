import os
import shutil
import pandas as pd
from tqdm import tqdm
import json
import torchvision
from h_test_IQM.datasets.CIFAR_100.VARS import CIFAR_100_RAW_DATA_DIR, CIFAR_100_META_CSV, CIFAR_100_IMAGE_DIR, CIFAR_100_LABELS


def download_CIFAR_100(redo_download=False):
    if redo_download == True:
        if os.path.exists(CIFAR_100_RAW_DATA_DIR):
            shutil.rmtree(CIFAR_100_RAW_DATA_DIR)
    
    if not os.path.exists(CIFAR_100_RAW_DATA_DIR) or not os.path.exists(CIFAR_100_IMAGE_DIR):
        # use torch vision to download CIFAR-10 and unzip
        train = torchvision.datasets.CIFAR100(root=CIFAR_100_RAW_DATA_DIR, train=True,
                                            download=True, transform=None)
        test = torchvision.datasets.CIFAR100(root=CIFAR_100_RAW_DATA_DIR, train=False,
                                            download=True, transform=None)
        # save to png
        save_torch_dataset_as_png([train, test], CIFAR_100_IMAGE_DIR)

    # delete raw files
    torch_folder = os.path.join(CIFAR_100_RAW_DATA_DIR, 'cifar-100-python')
    if os.path.exists(torch_folder):
        shutil.rmtree(torch_folder)
    tarfile = os.path.join(CIFAR_100_RAW_DATA_DIR, 'cifar-100-python.tar.gz')
    if os.path.exists(tarfile):
        os.remove(tarfile)


def save_torch_dataset_as_png(datasets, ims_dir):
    file_names = []
    labels = []
    one_hot_labels = []
    os.makedirs(ims_dir, exist_ok=True)
    count = 0
    for dataset in datasets:
        name = 'train' if dataset.train else 'test'
        for (im, label) in tqdm(dataset, desc=f'Saving {name} CIFAR_100 to PNG', leave=True, total=len(dataset)): 
            filename = f'{count}.png'
            im.save(os.path.join(ims_dir, filename))

            file_names.append(filename)
            labels.append(dataset.classes[label])
            one_hot_labels.append(label)
            count += 1

    meta_dict = {'filename': file_names, 
                 'label': labels,
                 'numerical_label': one_hot_labels}
    
    df = pd.DataFrame.from_dict(meta_dict)
    df.to_csv(CIFAR_100_META_CSV, index=False, header=list(meta_dict.keys()))

    # save unique labels
    unique_labels = list(set(labels))
    unique_labels.sort()
    unique_one_hot_labels = list(set(one_hot_labels))
    unique_one_hot_labels.sort()
    with open(CIFAR_100_LABELS, 'w') as f:
        json.dump({'labels':unique_labels, 'numerical':unique_one_hot_labels}, f)


if __name__ == '__main__':
    download_CIFAR_100(redo_download=True)

