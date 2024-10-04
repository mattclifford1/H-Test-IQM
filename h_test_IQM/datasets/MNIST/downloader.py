import os
import shutil
import pandas as pd
from tqdm import tqdm
import json
import torchvision
from h_test_IQM.datasets.MNIST.VARS import MNIST_RAW_DATA_DIR, MNIST_META_CSV, MNIST_IMAGE_DIR, MNIST_LABELS


def download_MNIST(redo_download=False):
    if redo_download == True:
        shutil.rmtree(MNIST_RAW_DATA_DIR)
    
    if not os.path.exists(MNIST_RAW_DATA_DIR) or not os.path.exists(MNIST_IMAGE_DIR):
        # use torch vision to download CIFAR-10 and unzip
        train = torchvision.datasets.MNIST(root=MNIST_RAW_DATA_DIR, train=True,
                                            download=True, transform=None)
        test = torchvision.datasets.MNIST(root=MNIST_RAW_DATA_DIR, train=False,
                                            download=True, transform=None)
        # save to png
        save_torch_dataset_as_png([train, test], MNIST_IMAGE_DIR)

    # delete raw files
    torch_folder = os.path.join(MNIST_RAW_DATA_DIR, 'MNIST')
    if os.path.exists(torch_folder):
        shutil.rmtree(torch_folder)


def save_torch_dataset_as_png(datasets, ims_dir):
    file_names = []
    labels = []
    one_hot_labels = []
    os.makedirs(ims_dir, exist_ok=True)
    count = 0
    for dataset in datasets:
        name = 'train' if dataset.train else 'test'
        for (im, label) in tqdm(dataset, desc=f'Saving {name} MNIST to PNG', leave=True, total=len(dataset)): 
            filename = f'{count}.png'
            im.save(os.path.join(ims_dir, filename))

            file_names.append(filename)
            labels.append(str(label))
            one_hot_labels.append(label)
            count += 1

    meta_dict = {'filename': file_names, 
                 'label': labels,
                 'numerical_label': one_hot_labels}
    
    df = pd.DataFrame.from_dict(meta_dict)
    df.to_csv(MNIST_META_CSV, index=False, header=list(meta_dict.keys()))

    # save unique labels
    unique_labels = list(set(labels))
    unique_labels.sort()
    unique_one_hot_labels = list(set(one_hot_labels))
    unique_one_hot_labels.sort()
    with open(MNIST_LABELS, 'w') as f:
        json.dump({'labels':unique_labels, 'numerical':unique_one_hot_labels}, f)


if __name__ == '__main__':
    download_MNIST(redo_download=True)

