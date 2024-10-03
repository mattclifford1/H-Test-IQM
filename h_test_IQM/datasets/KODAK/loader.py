import os
import glob
from h_test_IQM.datasets.utils.utils import load_image


def kodak_loader(**kwargs):
    file_path = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(file_path, 'raw_data')
    image_list = glob.glob(os.path.join(dir, '*'))
    # remove folders
    image_list = [f for f in image_list if os.path.isfile(f)]
    image_list.sort()
    ims = []
    for img_path in image_list:
        ims.append(load_image(img_path))
    return ims
