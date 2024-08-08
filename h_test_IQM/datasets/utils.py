import os
import cv2
import numpy as np
import PIL.Image

def load_image(image_path):
    '''
    load image as RGB float
    '''
    if not os.path.isfile(image_path):
        raise ValueError(f'Image file: {image_path} does not exist')
    img = PIL.Image.open(image_path)
    img = np.array(img)
    if img.dtype == 'uint8':
        img = img/255
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def resize_to_longest_side(im, side=128):
    '''
    resize image to longest side
    '''
    shape = im.shape
    if shape[0] > shape[1]:
        scale = side/shape[0]
        size = (side, int(shape[1]*scale))
    else:
        scale = side/shape[1]
        size = (int(shape[0]*scale), side)
    im = resize_image(im, size)
    return im


def resize_image(img, size=128):
    '''
    resize image to square or specified size
    '''
    if isinstance(size, int):
        size = (size, size)
    # cv2 much faster than skimage
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    # can change to PIL code below but only works with uint8 images
    # img = PIL.Image.fromarray(img_as_ubyte(img))
    # img = img.resize(size)
    # img = np.array(img)/255

    return img
