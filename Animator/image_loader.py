import cv2
import os


def load_image(im_path):
    im = cv2.imread(im_path)
    return im


def load_bboxes(dir_path):
    """
    data loader for a directory with cropped keyframes
    :param dir_path: the full path to the directory
    :return: dictionary from bbox name to numpy array
    """
    if not os.path.isdir(dir_path):
        raise Exception('Attempt to load images from directory while given a file and not a directory.')
    images = os.listdir(dir_path)
    bboxes = {}
    for im in images:
        im_path = os.path.join(dir_path, '', im)
        bboxes[im] = load_image(im_path)
    return bboxes
