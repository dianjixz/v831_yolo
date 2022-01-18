from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
from .widerface import WiderfaceDetection, WiderfaceAnnotationTransform, WIDERFACE_CLASSES, WIDERFACE_ROOT
from .custom import CustomDetection, CustomAnnotationTransform, CUSTOM_CLASSES, CUSTOM_ROOT
from .cocodataset import COCODataset, coco_root, coco_class_labels, coco_class_index
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean, std):
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    # x /= 255.
    # x -= mean
    # x /= std
    # or mean=mean*255
    #    std = 1/(std*255)
    #    x = (x-mean)*std
    x -= mean
    x *= std
    return x


class BaseTransform:
    def __init__(self, size, mean=(127.5, 127.5, 127.5), std=(0.0078125, 0.0078125, 0.0078125)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean, self.std), boxes, labels
