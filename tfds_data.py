import tensorflow as tf
import os
import numpy as np

from box_utils import compute_target
from image_utils import random_patching, horizontal_flip

import tensorflow_datasets as tfds


class Dataset:
    """ Class for TFDS Dataset

    Attributes:
        data_dir: dataset data dir (ex: '/data/tensorflow_datasets')
    """

    def __init__(self, data_dir, default_boxes,
                 new_size, augmentation=None):
        super(Dataset, self).__init__()
        self.idx_to_name = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
        self.name_to_idx = dict([(v, k)
                                 for k, v in enumerate(self.idx_to_name)])

        data = tfds.load(
            'voc', split='train',
            data_dir='/data/tensorflow_datasets',
            shuffle_files=True)

        self.default_boxes = default_boxes
        self.new_size = new_size

        self.data = data.map(self.clean_data)
        # self.data = self.scaleup_bbox(self.data)

        if augmentation == None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']


    @tf.function
    def scaleup_bbox(filename, image, labels, gt_bbox):
        shape = tf.cast(tf.shape(image), tf.float32)
        # tf.print('shape', shape)
        
        scale_value = tf.concat((shape[:2], shape[:2]), 0, 'scale_value')
        # tf.print('scale value', scale_value)
        
        scaled_bbox = gt_bbox * scale_value
        
        return filename, image, scaled_bbox, labels
    

    @tf.function
    def clean_data(self, data):
        image = tf.image.resize(data['image'], self.new_size)
        filename = data['image/filename']
        labels = data['labels']
        bbox = data['objects']['bbox']
        xy_bbox = tf.concat((bbox[..., -3::-1], bbox[..., -1:-3:-1]), 1)
        
        return filename, image, xy_bbox, labels


    @tf.function
    def map_compute_target(self, filename, image, bbox, labels):
        tout = [tf.int64, tf.float32]
        gt_confs, gt_locs = tf.py_function(compute_target, [self.default_boxes, bbox, labels], tout)
            
        return filename, image, gt_confs, gt_locs


    def generate(self):
        """
        Returns:
            filename: filename of image
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        data = self.data
        # rescale image
        
        data = data.map(self.map_compute_target)
        return data


def create_batch_generator(default_boxes,
                           new_size, batch_size, num_batches,
                           augmentation=None, data_dir='/data/tensorflow_datasets'):
    dataset = Dataset(data_dir, default_boxes,
                     new_size=new_size)

    info = {
        'idx_to_name': dataset.idx_to_name,
        'name_to_idx': dataset.name_to_idx,
        # 'length': len(dataset),
        # 'image_dir': dataset.image_dir,
        # 'anno_dir': dataset.anno_dir
    }

    data_gen = dataset.generate()
    if batch_size:
        data_gen = data_gen.batch(batch_size)
    if num_batches:
        return data_gen.take(num_batches), info
    else:
        return data_gen, info
