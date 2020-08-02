import tensorflow as tf
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import random

from box_utils import compute_target
from image_utils import random_patching, horizontal_flip
from functools import partial

import tensorflow_datasets as tfds



class Dataset:
    """ Class for TFDS Dataset

    Attributes:
        data_dir: dataset data dir (ex: '/data/tensorflow_datasets')
        num_examples: number of examples to be used
                      (in case one wants to overfit small data)
    """

    def __init__(self, data_dir, default_boxes,
                 new_size, num_examples=-1, augmentation=None):
        super(VOCDataset, self).__init__()
        self.idx_to_name = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
        self.name_to_idx = dict([(v, k)
                                 for k, v in enumerate(self.idx_to_name)])

        self.data = tfds.load(
            'voc', split='train',
            data_dir='/data/tensorflow_datasets',
            shuffle_files=True)
        
        self.data = self.clean_data(self.data)
        self.data = self.scaleup_bbox(self.data)

        self.default_boxes = default_boxes
        self.new_size = new_size

        if augmentation == None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']

    
    @tf.function
    def clean_data(data):
        image = data['image']
        filename = data['image/filename']
        labels = data['labels']
        bbox = data['objects']['bbox']
        xy_bbox = tf.concat((bbox[..., -3::-1], bbox[..., -1:-3:-1]), 1)
        
        return filename, image, labels, xy_bbox


    @tf.function
    def scaleup_bbox(filename, image, labels, gt_bbox):
        shape = tf.cast(tf.shape(image), tf.float32)
        # tf.print('shape', shape)
        
        scale_value = tf.concat((shape[:2], shape[:2]), 0, 'scale_value')
        # tf.print('scale value', scale_value)
        
        scaled_bbox = gt_bbox * scale_value
        
        return filename, image, scaled_bbox, labels
        
        

    def _get_annotation(self, index, orig_shape):
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1

        Args:
            index: the index to get filename from self.ids
            orig_shape: image's original shape

        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        h, w = orig_shape
        filename = self.ids[index]
        anno_path = os.path.join(self.anno_dir, filename + '.xml')
        objects = ET.parse(anno_path).findall('object')
        boxes = []
        labels = []

        for obj in objects:
            name = obj.find('name').text.lower().strip()
            bndbox = obj.find('bndbox')
            xmin = (float(bndbox.find('xmin').text) - 1) / w
            ymin = (float(bndbox.find('ymin').text) - 1) / h
            xmax = (float(bndbox.find('xmax').text) - 1) / w
            ymax = (float(bndbox.find('ymax').text) - 1) / h
            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(self.name_to_idx[name] + 1)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def generate_dep(self, subset=None):
        """ The __getitem__ method
            so that the object can be iterable

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        # if subset == 'train':
        #     indices = self.train_ids
        # elif subset == 'val':
        #     indices = self.val_ids
        # else:
        #     indices = self.ids

        for index in range(len(indices)):
            # img, orig_shape = self._get_image(index)
            filename = indices[index]
            img = self._get_image(index)
            w, h = img.size
            boxes, labels = self._get_annotation(index, (h, w))
            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)


            img = np.array(img.resize(
                (self.new_size, self.new_size)), dtype=np.float32)
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)

            gt_confs, gt_locs = compute_target(
                self.default_boxes, boxes, labels)

            yield filename, img, gt_confs, gt_locs

    def generate(self):
        data = self.data
        # resize image
        # rescale image
        # image type float32


def create_batch_generator(data_dir, default_boxes,
                           new_size, batch_size, num_batches,
                           mode,
                           augmentation=None):
    num_examples = batch_size * num_batches if num_batches > 0 else -1
    voc = VOCDataset(root_dir, year, default_boxes,
                     new_size, num_examples, augmentation)

    info = {
        'idx_to_name': voc.idx_to_name,
        'name_to_idx': voc.name_to_idx,
        'length': len(voc),
        'image_dir': voc.image_dir,
        'anno_dir': voc.anno_dir
    }

    if mode == 'train':
        train_gen = partial(voc.generate, subset='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32))
        val_gen = partial(voc.generate, subset='val')
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32))

        train_dataset = train_dataset.shuffle(40).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.take(num_batches), val_dataset.take(-1), info
    else:
        dataset = tf.data.Dataset.from_generator(
            voc.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset.take(num_batches), info
