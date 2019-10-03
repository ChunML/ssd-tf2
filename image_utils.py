import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import tensorflow as tf

from box_utils import compute_iou


class ImageVisualizer(object):
    """ Class for visualizing image

    Attributes:
        idx_to_name: list to convert integer to string label
        class_colors: colors for drawing boxes and labels
        save_dir: directory to store images
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = './'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def save_image(self, img, boxes, labels, name):
        """ Method to draw boxes and labels
            then save to dir

        Args:
            img: numpy array (width, height, 3)
            boxes: numpy array (num_boxes, 4)
            labels: numpy array (num_boxes)
            name: name of image to be saved
        """
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        save_path = os.path.join(self.save_dir, name)

        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            top_left = (box[0], box[1])
            bot_right = (box[2], box[3])
            ax.add_patch(patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=(0., 1., 0.),
                facecolor="none"))
            plt.text(
                box[0],
                box[1],
                s=cls_name,
                color="white",
                verticalalignment="top",
                bbox={"color": (0., 1., 0.), "pad": 0},
            )

        plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')


def generate_patch(boxes, threshold):
    """ Function to generate a random patch within the image
        If the patch overlaps any gt boxes at above the threshold,
        then the patch is picked, otherwise generate another patch

    Args:
        boxes: box tensor (num_boxes, 4)
        threshold: iou threshold to decide whether to choose the patch

    Returns:
        patch: the picked patch
        ious: an array to store IOUs of the patch and all gt boxes
    """
    while True:
        patch_w = random.uniform(0.1, 1)
        scale = random.uniform(0.5, 2)
        patch_h = patch_w * scale
        patch_xmin = random.uniform(0, 1 - patch_w)
        patch_ymin = random.uniform(0, 1 - patch_h)
        patch_xmax = patch_xmin + patch_w
        patch_ymax = patch_ymin + patch_h
        patch = np.array(
            [[patch_xmin, patch_ymin, patch_xmax, patch_ymax]],
            dtype=np.float32)
        patch = np.clip(patch, 0.0, 1.0)
        ious = compute_iou(tf.constant(patch), boxes)
        if tf.math.reduce_any(ious >= threshold):
            break

    return patch[0], ious[0]


def random_patching(img, boxes, labels):
    """ Function to apply random patching
        Firstly, a patch is randomly picked
        Then only gt boxes of which IOU with the patch is above a threshold
        and has center point lies within the patch will be selected

    Args:
        img: the original PIL Image
        boxes: gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)

    Returns:
        img: the cropped PIL Image
        boxes: selected gt boxes tensor (new_num_boxes, 4)
        labels: selected gt labels tensor (new_num_boxes,)
    """
    threshold = np.random.choice(np.linspace(0.1, 0.7, 4))

    patch, ious = generate_patch(boxes, threshold)

    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    keep_idx = (
        (ious > 0.3) &
        (box_centers[:, 0] > patch[0]) &
        (box_centers[:, 1] > patch[1]) &
        (box_centers[:, 0] < patch[2]) &
        (box_centers[:, 1] < patch[3])
    )

    if not tf.math.reduce_any(keep_idx):
        return img, boxes, labels

    img = img.crop(patch)

    boxes = boxes[keep_idx]
    patch_w = patch[2] - patch[0]
    patch_h = patch[3] - patch[1]
    boxes = tf.stack([
        (boxes[:, 0] - patch[0]) / patch_w,
        (boxes[:, 1] - patch[1]) / patch_h,
        (boxes[:, 2] - patch[0]) / patch_w,
        (boxes[:, 3] - patch[1]) / patch_h], axis=1)
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)

    labels = labels[keep_idx]

    return img, boxes, labels


def horizontal_flip(img, boxes, labels):
    """ Function to horizontally flip the image
        The gt boxes will be need to be modified accordingly

    Args:
        img: the original PIL Image
        boxes: gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)

    Returns:
        img: the horizontally flipped PIL Image
        boxes: horizontally flipped gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)
    """
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    boxes = tf.stack([
        1 - boxes[:, 2],
        boxes[:, 1],
        1 - boxes[:, 0],
        boxes[:, 3]], axis=1)

    return img, boxes, labels
