from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import os

from layers import create_vgg16_layers, create_extra_layers, create_conf_head_layers, create_loc_head_layers


class SSD(Model):
    """ Class for SSD model
    Attributes:
        num_classes: number of classes
    """

    def __init__(self, num_classes, arch='ssd300'):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.vgg16_conv4, self.vgg16_conv7 = create_vgg16_layers()
        self.batch_norm = layers.BatchNormalization(
            beta_initializer='glorot_uniform',
            gamma_initializer='glorot_uniform'
        )
        self.extra_layers = create_extra_layers()
        self.conf_head_layers = create_conf_head_layers(num_classes)
        self.loc_head_layers = create_loc_head_layers()

        if arch == 'ssd300':
            self.extra_layers.pop(-1)
            self.conf_head_layers.pop(-2)
            self.loc_head_layers.pop(-2)

    def compute_heads(self, x, idx):
        """ Compute outputs of classification and regression heads
        Args:
            x: the input feature map
            idx: index of the head layer
        Returns:
            conf: output of the idx-th classification head
            loc: output of the idx-th regression head
        """
        conf = self.conf_head_layers[idx](x)
        conf = tf.reshape(conf, [conf.shape[0], -1, self.num_classes])

        loc = self.loc_head_layers[idx](x)
        loc = tf.reshape(loc, [loc.shape[0], -1, 4])

        return conf, loc

    def init_vgg16(self):
        """ Initialize the VGG16 layers from pretrained weights
            and the rest from scratch using xavier initializer
        """
        origin_vgg = VGG16(weights='imagenet')
        for i in range(len(self.vgg16_conv4.layers)):
            self.vgg16_conv4.get_layer(index=i).set_weights(
                origin_vgg.get_layer(index=i).get_weights())

        fc1_weights, fc1_biases = origin_vgg.get_layer(index=-3).get_weights()
        fc2_weights, fc2_biases = origin_vgg.get_layer(index=-2).get_weights()

        conv6_weights = np.random.choice(
            np.reshape(fc1_weights, (-1,)), (3, 3, 512, 1024))
        conv6_biases = np.random.choice(
            fc1_biases, (1024,))

        conv7_weights = np.random.choice(
            np.reshape(fc2_weights, (-1,)), (1, 1, 1024, 1024))
        conv7_biases = np.random.choice(
            fc2_biases, (1024,))

        self.vgg16_conv7.get_layer(index=2).set_weights(
            [conv6_weights, conv6_biases])
        self.vgg16_conv7.get_layer(index=3).set_weights(
            [conv7_weights, conv7_biases])

    def call(self, x):
        """ The forward pass
        Args:
            x: the input image
        Returns:
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        confs = []
        locs = []
        head_idx = 0
        for i in range(len(self.vgg16_conv4.layers)):
            x = self.vgg16_conv4.get_layer(index=i)(x)
            if i == len(self.vgg16_conv4.layers) - 5:
                conf, loc = self.compute_heads(self.batch_norm(x), head_idx)
                confs.append(conf)
                locs.append(loc)
                head_idx += 1

        x = self.vgg16_conv7(x)

        conf, loc = self.compute_heads(x, head_idx)

        confs.append(conf)
        locs.append(loc)
        head_idx += 1

        for layer in self.extra_layers:
            x = layer(x)
            conf, loc = self.compute_heads(x, head_idx)
            confs.append(conf)
            locs.append(loc)
            head_idx += 1

        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)

        return confs, locs


def create_ssd(num_classes, arch, pretrained_type,
               checkpoint_dir=None,
               checkpoint_path=None):
    """ Create SSD model and load pretrained weights
    Args:
        num_classes: number of classes
        pretrained_type: type of pretrained weights, can be either 'VGG16' or 'ssd'
        weight_path: path to pretrained weights
    Returns:
        net: the SSD model
    """
    net = SSD(num_classes, arch)
    net(tf.random.normal((1, 512, 512, 3)))
    if pretrained_type == 'base':
        net.init_vgg16()
    elif pretrained_type == 'latest':
        try:
            paths = [os.path.join(checkpoint_dir, path)
                     for path in os.listdir(checkpoint_dir)]
            latest = sorted(paths, key=os.path.getmtime)[-1]
            net.load_weights(latest)
        except AttributeError as e:
            print('Please make sure there is at least one checkpoint at {}'.format(
                checkpoint_dir))
            print('The model will be loaded from base weights.')
            net.init_vgg16()
        except ValueError as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    latest, arch))
        except Exception as e:
            print(e)
            raise ValueError('Please check if checkpoint_dir is specified')
    elif pretrained_type == 'specified':
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                'Not a valid checkpoint file: {}'.format(checkpoint_path))

        try:
            net.load_weights(checkpoint_path)
        except Exception as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    checkpoint_path, arch))
    else:
        raise ValueError('Unknown pretrained type: {}'.format(pretrained_type))
    return net

