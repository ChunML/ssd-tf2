import argparse
import tensorflow as tf
import os
import sys
import time
import yaml

from data import create_batch_generator
from anchor import generate_default_boxes
from network import create_ssd
from losses import create_losses


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../dataset')
parser.add_argument('--data-year', default='2007')
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--num-batches', default=-1, type=int)
parser.add_argument('--neg-ratio', default=3, type=int)
parser.add_argument('--initial-lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--num-epochs', default=120, type=int)
parser.add_argument('--checkpoint-dir', default='checkpoints')
parser.add_argument('--pretrained-type', default='base')
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

NUM_CLASSES = 21


@tf.function
def train_step(imgs, gt_confs, gt_locs, ssd, criterion, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = ssd(imgs)

        conf_loss, loc_loss = criterion(
            confs, locs, gt_confs, gt_locs)

        loss = conf_loss + loc_loss

    gradients = tape.gradient(loss, ssd.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))

    return loss, conf_loss, loc_loss


if __name__ == '__main__':
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open('./config.yml') as f:
        cfg = yaml.load(f)

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    default_boxes = generate_default_boxes(config)

    batch_generator, info = create_batch_generator(
        args.data_dir, args.data_year, default_boxes,
        config['image_size'],
        args.batch_size, args.num_batches,
        do_shuffle=False, augmentation=['flip'])  # the patching algorithm is currently causing bottleneck sometimes
    
    try:
        ssd = create_ssd(NUM_CLASSES, args.arch,
                        args.pretrained_type,
                        checkpoint_dir=args.checkpoint_dir)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    criterion = create_losses(args.neg_ratio, NUM_CLASSES)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.initial_lr,
        momentum=args.momentum, decay=args.weight_decay)

    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()
        for i, (imgs, gt_confs, gt_locs) in enumerate(batch_generator):
            loss, conf_loss, loc_loss = train_step(
                imgs, gt_confs, gt_locs, ssd, criterion, optimizer)
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
            if (i + 1) % 50 == 0:
                print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                    epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))

        if (epoch + 1) % 10 == 0:
            ssd.save_weights(
                os.path.join(args.checkpoint_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))
