import os
import numpy as np
import xml.etree.ElementTree as ET
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../dataset')
parser.add_argument('--data-year', default='2007')
parser.add_argument('--detect-dir', default='./outputs/detects')
parser.add_argument('--use-07-metric', type=bool, default=False)
args = parser.parse_args()


def get_annotation(anno_file):
    tree = ET.parse(anno_file)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def compute_ap(rec, prec, ap, use_07_metric=False):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mprec = np.concatenate(([0.0], prec, [0.0]))

        for i in range(mprec.size - 1, 0, -1):
            mprec[i - 1] = np.maximum(mprec[i - 1], mprec[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mprec[i + 1])

    return ap


def voc_eval(det_path, anno_path, cls_name, iou_thresh=0.5, use_07_metric=False):
    det_file = det_path.format(cls_name)
    with open(det_file, 'r') as f:
        lines = f.readlines()

    lines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in lines]
    confs = np.array([float(x[1]) for x in lines])
    boxes = np.array([[float(z) for z in x[2:]] for x in lines])

    gts = {}
    cls_gts = {}
    npos = 0
    for image_id in image_ids:
        if image_id in cls_gts.keys():
            continue
        gts[image_id] = get_annotation(anno_path.format(image_id))
        R = [obj for obj in gts[image_id] if obj['name'] == cls_name]
        gt_boxes = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        cls_gts[image_id] = {
            'gt_boxes': gt_boxes,
            'difficult': difficult,
            'det': det
        }

    sorted_ids = np.argsort(-confs)
    sorted_scores = np.sort(-confs)
    boxes = boxes[sorted_ids, :]
    image_ids = [image_ids[x] for x in sorted_ids]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = cls_gts[image_ids[d]]
        box = boxes[d, :].astype(float)
        iou_max = -np.inf
        gt_box = R['gt_boxes'].astype(float)

        if gt_box.size > 0:
            ixmin = np.maximum(gt_box[:, 0], box[0])
            ixmax = np.minimum(gt_box[:, 2], box[2])
            iymin = np.maximum(gt_box[:, 1], box[1])
            iymax = np.minimum(gt_box[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            uni = ((box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0) +
                   (gt_box[:, 2] - gt_box[:, 0] + 1.0) *
                   (gt_box[:, 3] - gt_box[:, 1] + 1.0) - inters)

            ious = inters / uni
            iou_max = np.max(ious)
            jmax = np.argmax(ious)

        if iou_max > iou_thresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.0
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recall = tp / float(npos)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = compute_ap(recall, precision, use_07_metric)

    return recall, precision, ap


if __name__ == '__main__':
    aps = {
        'aeroplane': 0.0,
        'bicycle': 0.0,
        'bird': 0.0,
        'boat': 0.0,
        'bottle': 0.0,
        'bus': 0.0,
        'car': 0.0,
        'cat': 0.0,
        'chair': 0.0,
        'cow': 0.0,
        'diningtable': 0.0,
        'dog': 0.0,
        'horse': 0.0,
        'motorbike': 0.0,
        'person': 0.0,
        'pottedplant': 0.0,
        'sheep': 0.0,
        'sofa': 0.0,
        'train': 0.0,
        'tvmonitor': 0.0,
        'mAP': []
    }
    for cls_name in aps.keys():
        det_path = os.path.join(args.detect_dir, '{}.txt')
        anno_path = os.path.join(
            args.data_dir, 'VOC{}'.format(args.data_year), 'Annotations', '{}.xml')
        if os.path.exists(det_path.format(cls_name)):
            recall, precision, ap = voc_eval(
                det_path,
                anno_path,
                cls_name,
                use_07_metric=args.use_07_metric)
            aps[cls_name] = ap
            aps['mAP'].append(ap)

    aps['mAP'] = np.mean(aps['mAP'])
    for key, value in aps.items():
        print('{}: {}'.format(key, value))
