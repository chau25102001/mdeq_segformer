# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import re
import glob
import torch.nn.functional as F
import numpy as np


class FullModel(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce 
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """

    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs, labels, train_step=-1, **kwargs):
        outputs, jac_loss, sradius = self.model(inputs, train_step=train_step, **kwargs)
        losses = self.loss(outputs, labels)
        return losses, jac_loss.unsqueeze(0), outputs, sradius


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{sep}{n}"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / cfg_name
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD,
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


# def get_iou(label, pred, num_class, smooth=1e-9):
#     size = label.size()
#     assert num_class == 2, "Only support 2 class"
#     out_pred = nn.functional.interpolate(input=pred, size=(size[-2], size[-1]), mode='bilinear', align_corners=True)
#     segment = torch.argmax(out_pred, dim=1)
#     segment = segment.view(-1)
#     label = label.view(-1)
#     TP = (segment & label).sum()
#     FN_FP = (segment | label).sum()
#     return TP / (FN_FP + smooth)

def get_iou(label, pred, num_class, smooth=1e-9):
    size = label.size()
    assert num_class == 2, "Only support 2 class"
    out_pred = nn.functional.interpolate(input=pred, size=(size[-2], size[-1]), mode='bilinear', align_corners=True)
    segment = torch.argmax(out_pred, dim=1)
    # label = label.squeeze(1)
    # segment = torch.where(segment > 0.5, 1, 0)
    TP = torch.sum(segment & label)
    FN_FP = torch.sum(segment | label)

    return TP / (FN_FP + smooth)


def get_dice_score(label, pred, num_class, smooth=1e-9):
    size = label.size()
    assert num_class == 2, "Only support 2 class"
    out_pred = nn.functional.interpolate(input=pred, size=(size[-2], size[-1]), mode='bilinear', align_corners=True)
    segment = torch.argmax(out_pred, dim=1)
    # label = label.squeeze(1)
    # segment = torch.where(segment > 0.5, 1, 0)
    TP = torch.sum(segment & label)
    FN_FP = torch.sum(segment | label)

    return 2 * TP / (TP + FN_FP + smooth)


def iou(label, pred, lis, num_class, smooth=1e-9):
    # print(label.shape, pred.shape)
    # exit()
    # size = label.size()
    pred = F.sigmoid(pred)
    segment = torch.where(pred > 0.5, 1, 0)
    for i, seg in enumerate(segment):
        intersaction = torch.sum((seg>0) & (label[i]>0))
        union = torch.sum((seg>0) | (label[i]>0))
        x = intersaction / (union + smooth)
        lis.append(x.item())
    return lis


def dice(label, pred, lis, num_class, smooth=1e-9):
    pred = F.sigmoid(pred)
    segment = torch.where(pred > 0.5, 1, 0)

    for i, seg in enumerate(segment):
            intersaction = torch.sum((seg>0) & (label[i]>0))
            union = torch.sum((seg>0) | (label[i]>0))
            x = 2 * intersaction / (intersaction + union + smooth)
            lis.append(x.item())
    return lis


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

if __name__ == '__main__':
    a = torch.randn((10, 1, 10, 10))
    b = torch.argmax(torch.randn((10, 2, 10, 10)), dim = 1)
    ls = []
    print(iou(b, a, ls, 2))