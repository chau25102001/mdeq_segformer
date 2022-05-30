from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools
from termcolor import colored

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from models.mdeq_with_segformer_attention import MDEQNet
from lib.layer_utils import conv3x3

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        A bottleneck block with receptive field only 3x3. (This is not used in MDEQ; only
        in the classifier layer).
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, injection=None):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x) + injection
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class MDEQSegNet(MDEQNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Segmentation model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQSegNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)

        # Last layer
        last_inp_channels = self.num_channels[0]
        self.last_layer = nn.Sequential(nn.ConvTranspose2d(last_inp_channels, last_inp_channels, kernel_size=2, stride = 2, padding = 0),
                                        nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True),
                                        nn.ConvTranspose2d(last_inp_channels, cfg.DATASET.NUM_CLASSES,
                                                  kernel_size=2,
                                                  stride=2, padding=0))

    def segment(self, y):
        """
        Given outputs at multiple resolutions, segment the feature map by predicting the class of each pixel
        """
        # # Segmentation Head
        # y0_h, y0_w = y[0].size(2), y[0].size(3)
        # all_res = [y[0]]
        # for i in range(1, self.num_branches):
        #     all_res.append(F.interpolate(y[i], size=(y0_h, y0_w), mode='bilinear', align_corners=True))
        #
        # y = torch.cat(all_res, dim=1)
        # all_res = None
        y = self.last_layer(y)
        return y

    def forward(self, x, train_step=0, **kwargs):
        y, jac_loss, sradius = self._forward(x, train_step, **kwargs)
        return self.segment(y), jac_loss, sradius

    def init_weights(self, pretrained=''):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info(f'=> init weights from normal distribution. PRETRAINED={pretrained}')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()

            # Just verification...
            diff_modules = set()
            for k in pretrained_dict.keys():
                if k not in model_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In ImageNet MDEQ but not Cityscapes MDEQ: {sorted(list(diff_modules))}", "red"))
            diff_modules = set()
            for k in model_dict.keys():
                if k not in pretrained_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In Cityscapes MDEQ but not ImageNet MDEQ: {sorted(list(diff_modules))}", "green"))

            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_seg_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.01
    model = MDEQSegNet(config, **kwargs)
    model.init_weights(config.MODEL.PRETRAINED)
    return model

#
# if __name__ == "__main__":
#     import warnings
#     import yaml
#     import json
#
#     warnings.filterwarnings('ignore')
#     class obj:
#         def __init__(self, dict1):
#             self.__dict__.update(dict1)
#
#         def __getitem__(self, item):
#             return self.__dict__[item]
#
#
#     config = yaml.load(open("/home/chau/PycharmProjects/mdeq_new/experiments/segfomer_mdeq_thaysang.yaml", "r"), Loader=yaml.FullLoader)
#     config = json.loads(json.dumps(config), object_hook=obj)
#
#     model = get_seg_net(config)
#
#     image = torch.ones((1,3,352,352))
#     preds, _, _= model(image, train_step = -1)
#     print(preds.shape)
#
