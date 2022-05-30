from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import yaml
import json
import os
import logging
from termcolor import colored
from datasets.polyb import PolybDatset
from lib.solvers import anderson, broyden

from core.seg_criterion import PolypCriterion
from utils.utils import FullModel, increment_path, intersect_dicts

import numpy as np
from .segformer import *
# from segformer import *
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import torch.autograd as autograd
from lib.optimizations import VariationalHidDropout2d
from lib.jacobian import jac_loss_estimate, power_method
from lib.layer_utils import flat, restore_shape

BN_MOMENTUM = 0.1
BLOCK_GN_AFFINE = True  # Don't change the value here. The value is controlled by the yaml files.
FUSE_GN_AFFINE = True  # Don't change the value here. The value is controlled by the yaml files.
POST_GN_AFFINE = True  # Don't change the value here. The value is controlled by the yaml files.
DEQ_EXPAND = 5  # Don't change the value here. The value is controlled by the yaml files.
NUM_GROUPS = 4  # Don't change the value here. The value is controlled by the yaml files.
logger = logging.getLogger(__name__)


class MDEQ_SegformerModule(nn.Module):
    def __init__(self, img_size=256, version='mit_b0'):
        """
        An MDEQ layer (note that MDEQ only has one layer).
        """
        super(MDEQ_SegformerModule, self).__init__()
        self.img_size = img_size
        m = importlib.import_module('models.segformer')
        self.in_chans = 64
        if version == 'mit_b0':
            self.in_chans = 32
        self.segformer = getattr(m, version)(img_size=self.img_size, in_chans = self.in_chans)
        embed_dims = self.segformer.embed_dims
        self.stage0 = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=embed_dims[0], kernel_size=1, stride=1, padding=0, bias= True),
            nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[0], kernel_size=1, stride=1, padding=0, bias= True)
        ])
    def forward(self, z, x, *args):  # z is initialized, x is image
        """
        The two steps of a multiscale DEQ module (see paper): a per-resolution residual block and
        a parallel multiscale fusion step.
        """
        x = self.stage0[0](x)
        z = self.stage0[1](z)
        inputs = z + x
        return self.segformer(inputs)


class MDEQ_SegformerNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ model with the given hyperparameters

        Args:
            cfg ([config]): The configuration file (parsed from yaml) specifying the model settings
        """
        super(MDEQ_SegformerNet, self).__init__()

        self.parse_cfg(cfg)
        # init_chansize = self.init_chansize

        # PART II: MDEQ's f_\theta layer

        self.fullstage = self._make_stage(self.img_size, self.version)
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        self.num_channels = self.fullstage.segformer.embed_dims
        # print(self.num_channels)
        self.iodrop = VariationalHidDropout2d(0.0)
        self.hook = None

    def parse_cfg(self, cfg):
        """
        Parse a configuration file
        """
        global DEQ_EXPAND, NUM_GROUPS, BLOCK_GN_AFFINE, FUSE_GN_AFFINE, POST_GN_AFFINE
        self.num_branches = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_BRANCHES']
        # self.num_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_CHANNELS']
        # self.init_chansize = self.num_channels[0]
        self.img_size = cfg['MODEL']['IMG_SIZE']
        self.version = cfg['MODEL']['SEGFORMER_VERSION']
        self.num_layers = cfg['MODEL']['NUM_LAYERS']
        self.dropout = cfg['MODEL']['DROPOUT']
        self.wnorm = cfg['MODEL']['WNORM']
        self.num_classes = cfg['MODEL']['NUM_CLASSES']
        self.downsample_times = cfg['MODEL']['DOWNSAMPLE_TIMES']
        self.fullstage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']
        self.pretrain_steps = cfg['TRAIN']['PRETRAIN_STEPS']
        self.pretrain_path = cfg['MODEL']['PRETRAINED']
        # DEQ related
        self.f_solver = eval(cfg['DEQ']['F_SOLVER'])
        self.b_solver = eval(cfg['DEQ']['B_SOLVER'])
        if self.b_solver is None:
            self.b_solver = self.f_solver
        self.f_thres = cfg['DEQ']['F_THRES']
        self.b_thres = cfg['DEQ']['B_THRES']
        self.stop_mode = cfg['DEQ']['STOP_MODE']

        # Update global variables
        DEQ_EXPAND = cfg['MODEL']['EXPANSION_FACTOR']
        NUM_GROUPS = cfg['MODEL']['NUM_GROUPS']
        BLOCK_GN_AFFINE = cfg['MODEL']['BLOCK_GN_AFFINE']
        FUSE_GN_AFFINE = cfg['MODEL']['FUSE_GN_AFFINE']
        POST_GN_AFFINE = cfg['MODEL']['POST_GN_AFFINE']

    def _make_stage(self, img_size, version):
        """
        Build an MDEQ block with the given hyperparameters
        """
        return MDEQ_SegformerModule(img_size, version)

    def _forward(self, x, train_step=-1, compute_jac_loss=True, spectral_radius_mode=False, writer=None, logger=None,
                 **kwargs):
        """
        The core MDEQ module. In the starting phase, we can (optionally) enter a shallow stacked f_\theta training mode
        to warm up the weights (specified by the self.pretrain_steps; see below)
        """
        # x is image
        num_branches = self.num_branches
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        bsz, _, H, W = x.shape
        # Inject only to the highest resolution...

        z_0 = torch.zeros((bsz, self.num_channels[0], H, W)).to(x)  # initialize z
        size = [self.num_channels[0], H, W]
        z1 = flat(z_0)
        func = lambda z: flat(self.fullstage(restore_shape(z, size), x))
        # For variational dropout mask resetting and weight normalization re-computations

        jac_loss = torch.tensor(0.0).to(x)
        sradius = torch.zeros(bsz, 1).to(x)
        deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)

        # Multiscale Deep Equilibrium!
        if not deq_mode:
            for layer_ind in range(self.num_layers):
                z1 = func(z1)
            new_z1 = z1

            if self.training:
                if compute_jac_loss:
                    z2 = z1.clone().detach().requires_grad_()
                    new_z2 = func(z2)
                    jac_loss = jac_loss_estimate(new_z2, z2)
        else:
            with torch.no_grad():
                if self.training:
                    result = self.f_solver(func, z1, threshold=f_thres, stop_mode=self.stop_mode, name="forward",
                                           writer=writer, logger=logger)
                else:
                    result = self.f_solver(func, z1, threshold=f_thres, stop_mode=self.stop_mode, name='forward',
                                           writer=None)
                z1 = result['result']
            new_z1 = z1

            if (not self.training) and spectral_radius_mode:
                with torch.enable_grad():
                    new_z1 = func(z1.requires_grad_())
                _, sradius = power_method(new_z1, z1, n_iters=150)

            if self.training:
                new_z1 = func(z1.requires_grad_())
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1, z1)

                def backward_hook(grad):
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()
                    result = self.b_solver(lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad,
                                           torch.zeros_like(grad),
                                           threshold=b_thres, stop_mode=self.stop_mode, name="backward", writer=writer,
                                           logger=logger)
                    return result['result']

                self.hook = new_z1.register_hook(backward_hook)

        y_list = restore_shape(new_z1, size)
        y_list = self.iodrop(y_list)
        return y_list, jac_loss.view(1, -1), sradius.view(-1, 1)

    def forward(self, x, train_step=-1, **kwargs):
        raise NotImplemented  # To be inherited & implemented by MDEQClsNet and MDEQSegNet (see mdeq.py)


class MDEQSegNet(MDEQ_SegformerNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Segmentation model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQSegNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)

        # Last layer
        last_inp_channels = self.num_channels[0]
        self.last_layer = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, cfg.DATASET.NUM_CLASSES, kernel_size=1, stride=1, padding=0))

    def segment(self, y):
        """
        Given outputs at multiple resolutions, segment the feature map by predicting the class of each pixel
        """
        # Segmentation Head
        y = self.last_layer(y)
        # print(y.shape)
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
    return model
# if __name__ == "__main__":
#     class obj:
#         def __init__(self, dict1):
#             self.__dict__.update(dict1)
#
#         def __getitem__(self, item):
#             return self.__dict__[item]
#
#     config = yaml.load(open(r"C:\Users\chau\pythonProject2\experiments\segformer.yaml", "r"), Loader=yaml.FullLoader)
#     config = json.loads(json.dumps(config), object_hook=obj)
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     model = get_seg_net(config)
#     criterion = PolypCriterion(0.4, 0.6, smooth=1)
#     model = FullModel(model, criterion)
#     trainset = PolybDatset(
#         root_path=r"C:\Users\chau\new_Project2\TrainDataset",
#         img_subpath="image",
#         label_subpath="mask",
#         img_size=config.TRAIN.IMAGE_SIZE,
#         cache_train=False,
#         use_aug=True,
#         use_cutmix=0.5
#     )
#     img, label = trainset[0]
#     img = img.unsqueeze(0)
#     label = label.unsqueeze(0)
#     losses, jac_loss, preds, _ = model(
#         img,
#         label,
#         train_step=1,
#         compute_jac_loss=False,
#         f_thres=1,
#         b_thres=1,
#         writer=None,
#         logger=None
#     )
#     print(preds.shape)
#
