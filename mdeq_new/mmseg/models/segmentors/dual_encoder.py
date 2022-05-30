import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

from .hardnet_68 import hardnet

from .resnet import BasicBlock as ResBlock
from . import GSConv as gsc

import numpy as np
import cv2

from .lib.conv_layer import Conv, BNPReLU
from .lib.axial_atten import AA_kernel
from .lib.context_module import CFPModule

@SEGMENTORS.register_module()
class DualEncoderDecoder(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DualEncoderDecoder, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.hardnet = hardnet(arch=68)

    def forward(self, x):
        hardnetout = self.hardnet(x)
        segout = self.backbone(x)
        inputs = [hardnetout, segout]
        
        z = self.decode_head.forward(inputs)
        z = resize(
            input=z,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
            
        return z
        
@SEGMENTORS.register_module()
class DualEncoderDecoder_ver2(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DualEncoderDecoder_ver2, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.hardnet = hardnet(arch=68)

    def forward(self, x):
        hardnetout = self.hardnet(x)
        segout = self.backbone(x)
        inputs = [hardnetout, segout]
        
        z = self.decode_head.forward(inputs)
            
        return z
        
class EncoderDecoder_ver2(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder_ver2, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.hardnet = hardnet(arch=68)

    def forward(self, x):
        segout = self.backbone(x)
        
        z = self.decode_head.forward(segout)
            
        return z
        
        
@SEGMENTORS.register_module()
class DualSegUPer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DualSegUPer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.hardnet = hardnet(arch=68)

    def forward(self, x):
        segout = self.backbone(x)
        c1 = segout[0] 
        c2 = segout[1] 
        c3 = segout[2] 
        c4 = segout[3] 
        
        hardnetout = self.hardnet(x)
        x1 = hardnetout[0] 
        x2 = hardnetout[1] 
        x3 = hardnetout[2] 
        x4 = hardnetout[3]

        c1 = torch.cat((c1, x1), 1)
        c2 = torch.cat((c2, x2), 1)
        c3 = torch.cat((c3, x3), 1)
        c4 = torch.cat((c4, x4), 1)

        z = self.decode_head.forward([c1, c2, c3, c4])
            
        return z

@SEGMENTORS.register_module()
class EdgeSegUPer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EdgeSegUPer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(320, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 64, kernel_size=1)
        self.res1 = ResBlock(64, 64)
        self.d1 = nn.Conv2d(64, 32, kernel_size=1)
        self.res2 = ResBlock(32, 32)
        self.d2 = nn.Conv2d(32, 16, kernel_size=1)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        num_filters = 32
        self.expand = nn.Sequential(nn.Conv2d(1, num_filters, kernel_size=1),
                                    nn.BatchNorm2d(num_filters),
                                    nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()
        
        self.dec0 = Conv2dReLU(
            2*num_filters, num_filters,
            kernel_size=3,
            padding=1,
            use_batchnorm=True)
        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        x_size = x.size()
        
        segout = self.backbone(x)
        x1 = segout[0] 
        x2 = segout[1] 
        x3 = segout[2] 
        x4 = segout[3]

        z = self.decode_head.forward([x1, x2, x3, x4])
        z = F.interpolate(z, scale_factor=4, mode='bilinear')
        
        ss = F.interpolate(self.d0(x1), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(x2), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss, g1 = self.gate1(ss, c3)
        ss = self.res2(ss)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(x3), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss, g2 = self.gate2(ss, c4)
        ss = self.res3(ss)
        ss = self.d3(ss)
        c5 = F.interpolate(self.c5(x4), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss, g3 = self.gate3(ss, c5)
        ss = self.fuse(ss)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)
        
        ### Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()
        ### End Canny Edge
        
        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        edge = self.expand(acts)
        
        z = torch.cat([z, edge], dim=1)
        z = self.dec0(z)
        z = self.final(z)
        
        return z, edge_out

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=False)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
        
        
@SEGMENTORS.register_module()
class ShapeSegUPer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(ShapeSegUPer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.decode_head2 = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        self.decode_head2.init_weights()
        
        self.sigmoid = nn.Sigmoid()
        self.merge1 = Conv2dReLU(
            2, 1,
            kernel_size=3,
            padding=1,
            use_batchnorm=True)
            
        self.merge2 = Conv2dReLU(
            1, 1,
            kernel_size=3,
            padding=1,
            use_batchnorm=True)    
        

    def forward(self, x):
        segout = self.backbone(x)
        c1 = segout[0] 
        c2 = segout[1] 
        c3 = segout[2] 
        c4 = segout[3] 

        map = self.decode_head.forward([c1, c2, c3, c4])
        shape = self.decode_head2.forward([c1, c2, c3, c4])
        # shape = self.sigmoid(shape)
            
        z = torch.cat((map, shape), 1)
        z = self.merge1(z)
        z = self.merge2(z)
        
        return z, map, shape
        
@SEGMENTORS.register_module()
class CaraSegUPer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CaraSegUPer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.CFP_1 = CFPModule(128, d = 8)
        self.CFP_2 = CFPModule(320, d = 8)
        self.CFP_3 = CFPModule(512, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(128,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(320,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(512,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(128,128)
        self.aa_kernel_2 = AA_kernel(320,320)
        self.aa_kernel_3 = AA_kernel(512,512)
        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  64x88x88
        x2 = segout[1]  # 128x44x44
        x3 = segout[2]  # 320x22x22
        x4 = segout[3]  # 512x11x11

        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4) 
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(x3) 
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(aa_atten_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2) 
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1

@SEGMENTORS.register_module()
class SegCaraSegUPer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SegCaraSegUPer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.CFP_1 = CFPModule(128, d = 8)
        self.CFP_2 = CFPModule(320, d = 8)
        self.CFP_3 = CFPModule(512, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(128,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(320,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(512,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(128,128)
        self.aa_kernel_2 = AA_kernel(320,320)
        self.aa_kernel_3 = AA_kernel(512,512)
        
        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(320, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)
        
        self.d0 = nn.Conv2d(64, 64, kernel_size=1)
        self.res1 = ResBlock(64, 64)
        self.d1 = nn.Conv2d(64, 32, kernel_size=1)
        self.res2 = ResBlock(32, 32)
        self.d2 = nn.Conv2d(32, 16, kernel_size=1)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        num_filters = 32
        self.expand = nn.Sequential(nn.Conv2d(1, 32, kernel_size=1),
                                    nn.BatchNorm2d(num_filters),
                                    nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()
        
        self.dec0 = Conv2dReLU(
            33, 32,
            kernel_size=3,
            padding=1,
            use_batchnorm=True)
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  64x88x88
        x2 = segout[1]  # 128x44x44
        x3 = segout[2]  # 320x22x22
        x4 = segout[3]  # 512x11x11
        size = x.size()
        ss = F.interpolate(self.d0(x1), size[2:], mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(x2), size[2:], mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss, g1 = self.gate1(ss, c3)
        ss = self.res2(ss)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(x3), size[2:], mode='bilinear', align_corners=True)
        ss, g2 = self.gate2(ss, c4)
        ss = self.res3(ss)
        ss = self.d3(ss)
        c5 = F.interpolate(self.c5(x4), size[2:], mode='bilinear', align_corners=True)
        ss, g3 = self.gate3(ss, c5)
        ss = self.fuse(ss)
        ss = F.interpolate(ss, size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)
        
        ### Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((size[0], 1, size[2], size[3]))
        for i in range(size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()
        ### End Canny Edge
        
        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        edge = self.expand(acts)
        
        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4) 
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(x3) 
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(aa_atten_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2) 
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear')

        lateral_map_5 = torch.cat([lateral_map_5, edge], dim=1)
        lateral_map_5 = self.dec0(lateral_map_5)
        lateral_map_5 = self.final(lateral_map_5)
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1, edge_out
        
@SEGMENTORS.register_module()
class CaraSegUPer_woCFP(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CaraSegUPer_woCFP, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()

        self.ra1_conv1 = Conv(128,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(320,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(512,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(128,128)
        self.aa_kernel_2 = AA_kernel(320,320)
        self.aa_kernel_3 = AA_kernel(512,512)
        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  64x88x88
        x2 = segout[1]  # 128x44x44
        x3 = segout[2]  # 320x22x22
        x4 = segout[3]  # 512x11x11

        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = x4 
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = x3
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(aa_atten_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = x2
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1

@SEGMENTORS.register_module()
class CaraSegUPer_woAA(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CaraSegUPer_woAA, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.CFP_1 = CFPModule(128, d = 8)
        self.CFP_2 = CFPModule(320, d = 8)
        self.CFP_3 = CFPModule(512, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(128,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(320,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(512,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  64x88x88
        x2 = segout[1]  # 128x44x44
        x3 = segout[2]  # 320x22x22
        x4 = segout[3]  # 512x11x11

        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4) 
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(cfp_out_1)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(x3) 
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(cfp_out_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2) 
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(cfp_out_3)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1
        
        
@SEGMENTORS.register_module()
class CaraSegUPer_woAA_ver2(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CaraSegUPer_woAA_ver2, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.CFP_1 = CFPModule(128, d = 8)
        self.CFP_2 = CFPModule(320, d = 8)
        self.CFP_3 = CFPModule(512, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(128,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(320,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(512,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  64x88x88
        x2 = segout[1]  # 128x44x44
        x3 = segout[2]  # 320x22x22
        x4 = segout[3]  # 512x11x11

        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4) 
        cfp_out_1 += x4
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(cfp_out_1)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(x3)
        cfp_out_2 += x3
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(cfp_out_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2)
        cfp_out_3 += x2
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(cfp_out_3)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1
        
@SEGMENTORS.register_module()
class CaraSegUPer_ver2(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CaraSegUPer_ver2, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.CFP_1 = CFPModule(128, d = 8)
        self.CFP_2 = CFPModule(320, d = 8)
        self.CFP_3 = CFPModule(512, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(128,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(320,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(512,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(128,128)
        self.aa_kernel_2 = AA_kernel(320,320)
        self.aa_kernel_3 = AA_kernel(512,512)
        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  64x88x88
        x2 = segout[1]  # 128x44x44
        x3 = segout[2]  # 320x22x22
        x4 = segout[3]  # 512x11x11

        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4)
        # cfp_out_1 += x4
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3 += cfp_out_1
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(x3)
        # cfp_out_2 += x3
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2 += cfp_out_2
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(aa_atten_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2)
        # cfp_out_3 += x2
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1 += cfp_out_3
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1
      

@SEGMENTORS.register_module()
class NeoFormer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(NeoFormer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.CFP_1 = CFPModule(128, d = 8)
        self.CFP_2 = CFPModule(320, d = 8)
        self.CFP_3 = CFPModule(512, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(128,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(320,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(512,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(128,128)
        self.aa_kernel_2 = AA_kernel(320,320)
        self.aa_kernel_3 = AA_kernel(512,512)
        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  64x88x88
        x2 = segout[1]  # 128x44x44
        x3 = segout[2]  # 320x22x22
        x4 = segout[3]  # 512x11x11

        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        out = decoder_1[:, 1, :, :]
        out = torch.unsqueeze(out, 1)

        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(out, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4)
        # cfp_out_1 += x4
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3 += cfp_out_1
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        out2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        lateral_map_2 = lateral_map_1
        lateral_map_2[:, 1, :, :] = out2.squeeze()
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(x3)
        # cfp_out_2 += x3
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2 += cfp_out_2
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(aa_atten_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        out3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')      
        lateral_map_3 = lateral_map_2
        lateral_map_3[:, 1, :, :] = out3.squeeze()
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2)
        # cfp_out_3 += x2
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1 += cfp_out_3
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        out5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        lateral_map_5 = lateral_map_3
        lateral_map_5[:, 1, :, :] = out5.squeeze()
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1