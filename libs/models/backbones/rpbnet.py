from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging
from os.path import join
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
# from torchsummary import summary

class conv3x3(nn.Module):

    def __init__(self, Cin, Cout, stride=1):

        super(conv3x3, self).__init__()

        self.conv = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=3, stride=stride, padding=1, dilation=1, groups=1, bias=True, padding_mode='reflect')
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x):

        out = self.conv(x)

        return out
    
class conv1x1(nn.Module):

    def __init__(self, Cin, Cout):

        super(conv1x1, self).__init__()

        self.conv = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x):

        out = self.conv(x)

        return out


class conv3x3_act(nn.Module):
    
    def __init__(self, Cin, Cout, stride=1):

        super(conv3x3_act, self).__init__()

        self.LRelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv = conv3x3(Cin=Cin, Cout=Cout, stride=stride)
        

    def forward(self, x):

        out = self.conv(x)
        out = self.LRelu(out)

        return out
    

class BasicBlock(nn.Module): #bn사용X,

    def __init__(self, Cin, Cout, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):

        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if groups != 1 or base_width != 64: #WideResNet, ResNeXt 구현시 사용
            raise ValueError('BasicBlock only supports groups=1 and base width = 64')
        
        self.LRelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv1 = conv3x3_act(Cin=Cin, Cout=Cout, stride=1)
        self.conv2 = conv3x3(Cin=Cin, Cout=Cout, stride=1)
        self.downsample = downsample
        self.stride=stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.LRelu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.LRelu(out)

        return out


class ConvBlock(nn.Module):

    def __init__(self, Cin, Cout):

        super(ConvBlock, self).__init__()

        self.LRelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.cb_conv1 = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='reflect')
        nn.init.orthogonal_(self.cb_conv1.weight)
        self.cb_conv2 = nn.Conv2d(in_channels=Cout, out_channels=Cout, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='reflect')
        nn.init.orthogonal_(self.cb_conv2.weight)
        self.cb_conv3 = nn.Conv2d(in_channels=Cout, out_channels=Cout, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='reflect')
        nn.init.orthogonal_(self.cb_conv3.weight)

    def forward(self, x):
        
        out = self.cb_conv1(x)
        out = self.LRelu(out)
        out = self.cb_conv2(out)
        out = self.LRelu(out)
        out = self.cb_conv3(out)
        out = self.LRelu(out)

        return out
    

class RPB(nn.Module):

    def __init__(self, Cin, Cout):

        super(RPB, self).__init__()

        self.LRelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.r_conv1 = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='reflect')
        nn.init.orthogonal_(self.r_conv1.weight)
        self.r_deconv = nn.ConvTranspose2d(in_channels=Cout, out_channels=Cin, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros') #[B, 64, 80, 200]
        nn.init.orthogonal_(self.r_deconv.weight)
        self.r_conv2 = nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='reflect')
        nn.init.orthogonal_(self.r_conv2.weight)

    def forward(self, x):

        out = self.r_conv1(x)
        out = self.LRelu(out)
        out = self.r_deconv(out)
        out = self.LRelu(out)
        out = self.r_conv2(out)
        out = self.LRelu(out)

        return out
    


class RPB1(nn.Module):

    def __init__(self, Cin, Cout):

        super(RPB1, self).__init__()

        # channels=[64, 128, 256, 512]

        self.rpb = RPB(Cin=Cin, Cout=Cout)
        self.conv1 = conv3x3_act(Cin=Cout, Cout=Cout)
        # self.basicblock = BasicBlock(Cin=Cout, Cout=Cout)

    def forward(self, x):

        out = self.rpb(x)
        # out = self.basicblock(out)
        out = self.conv1(out)

        return out
    

class RPB2(nn.Module):

    def __init__(self, Cin, Cout):

        super(RPB2, self).__init__()

        self.rpb = RPB(Cin=Cin, Cout=Cout)
        self.conv1 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv2 = conv3x3_act(Cin=Cout, Cout=Cout)
        # self.basicblock1 = BasicBlock(Cin=Cout, Cout=Cout)
        # self.basicblock2 = BasicBlock(Cin=Cout, Cout=Cout)

    def forward(self, x):

        out = self.rpb(x)
        out = self.conv1(out)
        out = self.conv2(out)
        # out = self.basicblock1(out)
        # out = self.basicblock2(out)

        return out
    

class RPB3(nn.Module):

    def __init__(self, Cin, Cout):

        super(RPB3, self).__init__()

        self.rpb = RPB(Cin=Cin, Cout=Cout)
        self.conv1 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv2 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv3 = conv3x3_act(Cin=Cout, Cout=Cout)
        # self.basicblock1 = BasicBlock(Cin=Cout, Cout=Cout)
        # self.basicblock2 = BasicBlock(Cin=Cout, Cout=Cout)
        # self.basicblock3 = BasicBlock(Cin=Cout, Cout=Cout)

    def forward(self, x):

        out = self.rpb(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        # out = self.basicblock1(out)
        # out = self.basicblock2(out)
        # out = self.basicblock3(out)

        return out
    
class RPB4(nn.Module):

    def __init__(self, Cin, Cout):

        super(RPB4, self).__init__()

        self.rpb = RPB(Cin=Cin, Cout=Cout)
        self.conv1 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv2 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv3 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv4 = conv3x3_act(Cin=Cout, Cout=Cout)
        # self.basicblock1 = BasicBlock(Cin=Cout, Cout=Cout)
        # self.basicblock2 = BasicBlock(Cin=Cout, Cout=Cout)
        # self.basicblock3 = BasicBlock(Cin=Cout, Cout=Cout)

    def forward(self, x):

        out = self.rpb(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        # out = self.basicblock1(out)
        # out = self.basicblock2(out)
        # out = self.basicblock3(out)

        return out
    
class RPB5(nn.Module):

    def __init__(self, Cin, Cout):

        super(RPB5, self).__init__()

        self.rpb = RPB(Cin=Cin, Cout=Cout)
        self.conv1 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv2 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv3 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv4 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv5 = conv3x3_act(Cin=Cout, Cout=Cout)

    def forward(self, x):

        out = self.rpb(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out
    
class RPB6(nn.Module):

    def __init__(self, Cin, Cout):

        super(RPB6, self).__init__()

        self.rpb = RPB(Cin=Cin, Cout=Cout)
        self.conv1 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv2 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv3 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv4 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv5 = conv3x3_act(Cin=Cout, Cout=Cout)
        self.conv6 = conv3x3_act(Cin=Cout, Cout=Cout)

    def forward(self, x):

        out = self.rpb(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        return out
    
    
class Input_Processing(nn.Module):

    def __init__(self):

        super(Input_Processing, self).__init__()
        
        self.in_conv1 = conv1x1(Cin=48, Cout=64)


    def forward(self, x):

        depth_to_space = F.pixel_unshuffle(x, downscale_factor=4) #[B, 48, 80, 200]
        out = self.in_conv1(depth_to_space) #[B, 64, 80, 200]

        return out

    
class Trunk(nn.Module): 

    def __init__(self):

        super(Trunk, self).__init__()
        
        self.rpb1 = RPB6(Cin=64, Cout=128)
        self.rpb2 = RPB4(Cin=128, Cout=256)
        self.rpb3 = RPB2(Cin=256, Cout=512)

        
    def forward(self, x):

        fpn1 = self.rpb1(x)
        fpn2 = self.rpb2(fpn1)
        fpn3 = self.rpb3(fpn2)

        return fpn1, fpn2, fpn3
    
class Head1(nn.Module):

    def __init__(self, Cin, Cmid, Cout):

        super(Head1, self).__init__()

        self.head1 = conv3x3_act(Cin=Cin, Cout=Cmid)
        self.head2 = conv3x3_act(Cin=Cmid, Cout=Cmid)
        self.head3 = conv3x3_act(Cin=Cmid, Cout=Cmid)
        # self.head4 = conv3x3_act(Cin=Cin, Cout=Cin)
        # self.head5 = conv3x3_act(Cin=Cin, Cout=Cin)
        # self.head6 = conv3x3_act(Cin=Cin, Cout=Cin)
        # self.head7 = conv3x3_act(Cin=Cin, Cout=Cin)
        self.head = conv1x1(Cin=Cmid, Cout=Cout)
        
    def forward(self, x):

        out = self.head1(x)
        out = self.head2(out)
        out = self.head3(out)
        # out = self.head4(out)
        # out = self.head5(out)
        # out = self.head6(out)
        # out = self.head7(out)
        out = self.head(out)

        return out
    
class Head2(nn.Module):

    def __init__(self, Cin, Cmid, Cout):

        super(Head2, self).__init__()

        self.head1 = conv3x3_act(Cin=Cin, Cout=Cmid)
        self.head2 = conv3x3_act(Cin=Cmid, Cout=Cmid)
        # self.head3 = conv3x3_act(Cin=Cin, Cout=Cin)
        # self.head4 = conv3x3_act(Cin=Cin, Cout=Cin)
        self.head = conv1x1(Cin=Cmid, Cout=Cout)
        
    def forward(self, x):

        out = self.head1(x)
        out = self.head2(out)
        # out = self.head3(out)
        # out = self.head4(out)
        out = self.head(out)

        return out
    
class Head3(nn.Module):

    def __init__(self, Cin, Cmid, Cout):

        super(Head3, self).__init__()

        self.head1 = conv3x3_act(Cin=Cin, Cout=Cmid)
        self.head = conv1x1(Cin=Cmid, Cout=Cout)
        
    def forward(self, x):

        out = self.head1(x)
        out = self.head(out)

        return out
        
class Head(nn.Module): 
        
        def __init__(self):
            super(Head, self).__init__()

            self.head1 = Head1(Cin=128, Cmid=128, Cout=64)
            self.head2 = Head2(Cin=256, Cmid=128, Cout=64)
            self.head3 = Head3(Cin=512, Cmid=128, Cout=64)


        def forward(self, fpn1, fpn2, fpn3):
            
            fpn1 = self.head1(fpn1)
            fpn2 = self.head2(fpn2)
            fpn3 = self.head3(fpn3)

            return fpn1, fpn2, fpn3 #([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25])


@BACKBONES.register_module
class RPBNet(nn.Module):

    def __init__(self):
        
        super(RPBNet, self).__init__()
        self.input_processing = Input_Processing()
        self.trunk = Trunk()
        self.head = Head()

        
    def forward(self, x):
        
        input_processing = self.input_processing(x)
        rpb1, rpn2, rpn3 = self.trunk(input_processing)
        fpn1, fpn2, fpn3 = self.head(rpb1, rpn2, rpn3)

        return fpn1, fpn2, fpn3
