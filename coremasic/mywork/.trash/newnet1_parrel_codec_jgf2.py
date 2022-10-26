#jg:joint_gmm #codec:熵编码编解码分离工程实现
#gf:gmm_full:pixel wise-gmm -- jgf2 串行

import argparse
import math
import random
import shutil
import os
import sys
import numpy as np
import math
import torch.nn.functional as F
import time
from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq
#在这里改GMM计算方式
from compressai.entropy_models import (EntropyBottleneck, GaussianMixtureConditional,GaussianMixtureConditional_gf, GaussianConditional)
from compressai.layers import GDN, MaskedConv2d
from compressai.ans import BufferedRansEncoder, RansDecoder  # pylint: disable=E0611,E0401
from compressai.models.utils import update_registered_buffers, conv, deconv


import torch
import torch.optim as optim
import torch.nn as nn
import kornia
from torch.utils.data import DataLoader

from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
from PIL import Image

from compressai.layers import *

class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """
    def __init__(self, entropy_bottleneck_channels, init_weights=True):
        super().__init__()
        self.entropy_bottleneck1 = EntropyBottleneck(
            entropy_bottleneck_channels)

        self.entropy_bottleneck2 = EntropyBottleneck(
            entropy_bottleneck_channels)


        if init_weights:
            self._initialize_weights()

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(m.loss() for m in self.modules()
                       if isinstance(m, EntropyBottleneck))
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args):
        raise NotImplementedError()

    def parameters(self):
        """Returns an iterator over the model parameters."""
        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def aux_parameters(self):
        """
        Returns an iterator over the entropy bottleneck(s) parameters for
        the auxiliary loss.
        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            m.update(force=force)

######################################

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp_loss'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'].values())
        out['mse_loss'] = self.mse(output['x_hat'], target) #end to end
        out['loss'] = self.lmbda * 255**2 * out['mse_loss'] + out['bpp_loss']

        return out


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Enhancement_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.RB1 = ResidualBlock(32, 32)
        self.RB2 = ResidualBlock(32, 32)
        self.RB3 = ResidualBlock(32, 32)
    def forward(self, x):
        identity = x

        out = self.RB1(x)
        out = self.RB2(out)
        out = self.RB3(out)

        out = out + identity
        return out

class Enhancement(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv3x3(6, 32)

        self.EB1 = Enhancement_Block()
        self.EB2 = Enhancement_Block()
        self.EB3 = Enhancement_Block()

        self.conv2 = conv3x3(32, 3)

    def forward(self, x ,x_another_warp):
        identity = x

        out = torch.cat((x,x_another_warp),dim=-3)

        out = self.conv1(out)
        out = self.EB1(out)
        out = self.EB2(out)
        out = self.EB3(out)
        out = self.conv2(out)

        out = out + identity
        return out


###
#Entropy：
#生成z1,z2

class encode_hyper(nn.Module):
    def __init__(self,N,M):
        super().__init__()
        self.encode_hyper = nn.Sequential(
            #先abs 再输入进来
            conv(in_channels=M, out_channels=N, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),

            conv(in_channels=N, out_channels=N, kernel_size=5), #stride = 2
            nn.ReLU(inplace=True),

            conv(in_channels=N, out_channels=N, kernel_size=5), #stride = 2
            #出去后接bottleneck
        )
    def forward(self,y):
        self.y_abs = torch.abs(y)
        self.z = self.encode_hyper(self.y_abs)
        return self.z


#空间域池化 兼容不同patch_size
class spatial_pool2d(nn.Module):
    def __init__(self):
        super(spatial_pool2d, self).__init__()
    def forward(self,X):
        # 空间池化
        Y = torch.zeros([X.size()[0], X.size()[1], 1, 1])
        #cuda or cpu!!
        if X.device.type=='cuda':
            Y = Y.to(X.device.type+":"+str(X.device.index))
        for b in range(Y.size()[0]):
            for c in range(Y.size()[1]):
                Y[b, c, 0, 0] = X[b:(b + 1), c:(c + 1), :, :].max()
        return Y

# ##z1 for y1
# class gmm_hyper_y1(nn.Module):
#     def __init__(self,N,M,K): #K表示GMM对应正态分布个数
#         super().__init__()
#         self.N = N
#         self.M = M
#         self.K = K
#
#         # 每个都是上采样4倍才行
#         self.gmm_sigma = nn.Sequential(
#             deconv(in_channels=N,out_channels=N,kernel_size=5), #stride=2,padding=kernel_size//2,output_padding=stride-1
#             nn.ReLU(inplace=True),
#
#             deconv(in_channels=N, out_channels=N, kernel_size=5),# stride=2,padding=kernel_size//2,output_padding=stride-1
#             nn.ReLU(inplace=True),
#
#             conv(in_channels=N,out_channels=(M*K),kernel_size=5,stride=1), #padding=kernel_size//2
#             nn.ReLU(inplace=True),
#         )
#
#         self.gmm_means = nn.Sequential(
#             deconv(in_channels=N, out_channels=N, kernel_size=5),
#             # stride=2,padding=kernel_size//2,output_padding=stride-1
#             nn.LeakyReLU(inplace=True),
#
#             deconv(in_channels=N, out_channels=N, kernel_size=5),
#             # stride=2,padding=kernel_size//2,output_padding=stride-1
#             nn.LeakyReLU(inplace=True),
#
#             conv(in_channels=N, out_channels=(M * K), kernel_size=5, stride=1),  # padding=kernel_size//2
#         )
#
#
#         self.gmm_weights = nn.Sequential(
#             deconv(in_channels=N, out_channels=N, kernel_size=5),
#             # stride=2,padding=kernel_size//2,output_padding=stride-1
#             nn.LeakyReLU(inplace=True),
#
#             deconv(in_channels=N, out_channels=(M*K), kernel_size=5),
#             # stride=2,padding=kernel_size//2,output_padding=stride-1
#             # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
#             spatial_pool2d(),
#             nn.LeakyReLU(inplace=True),
#
#             conv(in_channels=(M*K), out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
#             #出去后要接一个softmax层，表示概率！！
#         )
#
#     def forward(self,z1):
#         self.sigma = self.gmm_sigma(z1)
#         self.means = self.gmm_means(z1)
#         #weights要加softmax！！  nn.functional.softmax
#         #softmax 加的有问题，需要按照
#         # self.weights = nn.functional.softmax(self.gmm_weights(z1),dim = -3)
#         #修正
#         temp = torch.reshape(self.gmm_weights(z1),(-1,self.K,self.M,1,1)) #方便后续reshape合并时同一个M的数据相邻成组
#         temp = nn.functional.softmax(temp,dim=-4)  #每个Mixture进行一次归一
#         self.weights = torch.reshape(temp,(-1,self.M*self.K,1,1))
#         ###
#         return self.sigma,self.means,self.weights
#
# ##z2+y1 for y2
# class gmm_hyper_y2(nn.Module):
#     def __init__(self,N,M,K): #K表示GMM对应正态分布个数
#         super().__init__()
#         self.N = N
#         self.M = M
#         self.K = K
#
#         self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=4) #固定为4倍 因为z和y分辨率本身就差4倍
#
#         # 输入与y1分辨率相同，但通道数是2倍
#         self.gmm_sigma = nn.Sequential(
#             conv(in_channels=(N+M),out_channels=N,kernel_size=5,stride=1),
#             nn.ReLU(inplace=True),
#
#             conv(in_channels=N,out_channels=N,kernel_size=5,stride=1),
#             nn.ReLU(inplace=True),
#
#             conv(in_channels=N,out_channels=(M*K),kernel_size=5,stride=1), #padding=kernel_size//2
#             nn.ReLU(inplace=True),
#         )
#
#         self.gmm_means = nn.Sequential(
#             conv(in_channels=(N+M),out_channels=N,kernel_size=5,stride=1),
#             nn.LeakyReLU(inplace=True),
#
#             conv(in_channels=N,out_channels=N,kernel_size=5,stride=1),
#             # stride=2,padding=kernel_size//2,output_padding=stride-1
#             nn.LeakyReLU(inplace=True),
#
#             conv(in_channels=N, out_channels=(M * K), kernel_size=5, stride=1),  # padding=kernel_size//2
#         )
#
#
#         self.gmm_weights = nn.Sequential(
#             conv(in_channels=(N+M),out_channels=N,kernel_size=5,stride=1),
#             nn.LeakyReLU(inplace=True),
#
#             conv(in_channels=N,out_channels= (M*K),kernel_size=5,stride=1),
#             # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
#             spatial_pool2d(),
#             nn.LeakyReLU(inplace=True),
#
#             conv(in_channels=(M*K), out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
#             #出去后要接一个softmax层，表示概率！！
#         )
#
#     def forward(self,z2,y1):
#         self.up_z2 = self.upsample_layer(z2)
#         self.cat_in = torch.cat((self.up_z2,y1),dim=-3)
#
#         self.sigma = self.gmm_sigma(self.cat_in)
#         self.means = self.gmm_means(self.cat_in)
#         #softmax!!
#         # self.weights = nn.functional.softmax(self.gmm_weights(self.cat_in),dim=-3)
#         # 修正
#         temp = torch.reshape(self.gmm_weights(self.cat_in), (-1, self.K, self.M,  1, 1))  # 方便后续reshape合并时同一个M的数据相邻成组
#         temp = nn.functional.softmax(temp, dim=-4)  # 每个Mixture进行一次归一
#         self.weights = torch.reshape(temp, (-1, self.M * self.K, 1, 1))
#         ###
#
#         return self.sigma,self.means,self.weights

##for y1 gmm _same_resolution
class gmm_hyper_y1_same_resolution(nn.Module):
    def __init__(self,N,M,K): #K表示GMM对应正态分布个数
        super().__init__()
        self.N = N
        self.M = M
        self.K = K

        # 已经同分辨率，不需要上采样
        self.gmm_sigma = nn.Sequential(
            deconv(in_channels=6*M,out_channels=6*M,kernel_size=1,stride=1), #,stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.ReLU(inplace=True),

            deconv(in_channels=6*M, out_channels=4*M, kernel_size=1,stride=1),# stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.ReLU(inplace=True),

            conv(in_channels=4*M,out_channels=(M*K),kernel_size=1,stride=1), #padding=kernel_size//2
            nn.ReLU(inplace=True),
        )

        self.gmm_means = nn.Sequential(
            deconv(in_channels=6*M, out_channels=6*M, kernel_size=1,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            deconv(in_channels=6*M, out_channels=4*M, kernel_size=1,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            conv(in_channels=4*M, out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
        )


        self.gmm_weights = nn.Sequential(
            deconv(in_channels=6*M, out_channels=6*M, kernel_size=1,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            deconv(in_channels=6*M, out_channels=(M*K), kernel_size=1,stride=1),
            # # stride=2,padding=kernel_size//2,output_padding=stride-1
            # # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
            #
            # spatial_pool2d(), #joint+gmm换1x1卷积不再需要
            nn.LeakyReLU(inplace=True),

            conv(in_channels=(M*K), out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
            #出去后要接一个softmax层，表示概率！！
        )

    def forward(self,z1):
        self.sigma = self.gmm_sigma(z1)
        self.means = self.gmm_means(z1)
        #weights要加softmax！！  nn.functional.softmax
        #softmax 加的有问题，需要按照
        # self.weights = nn.functional.softmax(self.gmm_weights(z1),dim = -3)
        #修正
        # print("z1",z1.shape)
        # print(self.gmm_weights(z1).shape)
        # raise ValueError("shape stop")

        temp = torch.reshape(self.gmm_weights(z1), (-1, self.K, self.M, z1.shape[-2],z1.shape[-1]))  # 方便后续reshape合并时同一个M的数据相邻成组
        # temp = torch.reshape(self.gmm_weights(z1),(-1,self.K,self.M,1,1)) #方便后续reshape合并时同一个M的数据相邻成组

        temp = nn.functional.softmax(temp,dim=-4)  #每个Mixture进行一次归一
        self.weights = torch.reshape(temp, (-1, self.M * self.K, z1.shape[-2],z1.shape[-1]))
        # self.weights = torch.reshape(temp,(-1,self.M*self.K,1,1))
        ###
        return self.sigma,self.means,self.weights

##z2_up+y1 for y2 gmm _same_resolution
# class gmm_hyper_y2_same_resolution(nn.Module):
#     def __init__(self,N,M,K): #K表示GMM对应正态分布个数
#         super().__init__()
#         self.N = N
#         self.M = M
#         self.K = K
#
#         # self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=4) #固定为4倍 因为z和y分辨率本身就差4倍
#
#         # 输入与y1分辨率相同，但通道数不同 2倍4M+2M -> 6*M
#         self.gmm_sigma = nn.Sequential(
#             conv(in_channels=6*M,out_channels=6*M,kernel_size=1,stride=1),
#             nn.ReLU(inplace=True),
#
#             conv(in_channels=6*M,out_channels=4*M,kernel_size=1,stride=1),
#             nn.ReLU(inplace=True),
#
#             conv(in_channels=4*M,out_channels=(M*K),kernel_size=1,stride=1), #padding=kernel_size//2
#             nn.ReLU(inplace=True),
#         )
#
#         self.gmm_means = nn.Sequential(
#             conv(in_channels=6*M,out_channels=6*M,kernel_size=1,stride=1),
#             nn.LeakyReLU(inplace=True),
#
#             conv(in_channels=6*M,out_channels=4*M,kernel_size=1,stride=1),
#             # stride=2,padding=kernel_size//2,output_padding=stride-1
#             nn.LeakyReLU(inplace=True),
#
#             conv(in_channels=4*M, out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
#         )
#
#
#         self.gmm_weights = nn.Sequential(
#             conv(in_channels=6*M,out_channels=6*M,kernel_size=1,stride=1),
#             nn.LeakyReLU(inplace=True),
#
#             conv(in_channels=6*M,out_channels= (M*K),kernel_size=1,stride=1),
#             # # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
#             # spatial_pool2d(), #joint+gmm换1x1卷积不再需要
#             nn.LeakyReLU(inplace=True),
#
#             conv(in_channels=(M*K), out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
#             #出去后要接一个softmax层，表示概率！！
#         )
#
#     def forward(self,allconcat):
#         # self.up_z2 = self.upsample_layer(z2)
#         # self.cat_in = torch.cat((self.up_z2,y1),dim=-3)
#
#         self.sigma = self.gmm_sigma(allconcat)
#         self.means = self.gmm_means(allconcat)
#         # #softmax!!
#         # # self.weights = nn.functional.softmax(self.gmm_weights(self.cat_in),dim=-3)
#         # # 修正
#         # temp = torch.reshape(self.gmm_weights(allconcat), (-1, self.K, self.M,  1, 1))  # 方便后续reshape合并时同一个M的数据相邻成组
#         # temp = nn.functional.softmax(temp, dim=-4)  # 每个Mixture进行一次归一
#         # self.weights = torch.reshape(temp, (-1, self.M * self.K, 1, 1))
#         ###
#         temp = torch.reshape(self.gmm_weights(allconcat),
#                              (-1, self.K, self.M, allconcat.shape[-2], allconcat.shape[-1]))  # 方便后续reshape合并时同一个M的数据相邻成组
#         # temp = torch.reshape(self.gmm_weights(allconcat),(-1,self.K,self.M,1,1)) #方便后续reshape合并时同一个M的数据相邻成组
#
#         temp = nn.functional.softmax(temp, dim=-4)  # 每个Mixture进行一次归一
#         self.weights = torch.reshape(temp, (-1, self.M * self.K, allconcat.shape[-2], allconcat.shape[-1]))
#         # self.weights = torch.reshape(temp,(-1,self.M*self.K,1,1))
#         ###
#
#         return self.sigma,self.means,self.weights

###################################
# class Encoder1(nn.Module):
#     def __init__(self, N, M, **kwargs):
#         super().__init__()
#         self.g_a_conv1 = conv(3, N)
#         self.g_a_gdn1 = GDN(N)
#         self.g_a_conv2 = conv(N, N)
#         self.g_a_gdn2 = GDN(N)
#         self.g_a_conv3 = conv(N, N)
#         self.g_a_gdn3 = GDN(N)
#         self.g_a_conv4 = conv(N, M)
#
#     def forward(self, x):
#         # self.y = self.g_a(x)
#         self.g_a_c1 = self.g_a_conv1(x) #Tensor
#         self.g_a_g1 = self.g_a_gdn1(self.g_a_c1)
#         self.g_a_c2 = self.g_a_conv2(self.g_a_g1)  # Tensor
#         self.g_a_g2 = self.g_a_gdn2(self.g_a_c2)
#         self.g_a_c3 = self.g_a_conv3(self.g_a_g2)  # Tensor
#         self.g_a_g3 = self.g_a_gdn3(self.g_a_c3)
#         self.g_a_c4 = self.g_a_conv4(self.g_a_g3)  # Tensor
#         self.y = self.g_a_c4
#         return self.y,self.g_a_g1,self.g_a_g2,self.g_a_g3
#
class Decoder1(nn.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.g_s_conv1 = deconv(M, N)
        self.g_s_gdn1 = GDN(N, inverse=True)
        self.g_s_conv2 = deconv(N, N)
        self.g_s_gdn2 = GDN(N, inverse=True)
        self.g_s_conv3 = deconv(N, N)
        self.g_s_gdn3 = GDN(N, inverse=True)
        self.g_s_conv4 = deconv(N, 3)

    def forward(self, y_hat):
        # self.x_hat = self.g_s(self.y_hat)
        self.g_s_c1 = self.g_s_conv1(y_hat)  # Tensor
        self.g_s_g1 = self.g_s_gdn1(self.g_s_c1)
        self.g_s_c2 = self.g_s_conv2(self.g_s_g1)  # Tensor
        self.g_s_g2 = self.g_s_gdn2(self.g_s_c2)
        self.g_s_c3 = self.g_s_conv3(self.g_s_g2)  # Tensor
        self.g_s_g3 = self.g_s_gdn3(self.g_s_c3)
        self.g_s_c4 = self.g_s_conv4(self.g_s_g3)  # Tensor
        self.x_hat = self.g_s_c4
        return self.x_hat,self.g_s_g1,self.g_s_g2,self.g_s_g3
#
class Encoder2(nn.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.pre_conv = conv(6,3,stride=1) #不缩放！
        self.pre_gdn = GDN(3)

        self.g_a_conv1 = conv(3, N)
        self.g_a_gdn1 = GDN(N)
        self.g_a_conv2 = conv(N, N)
        self.g_a_gdn2 = GDN(N)
        self.g_a_conv3 = conv(N, N)
        self.g_a_gdn3 = GDN(N)
        self.g_a_conv4 = conv(N, M)

    def forward(self, x1_warp,x2):
        # self.y = self.g_a(x)
        # self.g_a_c1 = self.g_a_conv1(x) #Tensor
        self.pre1 = self.pre_conv(torch.cat((x1_warp,x2),dim=-3))
        self.pre2 = self.pre_gdn(self.pre1)

        self.g_a_c1 = self.g_a_conv1(self.pre2) #直接concat
        #
        self.g_a_g1 = self.g_a_gdn1(self.g_a_c1)
        self.g_a_c2 = self.g_a_conv2(self.g_a_g1)  # Tensor
        self.g_a_g2 = self.g_a_gdn2(self.g_a_c2)
        self.g_a_c3 = self.g_a_conv3(self.g_a_g2)  # Tensor
        self.g_a_g3 = self.g_a_gdn3(self.g_a_c3)
        self.g_a_c4 = self.g_a_conv4(self.g_a_g3)  # Tensor
        self.y = self.g_a_c4
        return self.pre1, self.y#,self.g_a_g1,self.g_a_g2,self.g_a_g3

class Decoder2(nn.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.g_s_conv1 = deconv(M, N)
        self.g_s_gdn1 = GDN(N, inverse=True)
        self.g_s_conv2 = deconv(N, N)
        self.g_s_gdn2 = GDN(N, inverse=True)
        self.g_s_conv3 = deconv(N, N)
        self.g_s_gdn3 = GDN(N, inverse=True)
        self.g_s_conv4 = deconv(N, 3)

        # #
        # self.after_gdn = GDN(3, inverse=True)
        # self.after_conv = deconv(6, 3, stride=1)  # 不缩放！
        # # self.after_gdn = GDN(6, inverse=True)
        # # self.after_conv = deconv(6,3,stride=1) #不缩放！


    def forward(self, y_hat): #,x1_hat_warp):
        # self.x_hat = self.g_s(self.y_hat)
        self.g_s_c1 = self.g_s_conv1(y_hat)  # Tensor
        self.g_s_g1 = self.g_s_gdn1(self.g_s_c1)
        self.g_s_c2 = self.g_s_conv2(self.g_s_g1)  # Tensor
        self.g_s_g2 = self.g_s_gdn2(self.g_s_c2)
        self.g_s_c3 = self.g_s_conv3(self.g_s_g2)  # Tensor
        self.g_s_g3 = self.g_s_gdn3(self.g_s_c3)
        self.g_s_c4 = self.g_s_conv4(self.g_s_g3)  # Tensor
        #
        # self.after1 = self.after_gdn(self.g_s_c4)
        # self.after2 = self.after_conv(torch.cat((self.after1,x1_hat_warp),dim=-3))

        self.x_hat = self.g_s_c4
        # self.x_hat = self.g_s_c4
        return self.x_hat#,self.g_s_g1,self.g_s_g2,self.g_s_g3

class Decodercat(nn.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.after_gdn = GDN(3, inverse=True)
        self.after_conv = deconv(6, 3, stride=1)  # 不缩放！
        # self.after_gdn = GDN(6, inverse=True)
        # self.after_conv = deconv(6,3,stride=1) #不缩放！

    def forward(self, x1_hat,x2_hat_warp):
        self.after1 = self.after_gdn(x1_hat)
        self.after2 = self.after_conv(torch.cat((self.after1,x2_hat_warp),dim=-3))

        self.x_hat = self.after2
        return self.x_hat


###########################################################################


class HSIC(CompressionModel):
    def __init__(self,N=128,M=192,K=5,**kwargs): #'cuda:0' or 'cpu'
        super().__init__(entropy_bottleneck_channels=N, **kwargs)
        # super(DSIC, self).__init__()
        # self.entropy_bottleneck1 = CompressionModel(entropy_bottleneck_channels=N)
        # self.entropy_bottleneck2 = CompressionModel(entropy_bottleneck_channels=N)
        self.gaussian1 = GaussianMixtureConditional_gf(K = K)
        self.gaussian2 = GaussianMixtureConditional_gf(K = K)
        self.N = int(N)
        self.M = int(M)
        self.K = int(K)
        #定义组件
        self.encoder1 = Encoder2(N,M)
        self.encoder2 = Encoder2(N,M)
        self.decoder1 = Decoder2(N,M)
        self.decoder2 = Decoder2(N,M)

        self.decodercat1 = Decodercat(N, M)
        self.decodercat2 = Decodercat(N, M)
        # pic2 需要的组件


        #hyper
        self._h_a1 = encode_hyper(N=N,M=M)
        self._h_a2 = encode_hyper(N=N,M=M)
        # self._h_s1 = gmm_hyper_y1(N=N,M=M,K=K)
        # self._h_s2 = gmm_hyper_y2(N=N,M=M,K=K)

        #先将z上采样为y的大小，再和自回归的内容一起不变分辨率的卷出GMM
        self.h_s1_up = nn.Sequential(
            deconv(2*N, 2*M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(2*M, 2*M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(2*M * 3 // 2, 2*M * 2, stride=1, kernel_size=3),
        )
        self.h_s2_up = nn.Sequential(
            deconv(2*N, 2*M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(2*M, 2*M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(2*M * 3 // 2, 2*M * 2, stride=1, kernel_size=3),
        )

        self.context_prediction1 = MaskedConv2d(M,
                                               2 * M,
                                               kernel_size=5,
                                               padding=2,
                                               stride=1)

        self.context_prediction2 = MaskedConv2d(M,
                                                2 * M,
                                                kernel_size=5,
                                                padding=2,
                                                stride=1)
        self._h_s1_same_resolution = gmm_hyper_y1_same_resolution(N=N, M=M, K=K)
        self._h_s2_same_resolution = gmm_hyper_y1_same_resolution(N=N, M=M, K=K) #相同结构

        # self.need_int = need_int

        #quantize
    def _quantize(self, inputs, mode, means=None):
        # type: (Tensor, str, Optional[Tensor]) -> Tensor
        if mode not in ('noise', 'dequantize', 'symbols'):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == 'noise':
            if torch.jit.is_scripting():
                half = float(0.5)
                noise = torch.empty_like(inputs).uniform_(-half, half)
            else:
                noise = self._get_noise_cached(inputs)
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == 'dequantize':
            if means is not None:
                outputs += means
            return outputs

        assert mode == 'symbols', mode
        outputs = outputs.int()
        return outputs
    #erfc
    def _standardized_cumulative(self, inputs):
        # type: (Tensor) -> Tensor
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def forward(self,x1,x2,h_matrix):
        h_inv = torch.inverse(h_matrix)
        # encoder
        x1_warp = kornia.warp_perspective(x1, h_matrix, (x1.size()[-2], x1.size()[-1]))
        x2_concat_input,y2 = self.encoder2(x1_warp, x2)
        ##end encoder

        # hyper for pic2
        z2 = self._h_a2(y2)
        z2_hat, z2_likelihoods = self.entropy_bottleneck2(z2)

        x2_warp_new = kornia.warp_perspective(x2_concat_input, h_inv, (x2.size()[-2], x2.size()[-1]))
        # 定义结构
        _,y1 = self.encoder1(x2_warp_new, x1)
        z1 = self._h_a1(y1)
        # print(z1.device)
        z1_hat, z1_likelihoods = self.entropy_bottleneck1(z1)

        ###
        # gmm2 = self._h_s2(z2_hat, y1_hat)  # 三要素
        ###
        #############################################

        #z1 z2一同输入
        params1 = self.h_s1_up(torch.cat((z1_hat,z2_hat), dim=1)) # torch.Size([1, 384, 32, 32])
        y1_hat = self.gaussian1._quantize(  # pylint: disable=protected-access
            y1, 'noise' if self.training else 'dequantize')
        ctx_params1 = self.context_prediction1(y1_hat)  # 用两次！！ 2M # torch.Size([1, 384, 32, 32])
        # gaussian_params1 = self.entropy_parameters1(
        #     torch.cat((params1, ctx_params1), dim=1))
        gmm1_jg = self._h_s1_same_resolution(torch.cat((params1, ctx_params1), dim=1)) #通道数 4M
        ###
        y1_hat, y1_likelihoods = self.gaussian1(y1, gmm1_jg[0],gmm1_jg[1],gmm1_jg[2])  # sigma 每个都是M通道 与y1同

        # #save
        # save_y1_hat = y1_hat.cpu().numpy()
        # data = {"y1_hat":save_y1_hat}
        # np.save('y1_hat.npy',data)
        #
        # raise ValueError("stop")
        # #end save

        params2 = self.h_s2_up(torch.cat((z2_hat,z1_hat), dim=1))
        y2_hat = self.gaussian2._quantize(  # pylint: disable=protected-access
            y2, 'noise' if self.training else 'dequantize')
        ctx_params2 = self.context_prediction2(y2_hat)
        # gaussian_params2 = self.entropy_parameters2(
        #     torch.cat((params2, ctx_params2, ctx_params1), dim=1))
        gmm2_jq = self._h_s2_same_resolution(torch.cat((params2, ctx_params2), dim=1)) #通道数 4M
        ###
        y2_hat, y2_likelihoods = self.gaussian2(y2, gmm2_jq[0], gmm2_jq[1], gmm2_jq[2])  # 这里也是临时，待改gmm
        # end hyper for pic2

        ##decoder

        x1_hat_temp = self.decoder1(y1_hat)
        x2_hat_temp = self.decoder2(y2_hat)

        #串行解出 方案1：2-》1-》2
        x2_hat_warp = kornia.warp_perspective(x2_hat_temp, h_inv, (x2_hat_temp.size()[-2], x2_hat_temp.size()[-1]))
        x1_hat = self.decodercat1(x1_hat_temp, x2_hat_warp)

        x1_hat_warp_new = kornia.warp_perspective(x1_hat, h_matrix, (x1_hat_temp.size()[-2],x1_hat_temp.size()[-1]))
        x2_hat = self.decodercat1(x2_hat_temp, x1_hat_warp_new)

        #end decoder
        # print(x1.size())

        return {
            'x1_hat': x1_hat,
            'x2_hat': x2_hat,
            'y1_hat': y1_hat,
            'z1_hat': z1_hat,
            'likelihoods':{
                'y1': y1_likelihoods,
                'y2': y2_likelihoods,
                'z1': z1_likelihoods,
                'z2': z2_likelihoods,
            }
        }


    ###codec
    def compress(self,x1,x2,h_matrix,output_name,output_path="",device="cpu"):
        # 定义结构
        y1, g1_1, g1_2, g1_3 = self.encoder1(x1)
        z1 = self._h_a1(y1)
        # print(z1.device)
        # z1_hat, z1_likelihoods = self.entropy_bottleneck1(z1)
        # z1_hat,z1_likelihoods = self.entropy_bottleneck1(z1)
        # compress
        z1_strings = self.entropy_bottleneck1.compress(z1)
        z1_hat = self.entropy_bottleneck1.decompress(z1_strings, z1.size()[-2:])  # z解码后结果（压缩时仍需要）
        print(z1_strings)

        # change for jg
        params1 = self.h_s1_up(z1_hat)  # torch.Size([1, 384, 32, 32])
        #############################################
        # encoder
        x1_warp = kornia.warp_perspective(x1, h_matrix, (x1.size()[-2], x1.size()[-1]))
        y2 = self.encoder2(x1_warp, x2)
        ##end encoder
        # hyper for pic2
        z2 = self._h_a2(y2)
        # z2_hat, z2_likelihoods = self.entropy_bottleneck2(z2)
        z2_strings = self.entropy_bottleneck2.compress(z2)
        z2_hat = self.entropy_bottleneck2.decompress(z2_strings, z2.size()[-2:])  # z解码后结果（压缩时仍需要）

        params2 = self.h_s2_up(z2_hat)
        #已得到param1,2
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        y1_height = z1_hat.size(2) * s
        y1_width = z1_hat.size(3) * s
        y1_hat = self._quantize(y1,'dequantize',means=None)
        y1_hat = F.pad(y1_hat, (padding, padding, padding, padding))
        y1_strings = []

        y2_hat = self._quantize(y2,'dequantize',means=None)
        y2_hat = F.pad(y2_hat, (padding, padding, padding, padding))
        y2_strings = []
        #################
        ######encoding###
        #################
        y1_hat_cpu_np = y1_hat.cpu().numpy().astype('int')
        y2_hat_cpu_np = y2_hat.cpu().numpy().astype('int')
        # print(y1_hat.shape)
        # # print(y1_hat_cpu_np)
        #
        # raise  ValueError("stop 0")
        # ========debug
        # print(z1_strings)
        # print(len(z1_strings))
        # print(z1_strings[0].shape)

        # ========encoding for header of z1 and z2
        output1 = os.path.join(output_path, str(output_name) + ".npz")
        if os.path.exists(output1):
            os.remove(output1)
        fileobj = open(output1, mode='wb')
        # 图片尺寸信息
        fileobj.write(np.array(x1.shape[2:], dtype=np.uint16).tobytes())  # 4bytes 512x512
        # 非零通道号信息
        flag1 = np.zeros(y1_hat.shape[1], dtype=np.int)
        flag2 = np.zeros(y2_hat.shape[1], dtype=np.int)
        for ch_idx in range(y1_hat.shape[1]):
            if np.sum(abs(y1_hat_cpu_np[:, ch_idx, :, :])) > 0:
                flag1[ch_idx] = 1
        non_zero_idx_1 = np.squeeze(np.where(flag1 == 1))
        num1 = np.packbits(np.reshape(flag1, [8, y1_hat.shape[1] // 8]))
        minmax1 = np.maximum(abs(y1_hat_cpu_np.max()), abs(y1_hat_cpu_np.min()))
        minmax1 = int(np.maximum(minmax1, 1))
        ###
        for ch_idx in range(y2_hat.shape[1]):
            if np.sum(abs(y2_hat_cpu_np[:, ch_idx, :, :])) > 0:
                flag2[ch_idx] = 1
        non_zero_idx_2 = np.squeeze(np.where(flag2 == 1))
        num2 = np.packbits(np.reshape(flag2, [8, y2_hat.shape[1] // 8]))
        minmax2 = np.maximum(abs(y2_hat_cpu_np.max()), abs(y2_hat_cpu_np.min()))
        minmax2 = int(np.maximum(minmax2, 1))
        ########
        fileobj.write(np.array([len(z1_strings[0]), minmax1], dtype=np.uint16).tobytes())  # 4bytes
        fileobj.write(np.array(num1, dtype=np.uint8).tobytes())  # y_hat_channels_number // 8  eg:192//8=24
        fileobj.write(z1_strings[0])  # 按照len(z1_strings)来
        #
        fileobj.write(np.array([len(z2_strings[0]), minmax2], dtype=np.uint16).tobytes())  # 4bytes
        fileobj.write(np.array(num2, dtype=np.uint8).tobytes())  # y_hat_channels_number // 8  eg:192//8=24
        fileobj.write(z2_strings[0])  # 按照len(z2_strings)来

        fileobj.close()

        # ###
        # print("1 and 2:",bool(non_zero_idx_2==non_zero_idx_1))
        # raise  ValueError("stop test")
        # ###
        # ========encoding for range coder of y1 and y2
        output2 = os.path.join(output_path, str(output_name) + '.bin')
        if os.path.exists(output2):
            os.remove(output2)
        encoder = RangeEncoder(output2)
        # TINY = 1e-10
        # samples1 = np.arange(0, minmax1*2+1)
        # samples1 = torch.Tensor(samples1).to('cuda:0')
        samples1_np = np.arange(0, minmax1 * 2 + 1)
        samples2_np = np.arange(0, minmax2 * 2 + 1)

        start = time.time()
        # y1
        # for h_idx in range(y2_hat_cpu_np.shape[2]):
        #     for w_idx in range(y2_hat_cpu_np.shape[3]):
        # 如果是joint一类，在此计算当前像素点的概率
        # print("non_zero_idx_1",non_zero_idx_1)
        # print("y1hat_shape",y1_hat.shape)
        ##joint+gmm必须像素优先--maskcnn #y1 y2 通用
        for h_idx in range(y1_height): #y1 y2 通用 #y1_hat_cpu_np.shape[2]
            for w_idx in range(y1_width): #y1 y2 通用
                print(h_idx,",",w_idx)
                ##########################
                #y1
                #jg_codec
                y1_crop = y1_hat[:, :, h_idx:h_idx + kernel_size, w_idx:w_idx + kernel_size]
                # ctx_params1 = self.context_prediction1(y1_crop)
                #
                # # 1x1 conv for the entropy parameters prediction network, so
                # # we only keep the elements in the "center"
                # ctx_p1 = ctx_params1[:, :, padding:padding + 1, padding:padding + 1]

                ctx_p1 = F.conv2d(
                    y1_crop,
                    self.context_prediction1.weight,
                    bias=self.context_prediction1.bias,
                )

                p1 = params1[:, :, h_idx:h_idx + 1, w_idx:w_idx + 1]
                # print("ctx_p1 shape",ctx_p1.shape)
                # print("p1 shape",p1.shape)
                # gaussian_params1 = self.entropy_parameters(torch.cat((p1, ctx_p1), dim=1))
                gmm1_jg = self._h_s1_same_resolution(torch.cat((p1, ctx_p1), dim=1))  # 通道数 4M
                # print("gmm1_jg shape",gmm1_jg[1].shape)
                # raise  ValueError("stop 111") #[1,960,1,1]
                #y2
                y2_crop = y2_hat[:, :, h_idx:h_idx + kernel_size, w_idx:w_idx + kernel_size]
                # ctx_params2 = self.context_prediction2(y2_crop)
                # # 1x1 conv for the entropy parameters prediction network, so
                # # we only keep the elements in the "center"
                # ctx_p2 = ctx_params2[:, :, padding:padding + 1, padding:padding + 1]
                ctx_p2 = F.conv2d(
                    y2_crop,
                    self.context_prediction2.weight,
                    bias=self.context_prediction2.bias,
                )
                p2 = params2[:, :, h_idx:h_idx + 1, w_idx:w_idx + 1]
                gmm2_jg = self._h_s2_same_resolution(torch.cat((p2, ctx_p2, ctx_p1), dim=1))  # 通道数 4M

                # gaussian_params2 = self.entropy_parameters2(
                #     torch.cat((params2, ctx_params2, ctx_params1), dim=1))


                # for i in range(len(non_zero_idx_1)):
                for ch_idx in range(y1_hat_cpu_np.shape[1]):
                    if ch_idx in non_zero_idx_1:
                        samples1 = torch.Tensor(samples1_np).to(device)
                        samples1 = samples1.reshape([samples1.shape[0], 1, 1])
                        # samples1 = samples1.expand(samples1.shape[0], y1_hat_cpu_np.shape[2],
                        #                            y1_hat_cpu_np.shape[3])  # [arange(xx),y1_height,y1_width]
                        samples1 = samples1.expand(samples1.shape[0], y1_height,
                                                   y1_width)  # [arange(xx),y1_height,y1_width]
                        # 非零通道：一个通道算一次-> 并行优化
                        # ch_idx = non_zero_idx_1[i]  # 非零通道

                        ####begin-pmf-calulation##############
                        ######################################
                        # gmm1_jg = [scales, means,weights] 通道数均为M*K
                        ch_idx_list = [ch_idx + temp_i * self.M for temp_i in range(self.K)]
                        sigma1 = gmm1_jg[0][0, ch_idx_list, :, :]
                        means1 = gmm1_jg[1][0, ch_idx_list, :, :] + minmax1  # 注意这里~~~ 相当于平移
                        weights1 = gmm1_jg[2][0, ch_idx_list, 0, 0]  # (M*k):((k+1)*M) #1x1

                        # print(sigma1.shape)
                        # raise ValueError("stop shape")

                        # 计算pmf
                        pmf = None
                        for temp_K in range(self.K):
                            half = float(0.5)

                            values = samples1 - means1[temp_K]

                            temp_scales = self.gaussian1.lower_bound_scale(
                                sigma1[temp_K])  # scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

                            values = abs(values)
                            upper = self._standardized_cumulative((half - values) / temp_scales)
                            lower = self._standardized_cumulative((-half - values) / temp_scales)
                            if pmf == None:
                                pmf = (upper - lower) * weights1[temp_K]  # 指定对应的M个通道
                            else:
                                pmf += (upper - lower) * weights1[temp_K]  # 指定对应的M个通道

                        # print(pmf.shape)
                        # raise ValueError("stop")
                        # print("minmax1",minmax1)
                        pmf = pmf.cpu().numpy()
                        # print(pmf.shape)
                        # raise ValueError("stop")
                        ####end-pmf-calulation################
                        ######################################
                        # pmf_temp = pmf[:, h_idx + padding, w_idx + padding]
                        pmf_temp = pmf[:, h_idx, w_idx]
                        # print(pmf_temp.shape)
                        # raise ValueError("stop")
                        # To avoid the zero-probability
                        pmf_clip = np.clip(pmf_temp, 1.0 / 65536, 1.0)
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                        cdf = list(np.add.accumulate(pmf_clip))
                        cdf = [0] + [int(i) for i in cdf]

                        symbol = np.int(y1_hat[0, ch_idx, h_idx + padding, w_idx + padding] + minmax1)
                        # print("len-cdf",len(cdf))
                        # raise ValueError("stop")
                        encoder.encode([symbol], cdf)
                        # end y1

                    # if ch_idx in non_zero_idx_2:
                    #     samples2 = torch.Tensor(samples2_np).to(device)
                    #     samples2 = samples2.reshape([samples2.shape[0], 1, 1])
                    #     # samples2 = samples2.expand(samples2.shape[0], y2_hat_cpu_np.shape[2],
                    #     #                            y2_hat_cpu_np.shape[3])  # [arange(xx),width,height]
                    #     samples2 = samples2.expand(samples2.shape[0], y1_height,
                    #                                y1_width)  # [arange(xx),width,height]
                    #     # 非零通道：一个通道算一次-> 并行优化
                    #     # ch_idx = non_zero_idx_[i]  # 非零通道
                    #
                    #     ####begin-pmf-calulation##############
                    #     ######################################
                    #     # gmm2_jg = [scales, means,weights] 通道数均为M*K
                    #     ch_idx_list = [ch_idx + temp_i * self.M for temp_i in range(self.K)]
                    #     sigma2 = gmm2_jg[0][0, ch_idx_list, :, :]
                    #     means2 = gmm2_jg[2][0, ch_idx_list, :, :] + minmax2  # 注意这里~~~ 相当于平移
                    #     weights2 = gmm2_jg[2][0, ch_idx_list, 0, 0]  # (M*k):((k+1)*M) #1x1
                    #
                    #     # 计算pmf
                    #     pmf = None
                    #     for temp_K in range(self.K):
                    #         half = float(0.5)
                    #
                    #         values = samples2 - means2[temp_K]
                    #
                    #         temp_scales = self.gaussian2.lower_bound_scale(
                    #             sigma2[temp_K])  # scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用
                    #
                    #         values = abs(values)
                    #         upper = self._standardized_cumulative((half - values) / temp_scales)
                    #         lower = self._standardized_cumulative((-half - values) / temp_scales)
                    #         if pmf == None:
                    #             pmf = (upper - lower) * weights2[temp_K]  # 指定对应的M个通道
                    #         else:
                    #             pmf += (upper - lower) * weights2[temp_K]  # 指定对应的M个通道
                    #
                    #     # print(pmf.shape)
                    #     # raise ValueError("stop")
                    #     # print("minmax2",minmax2)
                    #     pmf = pmf.cpu().numpy()
                    #     # print(pmf.shape)
                    #     # raise ValueError("stop")
                    #     ####end-pmf-calulation################
                    #     ######################################
                    #     # pmf_temp = pmf[:, h_idx + padding, w_idx + padding]
                    #     pmf_temp = pmf[:, h_idx, w_idx]
                    #     # print(pmf_temp.shape)
                    #     # raise ValueError("stop")
                    #     # To avoid the zero-probability
                    #     pmf_clip = np.clip(pmf_temp, 1.0 / 65536, 1.0)
                    #     pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    #     cdf = list(np.add.accumulate(pmf_clip))
                    #     cdf = [0] + [int(i) for i in cdf]
                    #
                    #     symbol = np.int(y2_hat[0, ch_idx, h_idx + padding, w_idx + padding] + minmax2)
                    #     # print("len-cdf",len(cdf))
                    #     # raise ValueError("stop")
                    #     encoder.encode([symbol], cdf)
                    #     # end y2

        ##end encoding
        encoder.close()
        end = time.time()
        delta_time = (end - start)
        # return y2_hat[:, :, padding:-padding, padding:-padding]
        # raise  ValueError("stop encoder.")
        # y2

        # end y2

        num_pixels = x1.shape[2] * x1.shape[3] * 2  # 双目两张图
        size_real = os.path.getsize(output1) + os.path.getsize(output2)

        bpp_real = (os.path.getsize(output1) + os.path.getsize(output2)) * 8 / num_pixels  # 主要看这个
        bpp_side = (os.path.getsize(output1)) * 8 / num_pixels

        # print("Time : {:0.3f}".format(end-start))
        print("bpp_real:", bpp_real)

        return {
            'bpp_real': bpp_real,
            'enctime': delta_time,

            'y1_hat': y1_hat,
            'y2_hat': y2_hat,
            'z1_hat': z1_hat,
            'z2_hat': z2_hat,

        }

    #decompress
    def decompress(self,x1,x2,h_matrix,output_name,output_path="",device="cpu"):
        output1 = os.path.join(output_path,str(output_name)+ '.npz')
        output2 = os.path.join(output_path,str(output_name)+ '.bin')

        #=================decoding for z1 and z2
        fileobj = open(output1, mode='rb')
        x_shape = np.frombuffer(fileobj.read(4), dtype=np.uint16) #图片尺寸 [width,height] array([512, 512], dtype=uint16)
        #z1
        length1, minmax1 = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        num1 = np.frombuffer(fileobj.read(self.M//8), dtype=np.uint8)
        string1 = fileobj.read(length1)
        #z2
        length2, minmax2 = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        num2 = np.frombuffer(fileobj.read(self.M//8), dtype=np.uint8) #192//8=24
        string2 = fileobj.read(length2)
        fileobj.close()

        flag1 = np.unpackbits(num1)
        non_zero_idx_1 = np.squeeze(np.where(flag1 == 1))
        flag2 = np.unpackbits(num2)
        non_zero_idx_2 = np.squeeze(np.where(flag2 == 1))
        # print(len(string1))
        # print(len(string2))
        # print(minmax1)
        # print(minmax2)
        # z1_strings = torch.Tensor(string1).unsqueeze(0).to(device)
        # z2_strings = torch.Tensor(string2).unsqueeze(0).to(device)

        #
        # x_shape = torch.Tensor(x_shape).int().to(device)
        y_shape = x_shape//16
        z_shape = y_shape//4

        z1_hat = self.entropy_bottleneck1.decompress([string1], z_shape)  # z解码后结果
        z2_hat = self.entropy_bottleneck2.decompress([string2], z_shape)  # z解码后结果

        #change for jq
        params1 = self.h_s1_up(z1_hat) # torch.Size([1, 384, 32, 32])
        params2 = self.h_s2_up(z2_hat)

        #===================decoding for y1 and y2
        # samples1 = np.arange(0, minmax1*2+1)
        # samples1 = torch.Tensor(samples1).to('cuda:0')
        samples1_np = np.arange(0, minmax1*2+1)
        samples2_np = np.arange(0, minmax2*2+1)

        start = time.time()
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        y1_height = z1_hat.size(2) * s
        y1_width = z1_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        # yapf: disable
        y1_hat = torch.zeros((z1_hat.size(0), self.M, y1_height + 2 * padding, y1_width + 2 * padding),
                             device=z1_hat.device)
        y2_hat = torch.zeros((z1_hat.size(0), self.M, y1_height + 2 * padding, y1_width + 2 * padding),
                             device=z2_hat.device)
        # print("decy1hatdtype",y1_hat.dtype)
        #
        # # y1_hat = np.zeros([1] + [y_shape[1]+kernel_size-1] + [y_shape[2]+kernel_size-1] + [num_filters])
        # y1_hat = np.zeros([1]+[self.M]+[y_shape[0],y_shape[1]]) #[1,192,32,32]
        # y2_hat = np.zeros([1]+[self.M]+[y_shape[0],y_shape[1]]) #[1,192,32,32]

        decoder = RangeDecoder(output2)
        start = time.time()
        #gpu cpu?
        # gpucpufile = str(device)+".txt"
        # gpucpuwrite = open(gpucpufile,"w")
        for h_idx in range(y1_height): #y1 y2 通用 #y1_hat_cpu_np.shape[2]
            for w_idx in range(y1_width): #y1 y2 通用
                print(h_idx,",",w_idx)
                ##########################
                #y1
                #jg_codec
                y1_crop = y1_hat[:, :, h_idx:h_idx + kernel_size, w_idx:w_idx + kernel_size]
                # ctx_params1 = self.context_prediction1(y1_crop)
                #
                # # 1x1 conv for the entropy parameters prediction network, so
                # # we only keep the elements in the "center"
                # ctx_p1 = ctx_params1[:, :, padding:padding + 1, padding:padding + 1]
                ##########
                ctx_p1 = F.conv2d(
                    y1_crop,
                    self.context_prediction1.weight,
                    bias=self.context_prediction1.bias,
                )
                ###########
                p1 = params1[:, :, h_idx:h_idx + 1, w_idx:w_idx + 1]
                # print("ctx_p1 shape",ctx_p1.shape)
                # print("p1 shape",p1.shape)
                # gaussian_params1 = self.entropy_parameters(torch.cat((p1, ctx_p1), dim=1))
                gmm1_jg = self._h_s1_same_resolution(torch.cat((p1, ctx_p1), dim=1))  # 通道数 4M
                # print("gmm1_jg shape",gmm1_jg[1].shape)
                # raise  ValueError("stop 111") #[1,960,1,1]
                # print("gmm1_jg",gmm1_jg)
                # raise ValueError("stop gmm1_jg")
                # y2
                y2_crop = y2_hat[:, :, h_idx:h_idx + kernel_size, w_idx:w_idx + kernel_size]
                # ctx_params2 = self.context_prediction2(y2_crop)
                # # 1x1 conv for the entropy parameters prediction network, so
                # # we only keep the elements in the "center"
                # ctx_p2 = ctx_params2[:, :, padding:padding + 1, padding:padding + 1]

                ctx_p2 = F.conv2d(
                    y2_crop,
                    self.context_prediction2.weight,
                    bias=self.context_prediction2.bias,
                )

                p2 = params2[:, :, h_idx:h_idx + 1, w_idx:w_idx + 1]
                gmm2_jg = self._h_s2_same_resolution(torch.cat((p2, ctx_p2, ctx_p1), dim=1))  # 通道数 4M

                # print(non_zero_idx_1)
                # print(non_zero_idx_2)
                #
                # raise ValueError("stop c")

                # for i in range(len(non_zero_idx_1)):
                for ch_idx in range(y1_hat.shape[1]):
                    if ch_idx in non_zero_idx_1:
                        samples1 = torch.Tensor(samples1_np).to(device)
                        samples1 = samples1.reshape([samples1.shape[0], 1, 1])
                        samples1 = samples1.expand(samples1.shape[0], y1_hat.shape[2],
                                                   y1_hat.shape[3])  # [arange(xx),width,height]
                        # 非零通道：一个通道算一次-> 并行优化
                        # ch_idx = non_zero_idx_1[i]  # 非零通道

                        ####begin-pmf-calulation##############
                        ######################################
                        # gmm1_jg = [scales, means,weights] 通道数均为M*K
                        ch_idx_list = [ch_idx + temp_i * self.M for temp_i in range(self.K)]
                        sigma1 = gmm1_jg[0][0, ch_idx_list, :, :]
                        means1 = gmm1_jg[1][0, ch_idx_list, :, :] + minmax1  # 注意这里~~~ 相当于平移
                        weights1 = gmm1_jg[2][0, ch_idx_list, 0, 0]  # (M*k):((k+1)*M) #1x1
                        # torch.Size([5])
                        # print(sigma1.shape)
                        # print(means1.shape)
                        # # print(weights1.shape)
                        # raise ValueError("means shape",means1.shape)#[5,1,1]

                        # 计算pmf
                        pmf = None
                        for temp_K in range(self.K):
                            half = float(0.5)

                            values = samples1 - means1[temp_K]

                            temp_scales = self.gaussian1.lower_bound_scale(
                                sigma1[temp_K])  # scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

                            values = abs(values)
                            upper = self._standardized_cumulative((half - values) / temp_scales)
                            lower = self._standardized_cumulative((-half - values) / temp_scales)
                            if pmf == None:
                                pmf = (upper - lower) * weights1[temp_K]  # 指定对应的M个通道
                            else:
                                pmf += (upper - lower) * weights1[temp_K]  # 指定对应的M个通道

                        # print(pmf.shape)
                        # raise ValueError("stop")
                        # print("minmax1",minmax1)
                        pmf = pmf.cpu().numpy()
                        # print(pmf.shape)
                        # raise ValueError("stop")
                        ####end-pmf-calulation################
                        ######################################
                        pmf_temp = pmf[:, h_idx + padding, w_idx + padding]
                        # print(pmf_temp.shape)
                        # raise ValueError("stop")
                        # To avoid the zero-probability
                        pmf_clip = np.clip(pmf_temp, 1.0 / 65536, 1.0)
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536) #65536->65535
                        cdf = list(np.add.accumulate(pmf_clip))
                        cdf = [0] + [int(i) for i in cdf]

                        #拼接处
                        y1_hat[0, ch_idx, h_idx + padding, w_idx + padding] = decoder.decode(1, cdf)[0] - minmax1

                        # decodetempvalue0 = decoder.decode(1, cdf)[0]
                        # decodetempvalue = decodetempvalue0- minmax1
                        # y1_hat[0,ch_idx,h_idx+ padding,w_idx+ padding] =decodetempvalue
                        #
                        # if ch_idx==27: #21 27
                        #     # print(weights1)
                        #     print(cdf)
                        #     print(minmax1)
                        #     print(decodetempvalue0)
                        #     print(decodetempvalue)
                        #     raise ValueError(ch_idx)

                        # gpucpuwrite.write(str(h_idx)+","+str(w_idx)+"\t"+str(y1_hat[0,ch_idx,h_idx+ padding,w_idx+ padding])+"\t"+str(decodetempvalue)+"\n")
                    if ch_idx in non_zero_idx_2:
                        samples2 = torch.Tensor(samples2_np).to(device)
                        samples2 = samples2.reshape([samples2.shape[0], 1, 1])
                        samples2 = samples2.expand(samples2.shape[0], y2_hat.shape[2],
                                                   y2_hat.shape[3])  # [arange(xx),width,height]
                        # 非零通道：一个通道算一次-> 并行优化
                        # ch_idx = non_zero_idx_[i]  # 非零通道

                        ####begin-pmf-calulation##############
                        ######################################
                        # gmm2_jg = [scales, means,weights] 通道数均为M*K
                        ch_idx_list = [ch_idx + temp_i * self.M for temp_i in range(self.K)]
                        sigma2 = gmm2_jg[0][0, ch_idx_list, :, :]
                        means2 = gmm2_jg[2][0, ch_idx_list, :, :] + minmax2  # 注意这里~~~ 相当于平移
                        weights2 = gmm2_jg[2][0, ch_idx_list, 0, 0]  # (M*k):((k+1)*M) #1x1

                        # 计算pmf
                        pmf = None
                        for temp_K in range(self.K):
                            half = float(0.5)

                            values = samples2 - means2[temp_K]

                            temp_scales = self.gaussian2.lower_bound_scale(
                                sigma2[temp_K])  # scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

                            values = abs(values)
                            upper = self._standardized_cumulative((half - values) / temp_scales)
                            lower = self._standardized_cumulative((-half - values) / temp_scales)
                            if pmf == None:
                                pmf = (upper - lower) * weights2[temp_K]  # 指定对应的M个通道
                            else:
                                pmf += (upper - lower) * weights2[temp_K]  # 指定对应的M个通道

                        # print(pmf.shape)
                        # raise ValueError("stop")
                        # print("minmax2",minmax2)
                        pmf = pmf.cpu().numpy()
                        # print(pmf.shape)
                        # raise ValueError("stop")
                        ####end-pmf-calulation################
                        ######################################
                        pmf_temp = pmf[:, h_idx + padding, w_idx + padding]
                        # print(pmf_temp.shape)
                        # raise ValueError("stop")
                        # To avoid the zero-probability
                        pmf_clip = np.clip(pmf_temp, 1.0 / 65536, 1.0)
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536) #65536->65535
                        cdf = list(np.add.accumulate(pmf_clip))
                        cdf = [0] + [int(i) for i in cdf]
                        # 拼接处
                        y2_hat[0, ch_idx, h_idx + padding, w_idx + padding] =  decoder.decode(1, cdf)[0] - minmax2

                        # decodetempvalue0 = decoder.decode(1, cdf)[0]
                        # decodetempvalue = decodetempvalue0 - minmax2
                        # y2_hat[0, ch_idx, h_idx + padding, w_idx + padding] = decodetempvalue
                        # if ch_idx==26: # 22 26
                        #     print(samples2.dtype)
                        #     # print(weights1)
                        #     # print(pmf)
                        #     # print(pmf_temp)
                        #     np.save("data"+str(device),  z2_hat.cpu().numpy())
                        #     print(cdf)
                        #     print(minmax2)
                        #     print(decodetempvalue0)
                        #     print(decodetempvalue)
                        #     raise ValueError(ch_idx)

        decoder.close()

        # gpucpuwrite.close()

        end = time.time()
        delta_time = (end - start)

        y1_hat = y1_hat[:, :, padding:-padding, padding:-padding]
        y2_hat = y2_hat[:, :, padding:-padding, padding:-padding]
        # return y1_hat
        #########
        #gpu cpu?
        print("dec z1_hat",z1_hat)
        print("dec y1_hat cpu",y1_hat.to("cpu"))

        # y1_hat_cpu_np = y1_hat.cpu().numpy().astype('int')
        # for ch_idx in range(y1_hat.shape[1]):
        #     if np.sum(abs(y1_hat_cpu_np[:, ch_idx, :, :])) > 0:
        #         raise ValueError("exists not 0!!")

        #########
        # print(y1_hat.shape)
        # y1_hat = torch.Tensor(y1_hat).to(device)
        # y2_hat = torch.Tensor(y2_hat).to(device)
        #end y1

        #decoding for x1 and gmm2
        x1_hat,g1_4,g1_5,g1_6 = self.decoder1(y1_hat)

        ##decoder
        x1_hat_warp = kornia.warp_perspective(x1_hat, h_matrix, (x1_hat.size()[-2],x1_hat.size()[-1]))
        x2_hat = self.decoder2(y2_hat,x1_hat_warp)

        return {
            'enctime': delta_time,

            'x1_hat': x1_hat,
            'x2_hat': x2_hat,
            'y1_hat': y1_hat,
            'y2_hat': y2_hat,
            'z1_hat': z1_hat,
            'z2_hat': z2_hat,
        }

class Independent_EN(nn.Module):
    def __init__(self):
        super().__init__()

        # Enhancement
        self.EH1 = Enhancement()
        self.EH2 = Enhancement()

    #传入H矩阵进行交叉质量增强
    def forward(self, x1_hat,x2_hat,h_matrix):
        x1_hat_warp = kornia.warp_perspective(x1_hat, h_matrix, (x1_hat.size()[-2], x1_hat.size()[-1]))
        # end decoder
        # print(x1.size())
        h_inv = torch.inverse(h_matrix)
        x2_hat_warp = kornia.warp_perspective(x2_hat, h_inv, (x2_hat.size()[-2], x2_hat.size()[-1]))

        # 增强后
        x1_hat2 = self.EH1(x1_hat, x2_hat_warp)
        x2_hat2 = self.EH2(x2_hat, x1_hat_warp)

        return {
            'x1_hat': x1_hat2,
            'x2_hat': x2_hat2
        }


##合体版 HESIC
class GMM_together(nn.Module):
    def __init__(self,N=128,M=192,K=5,**kwargs):
        super().__init__()
        self.m1 = HSIC(N,M,K)
        self.m2 = Independent_EN()

    def forward(self,x1,x2,h):
        out1 = self.m1(x1,x2,h)
        out2 = self.m2(out1['x1_hat'],out1['x2_hat'],h)
        # out1['x1_hat'] = out2['x1_hat']
        # out1['x2_hat'] = out2['x2_hat']
        # return out1

        return {
            'x1_hat': out2['x1_hat'],
            'x2_hat': out2['x2_hat'],
            'likelihoods': out1['likelihoods'],
        }




