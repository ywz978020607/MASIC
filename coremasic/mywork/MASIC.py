#jg:joint_gmm #codec:熵编码编解码分离工程实现
#gf:gmm_full:pixel wise-gmm
#black3：熵模型部分进行warp(black2) + 修正掩膜使用错误
#black4：熵模型部分进行warp(black2) + 修正掩膜使用错误 -- warp使用的是解码后的x1_hat 而非左目原图

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
    def __init__(self, shape):
        super().__init__()
        self.RB1 = ResidualBlock(shape, shape)
        self.RB2 = ResidualBlock(shape, shape)
        self.RB3 = ResidualBlock(shape, shape)

    def forward(self, x):
        identity = x

        out = self.RB1(x)
        out = self.RB2(out)
        out = self.RB3(out)

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
            deconv(in_channels=4*M,out_channels=6*M,kernel_size=1,stride=1), #,stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.ReLU(inplace=True),

            deconv(in_channels=6*M, out_channels=4*M, kernel_size=1,stride=1),# stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.ReLU(inplace=True),

            conv(in_channels=4*M,out_channels=(M*K),kernel_size=1,stride=1), #padding=kernel_size//2
            nn.ReLU(inplace=True),
        )

        self.gmm_means = nn.Sequential(
            deconv(in_channels=4*M, out_channels=6*M, kernel_size=1,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            deconv(in_channels=6*M, out_channels=4*M, kernel_size=1,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            conv(in_channels=4*M, out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
        )


        self.gmm_weights = nn.Sequential(
            deconv(in_channels=4*M, out_channels=6*M, kernel_size=1,stride=1),
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
class gmm_hyper_y2_same_resolution(nn.Module):
    def __init__(self,N,M,K): #K表示GMM对应正态分布个数
        super().__init__()
        self.N = N
        self.M = M
        self.K = K

        # self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=4) #固定为4倍 因为z和y分辨率本身就差4倍

        # 输入与y1分辨率相同，但通道数不同 2倍4M+2M -> 6*M
        #输入如果有y1_hat 则5M 如果是z1_hat升采样的2M，则6M
        self.gmm_sigma = nn.Sequential(
            conv(in_channels=5*M,out_channels=6*M,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),

            conv(in_channels=6*M,out_channels=4*M,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),

            conv(in_channels=4*M,out_channels=(M*K),kernel_size=1,stride=1), #padding=kernel_size//2
            nn.ReLU(inplace=True),
        )

        self.gmm_means = nn.Sequential(
            conv(in_channels=5*M,out_channels=6*M,kernel_size=1,stride=1),
            nn.LeakyReLU(inplace=True),

            conv(in_channels=6*M,out_channels=4*M,kernel_size=1,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            conv(in_channels=4*M, out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
        )


        self.gmm_weights = nn.Sequential(
            conv(in_channels=5*M,out_channels=6*M,kernel_size=1,stride=1),
            nn.LeakyReLU(inplace=True),

            conv(in_channels=6*M,out_channels= (M*K),kernel_size=1,stride=1),
            # # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
            # spatial_pool2d(), #joint+gmm换1x1卷积不再需要
            nn.LeakyReLU(inplace=True),

            conv(in_channels=(M*K), out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
            #出去后要接一个softmax层，表示概率！！
        )

    def forward(self,allconcat):
        # self.up_z2 = self.upsample_layer(z2)
        # self.cat_in = torch.cat((self.up_z2,y1),dim=-3)

        self.sigma = self.gmm_sigma(allconcat)
        self.means = self.gmm_means(allconcat)
        # #softmax!!
        # # self.weights = nn.functional.softmax(self.gmm_weights(self.cat_in),dim=-3)
        # # 修正
        # temp = torch.reshape(self.gmm_weights(allconcat), (-1, self.K, self.M,  1, 1))  # 方便后续reshape合并时同一个M的数据相邻成组
        # temp = nn.functional.softmax(temp, dim=-4)  # 每个Mixture进行一次归一
        # self.weights = torch.reshape(temp, (-1, self.M * self.K, 1, 1))
        ###
        temp = torch.reshape(self.gmm_weights(allconcat),
                             (-1, self.K, self.M, allconcat.shape[-2], allconcat.shape[-1]))  # 方便后续reshape合并时同一个M的数据相邻成组
        # temp = torch.reshape(self.gmm_weights(allconcat),(-1,self.K,self.M,1,1)) #方便后续reshape合并时同一个M的数据相邻成组

        temp = nn.functional.softmax(temp, dim=-4)  # 每个Mixture进行一次归一
        self.weights = torch.reshape(temp, (-1, self.M * self.K, allconcat.shape[-2], allconcat.shape[-1]))
        # self.weights = torch.reshape(temp,(-1,self.M*self.K,1,1))
        ###

        return self.sigma,self.means,self.weights


#根据mask生成一系列归一化的权重值
class mask2weights(nn.Module):
    def __init__(self,N,M,Kw):
        super().__init__()
        self.maskconv = nn.Sequential(
            # 先abs 再输入进来
            conv(in_channels=1, out_channels=3, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            conv(in_channels=3, out_channels=6, kernel_size=3),  # stride = 2
            nn.ReLU(inplace=True),

            conv(in_channels=6, out_channels=6, kernel_size=3),  # stride = 2
            nn.ReLU(inplace=True),

            conv(in_channels=6, out_channels=3, kernel_size=3),  # stride = 2
            # 出去后接bottleneck
        )
        self.N = N
        self.M = M
        self.Kw=Kw

    def forward(self,allconcat):
        self.maskconvout = self.maskconv(allconcat)

        # print(self.maskconvout.shape)
        temp = torch.reshape(self.maskconvout,
                             (-1,self.Kw,1, self.maskconvout.shape[-2], self.maskconvout.shape[-1]))  # 方便后续reshape合并时同一个M的数据相邻成组
        # temp = torch.reshape(self.gmm_weights(allconcat),(-1,self.K,self.M,1,1)) #方便后续reshape合并时同一个M的数据相邻成组

        temp = nn.functional.softmax(temp, dim=-4)  # 每个Mixture进行一次归一
        self.weights = torch.reshape(temp, (-1,1 * self.Kw, self.maskconvout.shape[-2], self.maskconvout.shape[-1]))
        # self.weights = torch.reshape(temp,(-1,self.M*self.K,1,1))
        ###

        return self.weights


###################################
class Encoder1(nn.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.g_a_conv1 = conv(3, N)
        self.g_a_gdn1 = GDN(N)
        self.g_a_conv2 = conv(N, N)
        self.g_a_gdn2 = GDN(N)
        self.g_a_conv3 = conv(N, N)
        self.g_a_gdn3 = GDN(N)
        self.g_a_conv4 = conv(N, M)

    def forward(self, x):
        # self.y = self.g_a(x)
        self.g_a_c1 = self.g_a_conv1(x) #Tensor
        self.g_a_g1 = self.g_a_gdn1(self.g_a_c1)
        self.g_a_c2 = self.g_a_conv2(self.g_a_g1)  # Tensor
        self.g_a_g2 = self.g_a_gdn2(self.g_a_c2)
        self.g_a_c3 = self.g_a_conv3(self.g_a_g2)  # Tensor
        self.g_a_g3 = self.g_a_gdn3(self.g_a_c3)
        self.g_a_c4 = self.g_a_conv4(self.g_a_g3)  # Tensor
        self.y = self.g_a_c4
        return self.y,self.g_a_g1,self.g_a_g2,self.g_a_g3

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
##
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
        return self.y#,self.g_a_g1,self.g_a_g2,self.g_a_g3

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

        #
        self.after_gdn = GDN(3, inverse=True)
        self.after_conv = deconv(6, 3, stride=1)  # 不缩放！
        # self.after_gdn = GDN(6, inverse=True)
        # self.after_conv = deconv(6,3,stride=1) #不缩放！


    def forward(self, y_hat,x1_hat_warp):
        # self.x_hat = self.g_s(self.y_hat)
        self.g_s_c1 = self.g_s_conv1(y_hat)  # Tensor
        self.g_s_g1 = self.g_s_gdn1(self.g_s_c1)
        self.g_s_c2 = self.g_s_conv2(self.g_s_g1)  # Tensor
        self.g_s_g2 = self.g_s_gdn2(self.g_s_c2)
        self.g_s_c3 = self.g_s_conv3(self.g_s_g2)  # Tensor
        self.g_s_g3 = self.g_s_gdn3(self.g_s_c3)
        self.g_s_c4 = self.g_s_conv4(self.g_s_g3)  # Tensor
        #
        self.after1 = self.after_gdn(self.g_s_c4)
        self.after2 = self.after_conv(torch.cat((self.after1,x1_hat_warp),dim=-3))
        # self.after1 = self.after_gdn(torch.cat((self.g_s_c4, x1_hat_warp), dim=-3))
        # self.after2 = self.after_conv(self.after1)

        self.x_hat = self.after2
        # self.x_hat = self.g_s_c4
        return self.x_hat#,self.g_s_g1,self.g_s_g2,self.g_s_g3


###########################################################################
# torch.where(a!=0,1,0)
def mask(im1, H_inv):  # 输入左目的mask和H矩阵左->右
    # # mask1 = torch.ones_like(im1)
    # mask1 = kornia.warp_perspective(im1, H_inv, (im1.shape[-2], im1.shape[-1]))

    # # batch_size > 1时的情况要兼容
    # print(H_inv.shape)
    # raise ValueError("stop h shape")

    # im1 = torch.ones_like(im1)
    im1 = torch.ones([im1.shape[0], 1, im1.shape[-2], im1.shape[-1]], dtype=im1.dtype, layout=im1.layout,
                     device=im1.device)
    mask1 = kornia.warp_perspective(im1, H_inv, (im1.shape[-2], im1.shape[-1]))  # 右目的mask

    # mask1[torch.nonzero(mask1)] = 1.
    torch.where(mask1!=0,1,0)
    # torch.where(mask1 != 0, torch.Tensor([1]).to(im1.device), torch.Tensor([0]).to(im1.device))

    mask2 = kornia.warp_perspective(mask1, torch.inverse(H_inv), (im1.shape[-2], im1.shape[-1]))  # 换算回左目的mask
    # mask2[torch.nonzero(mask2)] = 1.
    torch.where(mask2 != 0, 1, 0)
    # torch.where(mask2 != 0, torch.Tensor([1]).to(im1.device), torch.Tensor([0]).to(im1.device))

    return mask1, mask2 #左->右->左中  右目掩膜， 左目掩膜


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
        self.encoder1 = Encoder1(N,M)
        self.encoder2 = Encoder2(N,M)
        self.decoder1 = Decoder1(N,M)
        self.decoder2 = Decoder2(N,M)
        # pic2 需要的组件


        #hyper
        self._h_a1 = encode_hyper(N=N,M=M)
        self._h_a2 = encode_hyper(N=N,M=M)
        # self._h_s1 = gmm_hyper_y1(N=N,M=M,K=K)
        # self._h_s2 = gmm_hyper_y2(N=N,M=M,K=K)

        #先将z上采样为y的大小，再和自回归的内容一起不变分辨率的卷出GMM
        self.h_s1_up = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        self.h_s2_up = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
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
        self._h_s2_same_resolution = gmm_hyper_y2_same_resolution(N=N, M=M, K=K)
        self.mask2weights_unit = mask2weights(N=N,M=M,Kw=3)
        # self.need_int = need_int

        #quantize
    def _quantize(self, inputs, mode, means=None):
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
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def forward(self,x1,x2,h_matrix):
        #定义结构
        y1,g1_1,g1_2,g1_3 = self.encoder1(x1)
        z1 = self._h_a1(y1)
        #print(z1.device)
        z1_hat,z1_likelihoods = self.entropy_bottleneck1(z1)
        ###
        # gmm1 = self._h_s1(z1_hat) #三要素
        ###
        #change for jq
        params1 = self.h_s1_up(z1_hat) # torch.Size([1, 384, 32, 32])
        y1_hat = self.gaussian1._quantize(  # pylint: disable=protected-access
            y1, 'noise' if self.training else 'dequantize')
        ctx_params1 = self.context_prediction1(y1_hat)  # 用两次！！ 2M # torch.Size([1, 384, 32, 32])
        # gaussian_params1 = self.entropy_parameters1(
        #     torch.cat((params1, ctx_params1), dim=1))
        # print(z1_hat.shape)
        # print(y1_hat.shape)
        # print(params1.shape)
        # print(ctx_params1.shape)
        # raise  ValueError("stop")
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

        x1_hat,g1_4,g1_5,g1_6 = self.decoder1(y1_hat)

        #############################################
        #encoder
        x1_warp = kornia.warp_perspective(x1, h_matrix, (x1.size()[-2],x1.size()[-1]))
        y2 = self.encoder2(x1_warp,x2)
        ##end encoder

        # hyper for pic2
        z2 = self._h_a2(y2)
        z2_hat, z2_likelihoods = self.entropy_bottleneck2(z2)
        ###
        # gmm2 = self._h_s2(z2_hat, y1_hat)  # 三要素
        ###
        #change
        # change
        params2 = self.h_s2_up(z2_hat)
        y2_hat = self.gaussian2._quantize(  # pylint: disable=protected-access
            y2, 'noise' if self.training else 'dequantize')
        ctx_params2 = self.context_prediction2(y2_hat)
        # gaussian_params2 = self.entropy_parameters2(
        #     torch.cat((params2, ctx_params2, ctx_params1), dim=1))

        ###########################
        ## add black1: 通过mask1-2-1后进行卷积降维出weights再乘上去，3组归一*xxx
        # x1_mask = None
        x1_mask_R,x1_mask_L = mask(x1,h_matrix)
        #根据x1_mask生成每个像素一组权重，去乘params2,ctx_param2,ctx_params1
        mask_weights = self.mask2weights_unit(x1_mask_R)
        # print(mask_weights.shape)
        # print(mask_weights[0,5,6]+mask_weights[1,5,6]+mask_weights[2,5,6])
        # print(mask_weights[2:3,:,:].shape)
        # print(params2.shape)
        # print(ctx_params2.shape)
        # print(ctx_params1.shape)
        # # torch.Size([1, 384, 32, 32]) #512->input
        # # torch.Size([1, 384, 32, 32])
        # # torch.Size([1, 384, 32, 32])
        # print(mask_weights.requires_grad)
        # raise ValueError("stopshape")

        ###########################
        #black2：考虑warp对熵编码影响 + 考虑左目完全恢复的全信息给入
        #black2–仍使用左目codec通道编x1_warp
        x1_warp_aftercodec = kornia.warp_perspective(x1_hat, h_matrix, (x1.size()[-2],x1.size()[-1]))
        y1_warpf2, _, _, _ = self.encoder1(x1_warp_aftercodec)
        y1_hat_warpf2 = self.gaussian1._quantize(  # pylint: disable=protected-access
            y1_warpf2, 'noise' if self.training else 'dequantize')

        ###########################
        gmm2_jq = self._h_s2_same_resolution(torch.cat((params2*mask_weights[:,0:1,:,:], ctx_params2*mask_weights[:,1:2,:,:],y1_hat_warpf2*mask_weights[:,2:3,:,:]), dim=1)) #通道数 4M
        ###
        y2_hat, y2_likelihoods = self.gaussian2(y2, gmm2_jq[0], gmm2_jq[1], gmm2_jq[2])  # 这里也是临时，待改gmm
        # end hyper for pic2

        ##decoder
        x1_hat_warp = kornia.warp_perspective(x1_hat, h_matrix, (x1_hat.size()[-2],x1_hat.size()[-1]))
        x2_hat = self.decoder2(y2_hat,x1_hat_warp)
        #end decoder
        # print(x1.size())

        return {
            'x1_hat': x1_hat,
            'x2_hat': x2_hat,
            'y1_hat': y1_hat,
            'z1_hat': z1_hat,
            'x1_mask_R':x1_mask_R,
            'x1_mask_L': x1_mask_L,
            'likelihoods':{
                'y1': y1_likelihoods,
                'y2': y2_likelihoods,
                'z1': z1_likelihoods,
                'z2': z2_likelihoods,
            }
        }


    ###codec
    def compress(self,x1,x2,h_matrix,output_name,output_path="",device="cpu"):
        # print("train status")
        # print(self.training)
        # 定义结构
        y1, g1_1, g1_2, g1_3 = self.encoder1(x1)
        z1 = self._h_a1(y1)
        # print(z1.device)
        # z1_hat, z1_likelihoods = self.entropy_bottleneck1(z1)
        z1_strings = self.entropy_bottleneck1.compress(z1)
        z1_hat = self.entropy_bottleneck1.decompress(z1_strings, z1.size()[-2:])  # z解码后结果（压缩时仍需要）

        # change for jq
        params1 = self.h_s1_up(z1_hat)  # torch.Size([1, 384, 32, 32])
        y1_hat = self.gaussian1._quantize(  # pylint: disable=protected-access
            y1, 'noise' if self.training else 'dequantize')
        ctx_params1 = self.context_prediction1(y1_hat)  # 用两次！！ 2M # torch.Size([1, 384, 32, 32])
        gmm1_jg = self._h_s1_same_resolution(torch.cat((params1, ctx_params1), dim=1))  # 通道数 4M
        ###
        y1_hat, y1_likelihoods = self.gaussian1(y1, gmm1_jg[0], gmm1_jg[1], gmm1_jg[2])  # sigma 每个都是M通道 与y1同

        x1_hat, g1_4, g1_5, g1_6 = self.decoder1(y1_hat)

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
        y2_hat = self.gaussian2._quantize(  # pylint: disable=protected-access
            y2, 'noise' if self.training else 'dequantize')
        ctx_params2 = self.context_prediction2(y2_hat)
        x1_mask_R, x1_mask_L = mask(x1, h_matrix)
        mask_weights = self.mask2weights_unit(x1_mask_R)
        ###########################
        # black2：考虑warp对熵编码影响 + 考虑左目完全恢复的全信息给入
        # black2–仍使用左目codec通道编x1_warp
        x1_warp_aftercodec = kornia.warp_perspective(x1_hat, h_matrix, (x1.size()[-2], x1.size()[-1]))
        y1_warpf2, _, _, _ = self.encoder1(x1_warp_aftercodec)
        y1_hat_warpf2 = self.gaussian1._quantize(  # pylint: disable=protected-access
            y1_warpf2, 'noise' if self.training else 'dequantize')

        ###########################
        gmm2_jq = self._h_s2_same_resolution(torch.cat((params2 * mask_weights[:, 0:1, :, :],
                                                        ctx_params2 * mask_weights[:, 1:2, :, :],
                                                        y1_hat_warpf2 * mask_weights[:, 2:3, :, :]), dim=1))  # 通道数 4M
        ###

        ##encoding
        y1_hat_cpu_np = y1_hat.cpu().numpy().astype('int')
        y2_hat_cpu_np = y2_hat.cpu().numpy().astype('int')
        # print(y1_hat)
        # print(y1_hat_cpu_np)
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
        # ========encoding for range coder of y1 and y2
        # 逐像素计算概率分布、进行熵编码->后续考虑加入joint 进一步提高
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
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        y1_hat = F.pad(y1_hat, (padding, padding, padding, padding))
        y2_hat = F.pad(y2_hat, (padding, padding, padding, padding))
        # y1
        # 如果是joint一类，在此计算当前像素点的概率
        for h_idx in range(y1_hat_cpu_np.shape[2]):
            for w_idx in range(y1_hat_cpu_np.shape[3]):
                # 计算mask cnn  #逐像素计算
                y1_crop = y1_hat[0:0 + 1, :, h_idx:h_idx + kernel_size, w_idx:w_idx + kernel_size]
                ctx_params1 = self.context_prediction1(y1_crop)
                ctx_p1 = ctx_params1[0:0 + 1, :, padding:padding + 1, padding:padding + 1]
                p1 = params1[0:0 + 1, :, h_idx:h_idx + 1, w_idx:w_idx + 1]
                # gaussian_params1 = self.entropy_parameters1(torch.cat((p1, ctx_p1), dim=1))
                # scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
                ####
                # ctx_params1 = self.context_prediction1(y1_hat)
                gmm1_jg = self._h_s1_same_resolution(torch.cat((p1, ctx_p1), dim=1))  # 通道数 4M

                # print(ctx_params1)
                # print(gmm1_jg[0][0,0:10])
                for i in range(len(non_zero_idx_1)):
                    samples1 = torch.Tensor(samples1_np).to('cuda:0')
                    samples1 = samples1.reshape([samples1.shape[0], 1, 1])
                    samples1 = samples1.expand(samples1.shape[0], y1_hat_cpu_np.shape[2],
                                               y1_hat_cpu_np.shape[3])  # [arange(xx),width,height]
                    # 非零通道：一个通道算一次-> 并行优化
                    ch_idx = non_zero_idx_1[i]  # 非零通道

                    # gmm1_jg = [scales, means,weights] 通道数均为M*K
                    ch_idx_list = [ch_idx + temp_i * self.M for temp_i in range(self.K)]
                    # joint 改进--当前像素点
                    sigma1 = gmm1_jg[0][0, ch_idx_list]
                    means1 = gmm1_jg[1][0, ch_idx_list] + minmax1  # 注意这里~~~ 相当于平移
                    weights1 = gmm1_jg[2][0, ch_idx_list]  # (M*k):((k+1)*M) #1x1 ->改为:,: 兼容非通道共用权重
                    # torch.Size([5])
                    # print(sigma1.shape)
                    # print(means1.shape)
                    # print(weights1.shape)

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


                    # raise ValueError("stop")
                    # print("minmax1",minmax1)

                    pmf = pmf.cpu().numpy()

                    # print(pmf)
                    # print(pmf.shape)
                    # raise ValueError("stop")


                    pmf_temp = pmf[:, h_idx, w_idx]
                    # print(pmf_temp.shape)
                    # raise ValueError("stop")

                    # To avoid the zero-probability
                    pmf_clip = np.clip(pmf_temp, 1.0 / 65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(i) for i in cdf]

                    symbol = np.int(y1_hat[0, ch_idx, h_idx+padding, w_idx+padding] + minmax1)
                    # print("len-cdf",len(cdf))
                    # raise ValueError("stop")
                    encoder.encode([symbol], cdf)

                    # print(cdf)
        # end y1
        # y2
        x1_mask_R, x1_mask_L = mask(x1, h_matrix)
        # 根据x1_mask生成每个像素一组权重，去乘params2,ctx_param2,ctx_params1
        mask_weights = self.mask2weights_unit(x1_mask_R)
        x1_warp_aftercodec = kornia.warp_perspective(x1_hat, h_matrix, (x1.size()[-2], x1.size()[-1]))
        y1_warpf2, _, _, _ = self.encoder1(x1_warp_aftercodec)
        y1_hat_warpf2 = self.gaussian1._quantize(  # pylint: disable=protected-access
            y1_warpf2, 'noise' if self.training else 'dequantize')

        for h_idx in range(y2_hat_cpu_np.shape[2]):
            for w_idx in range(y2_hat_cpu_np.shape[3]):
                y2_crop = y2_hat[0:0 + 1, :, h_idx:h_idx + kernel_size, w_idx:w_idx + kernel_size]
                ctx_params2 = self.context_prediction2(y2_crop)
                ctx_p2 = ctx_params2[0:0 + 1, :, padding:padding + 1, padding:padding + 1]

                p2 = params2[0:0 + 1, :, h_idx:h_idx + 1, w_idx:w_idx + 1]

                gmm2_jq = self._h_s2_same_resolution(torch.cat((p2 * mask_weights[:, 0:1, :, :],
                                                                ctx_p2 * mask_weights[:, 1:2, :, :],
                                                                y1_hat_warpf2 * mask_weights[:, 2:3, :, :]),
                                                               dim=1))  # 通道数 4M

                for i in range(len(non_zero_idx_2)):
                    samples2 = torch.Tensor(samples2_np).to('cuda:0')
                    samples2 = samples2.reshape([samples2.shape[0], 1, 1])
                    samples2 = samples2.expand(samples2.shape[0], y2_hat_cpu_np.shape[2],
                                               y2_hat_cpu_np.shape[3])  # [arange(xx),width,height]
                    # 非零通道：一个通道算一次-> 并行优化
                    ch_idx = non_zero_idx_2[i]  # 非零通道

                    # gmm2_jq = [scales, means,weights] 通道数均为M*K
                    ch_idx_list = [ch_idx + temp_i * self.M for temp_i in range(self.K)]
                    sigma2 = gmm2_jq[0][0, ch_idx_list]
                    means2 = gmm2_jq[1][0, ch_idx_list] + minmax2  # 注意这里~~~ 相当于平移
                    weights2 = gmm2_jq[2][0, ch_idx_list]  # (M*k):((k+1)*M) #1x1

                    # 计算pmf -- 还需优化--只计算一个点的概率即可
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
                    pmf = pmf.cpu().numpy()

                    ##原本是缩进 但把逐像素放到开头，这里取消缩进

                    pmf_temp = pmf[:, h_idx, w_idx]
                    # print(pmf_temp.shape)
                    # raise ValueError("stop")

                    # To avoid the zero-probability
                    pmf_clip = np.clip(pmf_temp, 1.0 / 65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(i) for i in cdf]

                    symbol = np.int(y2_hat[0, ch_idx, h_idx+padding, w_idx+padding] + minmax2)
                    # print("len-cdf",len(cdf))
                    # raise ValueError("stop")
                    encoder.encode([symbol], cdf)
        # end y2
        ##end encoding
        encoder.close()

        #### for debug - 21.10.29
        # y1_hat_for_print = y1_hat[:, :, padding:-padding, padding:-padding]
        # print(z1_hat)
        # print("z1_hat ----- y1_hat")
        # print(y1_hat_for_print)
        # raise ValueError("stop encoder y1_hat")

        end = time.time()

        #################################
        #################################
        #################################
        num_pixels = x1.shape[2] * x1.shape[3] * 2  # 双目两张图
        size_real = os.path.getsize(output1) + os.path.getsize(output2)

        bpp_real = (os.path.getsize(output1) + os.path.getsize(output2)) * 8 / num_pixels  # 主要看这个
        bpp_side = (os.path.getsize(output1)) * 8 / num_pixels

        # print("Time : {:0.3f}".format(end-start))
        print("bpp_real:", bpp_real)
        delta_time = end-start
        print("enc-time",delta_time)
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
        output1 = os.path.join(output_path, str(output_name) + '.npz')
        output2 = os.path.join(output_path, str(output_name) + '.bin')

        # =================decoding for z1 and z2
        fileobj = open(output1, mode='rb')
        x_shape = np.frombuffer(fileobj.read(4), dtype=np.uint16)  # 图片尺寸 [width,height] array([512, 512], dtype=uint16)
        # z1
        length1, minmax1 = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        num1 = np.frombuffer(fileobj.read(self.M // 8), dtype=np.uint8)
        string1 = fileobj.read(length1)
        # z2
        length2, minmax2 = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        num2 = np.frombuffer(fileobj.read(self.M // 8), dtype=np.uint8)  # 192//8=24
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
        y_shape = x_shape // 16
        z_shape = y_shape // 4

        z1_hat = self.entropy_bottleneck1.decompress([string1], z_shape)  # z解码后结果
        z2_hat = self.entropy_bottleneck2.decompress([string2], z_shape)  # z解码后结果

        #接z1_hat次级解码
        # change for jq
        params1 = self.h_s1_up(z1_hat)  # torch.Size([1, 384, 32, 32])
        params2 = self.h_s2_up(z2_hat)

        # ===================decoding for y1 and y2
        # 逐像素计算概率分布、进行熵编码->后续考虑加入joint 进一步提高
        output2 = os.path.join(output_path, str(output_name) + '.bin')

        # samples1 = np.arange(0, minmax1*2+1)
        # samples1 = torch.Tensor(samples1).to('cuda:0')
        samples1_np = np.arange(0, minmax1 * 2 + 1)
        samples2_np = np.arange(0, minmax2 * 2 + 1)

        start = time.time()
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        #
        # y1_hat = np.zeros([1] + [y_shape[1]+kernel_size-1] + [y_shape[2]+kernel_size-1] + [num_filters])
        y1_hat = np.zeros([1] + [self.M] + [y_shape[0] + 2 * padding, y_shape[1] + 2 * padding])  # [1,192,32,32]
        y2_hat = np.zeros([1] + [self.M] + [y_shape[0] + 2 * padding, y_shape[1] + 2 * padding])  # [1,192,32,32]
        y1_hat = torch.Tensor(y1_hat).to(device)
        y2_hat = torch.Tensor(y2_hat).to(device)
        
        decoder = RangeDecoder(output2)
        # joint要改一些东西：保证循环

        # y1

        # 逐像素解码
        for h_idx in range(y_shape[0]):
            for w_idx in range(y_shape[1]):
                #更新下一像素的先验
                # ctx_params1 = self.context_prediction1(y1_hat)
                # gmm1_jg = self._h_s1_same_resolution(torch.cat((params1, ctx_params1), dim=1))
                y1_crop = y1_hat[0:0 + 1, :, h_idx:h_idx + kernel_size, w_idx:w_idx + kernel_size]
                ctx_params1 = self.context_prediction1(y1_crop)
                ctx_p1 = ctx_params1[0:0 + 1, :, padding:padding + 1, padding:padding + 1]
                p1 = params1[0:0 + 1, :, h_idx:h_idx + 1, w_idx:w_idx + 1]
                # gaussian_params1 = self.entropy_parameters1(torch.cat((p1, ctx_p1), dim=1))
                # scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
                ####
                # ctx_params1 = self.context_prediction1(y1_hat)
                gmm1_jg = self._h_s1_same_resolution(torch.cat((p1, ctx_p1), dim=1))  # 通道数 4M

                # print(ctx_params1)
                # print(gmm1_jg[0][0, 0:10])
                for i in range(len(non_zero_idx_1)):
                    samples1 = torch.Tensor(samples1_np).to(device)
                    samples1 = samples1.reshape([samples1.shape[0], 1, 1])
                    samples1 = samples1.expand(samples1.shape[0], y_shape[0], y_shape[1])  # [arange(xx),width,height]

                    # 非零通道：一个通道算一次-> 并行优化
                    ch_idx = non_zero_idx_1[i]  # 非零通道

                    # gmm1 = [scales, means,weights] 通道数均为M*K
                    ch_idx_list = [ch_idx + temp_i * self.M for temp_i in range(self.K)]
                    sigma1 = gmm1_jg[0][0, ch_idx_list]
                    means1 = gmm1_jg[1][0, ch_idx_list] + minmax1  # 注意这里~~~ 相当于平移
                    weights1 = gmm1_jg[2][0, ch_idx_list]  # (M*k):((k+1)*M) #1x1
                    # torch.Size([5])
                    # print(sigma1.shape)
                    # print(means1.shape)
                    # print(weights1.shape)

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


                    # raise ValueError("stop")
                    # print("minmax1",minmax1)

                    pmf = pmf.cpu().numpy()

                    # print(pmf)
                    # raise ValueError("stop1")


                    pmf_temp = pmf[:, h_idx, w_idx]

                    # To avoid the zero-probability
                    pmf_clip = np.clip(pmf_temp, 1.0 / 65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(i) for i in cdf]

                    y1_hat[0, ch_idx, h_idx+padding, w_idx+padding] = decoder.decode(1, cdf)[0] - minmax1

                    # print(y1_hat.shape)
        y1_hat = y1_hat[:, :, padding:-padding, padding:-padding]

        # print(z1_hat)
        # print("z1_hat--------y1_hat")
        # print(y1_hat)
        # raise ValueError("stop decoder y1_hat")
        # y1_hat = torch.Tensor(y1_hat).to(device)
        # end y1

        # decoding for x1 and gmm2
        x1_hat, g1_4, g1_5, g1_6 = self.decoder1(y1_hat)
        x1_mask_R, x1_mask_L = mask(x1_hat, h_matrix)
        mask_weights = self.mask2weights_unit(x1_mask_R)
        x1_warp_aftercodec = kornia.warp_perspective(x1_hat, h_matrix, (x1.size()[-2], x1.size()[-1]))
        y1_warpf2, _, _, _ = self.encoder1(x1_warp_aftercodec)
        y1_hat_warpf2 = self.gaussian1._quantize(  # pylint: disable=protected-access
            y1_warpf2, 'noise' if self.training else 'dequantize')

        # y2
        # 逐像素解码
        for h_idx in range(y_shape[0]):
            for w_idx in range(y_shape[1]):
                y2_crop = y2_hat[0:0 + 1, :, h_idx:h_idx + kernel_size, w_idx:w_idx + kernel_size]
                ctx_params2 = self.context_prediction2(y2_crop)
                ctx_p2 = ctx_params2[0:0 + 1, :, padding:padding + 1, padding:padding + 1]

                p2 = params2[0:0 + 1, :, h_idx:h_idx + 1, w_idx:w_idx + 1]

                gmm2_jq = self._h_s2_same_resolution(torch.cat((p2 * mask_weights[:, 0:1, :, :],
                                                                ctx_p2 * mask_weights[:, 1:2, :, :],
                                                                y1_hat_warpf2 * mask_weights[:, 2:3, :, :]),
                                                               dim=1))  # 通道数 4M

                for i in range(len(non_zero_idx_2)):
                    samples2 = torch.Tensor(samples2_np).to('cuda:0')
                    samples2 = samples2.reshape([samples2.shape[0], 1, 1])
                    samples2 = samples2.expand(samples2.shape[0], y_shape[0], y_shape[1])  # [arange(xx),width,height]

                    # 非零通道：一个通道算一次-> 并行优化
                    ch_idx = non_zero_idx_2[i]  # 非零通道

                    # gmm2 = [scales, means,weights] 通道数均为M*K
                    ch_idx_list = [ch_idx + temp_i * self.M for temp_i in range(self.K)]
                    sigma2 = gmm2_jq[0][0, ch_idx_list]
                    means2 = gmm2_jq[1][0, ch_idx_list] + minmax2  # 注意这里~~~ 相当于平移
                    weights2 = gmm2_jq[2][0, ch_idx_list]  # (M*k):((k+1)*M) #1x1
                    # torch.Size([5])
                    # print(sigma2.shape)
                    # print(means2.shape)
                    # print(weights2.shape)

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

                    pmf = pmf.cpu().numpy()
                    # print(pmf.shape)
                    # raise ValueError("stop2")


                    pmf_temp = pmf[:, h_idx, w_idx]

                    # To avoid the zero-probability
                    pmf_clip = np.clip(pmf_temp, 1.0 / 65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(i) for i in cdf]

                    y2_hat[0, ch_idx, h_idx+padding, w_idx+padding] = decoder.decode(1, cdf)[0] - minmax2
        y2_hat = y2_hat[:, :, padding:-padding, padding:-padding]
        # print(y2_hat.shape)
        # y2_hat = torch.Tensor(y2_hat).to(device)
        # end y2

        # decode for x2
        x1_hat_warp = kornia.warp_perspective(x1_hat, h_matrix, (x1_hat.size()[-2], x1_hat.size()[-1]))
        x2_hat = self.decoder2(y2_hat, x1_hat_warp)
        # end x2
        ###
        # raise ValueError('stop')

        decoder.close()
        end = time.time()
        delta_time = end - start
        print("dec-time", delta_time)

        return {
            'x1_hat': x1_hat,
            'x2_hat': x2_hat,
            'y1_hat': y1_hat,
            'y2_hat': y2_hat,
            'z1_hat': z1_hat,
            'z2_hat': z2_hat,
        }


class mask2weights_EN(nn.Module):
    def __init__(self, Kw=2):
        super().__init__()
        self.maskconv = nn.Sequential(
            # 先abs 再输入进来
            conv(in_channels=1, out_channels=Kw, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            conv(in_channels=Kw, out_channels=Kw*2, kernel_size=3, stride=1),  # stride = 2
            nn.ReLU(inplace=True),

            conv(in_channels=Kw*2, out_channels=Kw*2, kernel_size=3, stride=1),  # stride = 2
            nn.ReLU(inplace=True),

            conv(in_channels=Kw*2, out_channels=Kw, kernel_size=3, stride=1),  # stride = 2
            # 出去后接bottleneck
        )

    def forward(self, allconcat):
        self.maskconvout = self.maskconv(allconcat)
        temp = self.maskconvout
        self.weights = nn.functional.softmax(temp, dim=-3)  # 每个Mixture进行一次归一

        return self.weights

class Independent_EN(nn.Module):
    def __init__(self):
        super().__init__()

        self.EBl1 = Enhancement_Block(shape=32)
        self.EBl2 = Enhancement_Block(shape=64)
        self.EBl3 = Enhancement_Block(shape=96)
        self.EBr1 = Enhancement_Block(shape=32)
        self.EBr2 = Enhancement_Block(shape=64)
        self.EBr3 = Enhancement_Block(shape=96)
        self.conv0 = conv3x3(3, 32)
        self.conv1 = conv3x3(6, 32)
        self.conv2 = conv3x3(96, 3)
        self.mask2weights_unit = mask2weights_EN()

        # Enhancement
        # self.EH1 = Enhancement()
        # self.EH2 = Enhancement()

    # 传入H矩阵进行交叉质量增强
    def forward(self, x1_hat, x2_hat, h_matrix):
        h_inv = torch.inverse(h_matrix)
        x1_mask_R, x1_mask_L = mask(x1_hat, h_matrix)
        mask_weights_R = self.mask2weights_unit(x1_mask_R)
        mask_weights_L = self.mask2weights_unit(x1_mask_L)
        x1_hat_warp = kornia.warp_perspective(x1_hat, h_matrix, (x1_hat.size()[-2], x1_hat.size()[-1]))
        # end decoder

        x2_hat_warp = kornia.warp_perspective(x2_hat, h_inv, (x2_hat.size()[-2], x2_hat.size()[-1]))
        identity1 = x1_hat
        identity2 = x2_hat
        x1_hat_conv = self.conv0(x1_hat)
        x2_hat_conv = self.conv0(x2_hat)

        out1 = torch.cat((x2_hat_warp * mask_weights_L[:, 0:1, :, :], x1_hat * mask_weights_L[:, 1:2, :, :]), dim=-3)
        out2 = torch.cat((x1_hat_warp * mask_weights_R[:, 0:1, :, :], x2_hat * mask_weights_R[:, 1:2, :, :]), dim=-3)

        out1 = self.conv1(out1)
        out2 = self.conv1(out2)

        out1 = self.EBl1(out1)
        out2 = self.EBr1(out2)

        out1_warp = kornia.warp_perspective(out1, h_matrix, (out1.size()[-2], out1.size()[-1]))
        out2_warp = kornia.warp_perspective(out2, h_inv, (out2.size()[-2], out2.size()[-1]))
        out1 = torch.cat((out1 * mask_weights_L[:, 1:2, :, :], out2_warp * mask_weights_L[:, 0:1, :, :]), dim=-3)
        out2 = torch.cat((out2 * mask_weights_R[:, 1:2, :, :], out1_warp * mask_weights_R[:, 0:1, :, :]), dim=-3)
        out1 = self.EBl2(out1)
        out2 = self.EBr2(out2)

        out1 = torch.cat((out1, x1_hat_conv), dim=-3)
        out2 = torch.cat((out2, x2_hat_conv), dim=-3)
        out1 = self.EBl3(out1)
        out2 = self.EBr3(out2)

        out1 = self.conv2(out1)
        out2 = self.conv2(out2)

        # 增强后
        x1_hat2 = out1 + identity1
        x2_hat2 = out2 + identity2

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
