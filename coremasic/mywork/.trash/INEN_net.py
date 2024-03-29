#INEN 增强网络 单独用 只需关注Independent_EN


import argparse
import math
import random
import shutil
import os
import sys
import numpy as np
import math

#在这里改GMM计算方式
from compressai.entropy_models import (EntropyBottleneck, GaussianMixtureConditional, GaussianConditional)
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

##z1 for y1
class gmm_hyper_y1(nn.Module):
    def __init__(self,N,M,K): #K表示GMM对应正态分布个数
        super().__init__()
        self.N = N
        self.M = M
        self.K = K

        # 每个都是上采样4倍才行
        self.gmm_sigma = nn.Sequential(
            deconv(in_channels=N,out_channels=N,kernel_size=5), #stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.ReLU(inplace=True),

            deconv(in_channels=N, out_channels=N, kernel_size=5),# stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.ReLU(inplace=True),

            conv(in_channels=N,out_channels=(M*K),kernel_size=5,stride=1), #padding=kernel_size//2
            nn.ReLU(inplace=True),
        )

        self.gmm_means = nn.Sequential(
            deconv(in_channels=N, out_channels=N, kernel_size=5),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            deconv(in_channels=N, out_channels=N, kernel_size=5),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            conv(in_channels=N, out_channels=(M * K), kernel_size=5, stride=1),  # padding=kernel_size//2
        )


        self.gmm_weights = nn.Sequential(
            deconv(in_channels=N, out_channels=N, kernel_size=5),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            deconv(in_channels=N, out_channels=(M*K), kernel_size=5),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
            spatial_pool2d(),
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
        temp = torch.reshape(self.gmm_weights(z1),(-1,self.K,self.M,1,1)) #方便后续reshape合并时同一个M的数据相邻成组
        temp = nn.functional.softmax(temp,dim=-4)  #每个Mixture进行一次归一
        self.weights = torch.reshape(temp,(-1,self.M*self.K,1,1))
        ###
        return self.sigma,self.means,self.weights

##z2+y1 for y2
class gmm_hyper_y2(nn.Module):
    def __init__(self,N,M,K): #K表示GMM对应正态分布个数
        super().__init__()
        self.N = N
        self.M = M
        self.K = K

        self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=4) #固定为4倍 因为z和y分辨率本身就差4倍

        # 输入与y1分辨率相同，但通道数是2倍
        self.gmm_sigma = nn.Sequential(
            conv(in_channels=(N+M),out_channels=N,kernel_size=5,stride=1),
            nn.ReLU(inplace=True),

            conv(in_channels=N,out_channels=N,kernel_size=5,stride=1),
            nn.ReLU(inplace=True),

            conv(in_channels=N,out_channels=(M*K),kernel_size=5,stride=1), #padding=kernel_size//2
            nn.ReLU(inplace=True),
        )

        self.gmm_means = nn.Sequential(
            conv(in_channels=(N+M),out_channels=N,kernel_size=5,stride=1),
            nn.LeakyReLU(inplace=True),

            conv(in_channels=N,out_channels=N,kernel_size=5,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            conv(in_channels=N, out_channels=(M * K), kernel_size=5, stride=1),  # padding=kernel_size//2
        )


        self.gmm_weights = nn.Sequential(
            conv(in_channels=(N+M),out_channels=N,kernel_size=5,stride=1),
            nn.LeakyReLU(inplace=True),

            conv(in_channels=N,out_channels= (M*K),kernel_size=5,stride=1),
            # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
            spatial_pool2d(),
            nn.LeakyReLU(inplace=True),

            conv(in_channels=(M*K), out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
            #出去后要接一个softmax层，表示概率！！
        )

    def forward(self,z2,y1):
        self.up_z2 = self.upsample_layer(z2)
        self.cat_in = torch.cat((self.up_z2,y1),dim=-3)

        self.sigma = self.gmm_sigma(self.cat_in)
        self.means = self.gmm_means(self.cat_in)
        #softmax!!
        # self.weights = nn.functional.softmax(self.gmm_weights(self.cat_in),dim=-3)
        # 修正
        temp = torch.reshape(self.gmm_weights(self.cat_in), (-1, self.K, self.M,  1, 1))  # 方便后续reshape合并时同一个M的数据相邻成组
        temp = nn.functional.softmax(temp, dim=-4)  # 每个Mixture进行一次归一
        self.weights = torch.reshape(temp, (-1, self.M * self.K, 1, 1))
        ###

        return self.sigma,self.means,self.weights

##for y1 gmm _same_resolution
class gmm_hyper_y1_same_resolution(nn.Module):
    def __init__(self,N,M,K): #K表示GMM对应正态分布个数
        super().__init__()
        self.N = N
        self.M = M
        self.K = K

        # 已经同分辨率，不需要上采样
        self.gmm_sigma = nn.Sequential(
            deconv(in_channels=4*M,out_channels=2*M,kernel_size=5,stride=1), #,stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.ReLU(inplace=True),

            deconv(in_channels=2*M, out_channels=2*M, kernel_size=5,stride=1),# stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.ReLU(inplace=True),

            conv(in_channels=2*M,out_channels=(M*K),kernel_size=5,stride=1), #padding=kernel_size//2
            nn.ReLU(inplace=True),
        )

        self.gmm_means = nn.Sequential(
            deconv(in_channels=4*M, out_channels=2*M, kernel_size=5,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            deconv(in_channels=2*M, out_channels=2*M, kernel_size=5,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            conv(in_channels=2*M, out_channels=(M * K), kernel_size=5, stride=1),  # padding=kernel_size//2
        )


        self.gmm_weights = nn.Sequential(
            deconv(in_channels=4*M, out_channels=2*M, kernel_size=5,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            deconv(in_channels=2*M, out_channels=(M*K), kernel_size=5,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
            spatial_pool2d(),
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
        temp = torch.reshape(self.gmm_weights(z1),(-1,self.K,self.M,1,1)) #方便后续reshape合并时同一个M的数据相邻成组
        temp = nn.functional.softmax(temp,dim=-4)  #每个Mixture进行一次归一
        self.weights = torch.reshape(temp,(-1,self.M*self.K,1,1))
        ###
        return self.sigma,self.means,self.weights

##z2_up+y1 for y2 gmm _same_resolution
class gmm_hyper_y2_same_resolution(nn.Module):
    def __init__(self,N,M,K): #K表示GMM对应正态分布个数
        super().__init__()
        self.N = N
        self.M = M
        self.K = K

        self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=4) #固定为4倍 因为z和y分辨率本身就差4倍

        # 输入与y1分辨率相同，但通道数不同 2倍4M+2M -> 6*M
        self.gmm_sigma = nn.Sequential(
            conv(in_channels=6*M,out_channels=4*M,kernel_size=5,stride=1),
            nn.ReLU(inplace=True),

            conv(in_channels=4*M,out_channels=2*M,kernel_size=5,stride=1),
            nn.ReLU(inplace=True),

            conv(in_channels=2*M,out_channels=(M*K),kernel_size=5,stride=1), #padding=kernel_size//2
            nn.ReLU(inplace=True),
        )

        self.gmm_means = nn.Sequential(
            conv(in_channels=6*M,out_channels=4*M,kernel_size=5,stride=1),
            nn.LeakyReLU(inplace=True),

            conv(in_channels=4*M,out_channels=2*M,kernel_size=5,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(inplace=True),

            conv(in_channels=2*M, out_channels=(M * K), kernel_size=5, stride=1),  # padding=kernel_size//2
        )


        self.gmm_weights = nn.Sequential(
            conv(in_channels=6*M,out_channels=4*M,kernel_size=5,stride=1),
            nn.LeakyReLU(inplace=True),

            conv(in_channels=4*M,out_channels= (M*K),kernel_size=5,stride=1),
            # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
            spatial_pool2d(),
            nn.LeakyReLU(inplace=True),

            conv(in_channels=(M*K), out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
            #出去后要接一个softmax层，表示概率！！
        )

    def forward(self,allconcat):
        # self.up_z2 = self.upsample_layer(z2)
        # self.cat_in = torch.cat((self.up_z2,y1),dim=-3)

        self.sigma = self.gmm_sigma(allconcat)
        self.means = self.gmm_means(allconcat)
        #softmax!!
        # self.weights = nn.functional.softmax(self.gmm_weights(self.cat_in),dim=-3)
        # 修正
        temp = torch.reshape(self.gmm_weights(allconcat), (-1, self.K, self.M,  1, 1))  # 方便后续reshape合并时同一个M的数据相邻成组
        temp = nn.functional.softmax(temp, dim=-4)  # 每个Mixture进行一次归一
        self.weights = torch.reshape(temp, (-1, self.M * self.K, 1, 1))
        ###

        return self.sigma,self.means,self.weights

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


class HSIC(CompressionModel):
    def __init__(self,N=128,M=192,K=5,**kwargs): #'cuda:0' or 'cpu'
        super().__init__(entropy_bottleneck_channels=N, **kwargs)
        # super(DSIC, self).__init__()
        # self.entropy_bottleneck1 = CompressionModel(entropy_bottleneck_channels=N)
        # self.entropy_bottleneck2 = CompressionModel(entropy_bottleneck_channels=N)
        self.gaussian1 = GaussianMixtureConditional(K = K)
        self.gaussian2 = GaussianMixtureConditional(K = K)
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

        # self.need_int = need_int
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
        gmm2_jq = self._h_s2_same_resolution(torch.cat((params2, ctx_params2,ctx_params1), dim=1)) #通道数 4M
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
            'likelihoods':{
                'y1': y1_likelihoods,
                'y2': y2_likelihoods,
                'z1': z1_likelihoods,
                'z2': z2_likelihoods,
            }
        }


    ###codec
    def compress(self,x1,x2,h_matrix):
        # 定义结构
        y1, g1_1, g1_2, g1_3 = self.encoder1(x1)
        z1 = self._h_a1(y1)
        z1_strings = self.entropy_bottleneck1.compress(z1)
        z1_hat = self.entropy_bottleneck1.decompress(z1_strings)
        gmm1 = self._h_s1(z1_hat)  # 三要素  #scales_hat





        # y1_hat, y1_likelihoods = self.gaussian1(y1, gmm1[0], gmm1[1], gmm1[2])  # sigma
        #
        # x1_hat, g1_4, g1_5, g1_6 = self.decoder1(y1_hat)
        #
        # #############################################
        # # encoder
        # x1_warp = kornia.warp_perspective(x1, h_matrix, (x1.size()[-2], x1.size()[-1]))
        # y2 = self.encoder2(x1_warp, x2)
        # ##end encoder
        #
        # # hyper for pic2
        # z2 = self._h_a2(y2)
        # z2_hat, z2_likelihoods = self.entropy_bottleneck2(z2)
        # gmm2 = self._h_s2(z2_hat, y1_hat)  # 三要素
        #
        # y2_hat, y2_likelihoods = self.gaussian2(y2, gmm2[0], gmm2[1], gmm2[2])  # 这里也是临时，待改gmm
        # # end hyper for pic2
        #
        # ##decoder
        # x1_hat_warp = kornia.warp_perspective(x1_hat, h_matrix, (x1_hat.size()[-2], x1_hat.size()[-1]))
        # x2_hat = self.decoder2(y2_hat, x1_hat_warp)
        # # end decoder
        # # print(x1.size())
        #
        # return {
        #     'x1_hat': x1_hat,
        #     'x2_hat': x2_hat,
        #     'likelihoods': {
        #         'y1': y1_likelihoods,
        #         'y2': y2_likelihoods,
        #         'z1': z1_likelihoods,
        #         'z2': z2_likelihoods,
        #     }
        # }


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




