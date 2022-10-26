# test3_real: udh+HSIC+CQE

#python homo_real_or_not.py -d "/home/ywz/database/aftercut512"  --seed 0 --cuda 0 --patch-size 512 512 --batch-size 1 --test-batch-size 1

#net defination
from pickletools import uint8
from MASIC import *
import random,cv2
import argparse
import math
import random
import shutil
import os,glob
import os.path as osp
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import transforms

import csv
from compressai.datasets import ImageFolder
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import kornia, imageio
import lpips
loss_fn = lpips.LPIPS(net='alex', spatial=False) # Can also set net = 'squeeze' or 'vgg'
from thop import profile
from thop import clever_format
##save
out_root_path = "out_pic2"
if not os.path.exists(out_root_path):
    print("not ex")
    os.system("mkdir "+out_root_path)
left_save_path = osp.join(out_root_path,"warp_cv")
right_save_path = osp.join(out_root_path,"warp_udh")
if not os.path.exists(left_save_path):
    os.system("mkdir " + left_save_path)
if not os.path.exists(right_save_path):
    os.system("mkdir " + right_save_path)

#file

csvfile = open(osp.join(out_root_path,"details3.csv"), 'w', newline='') #

###homo
from model import Net, photometric_loss
pic_size = 256
patch_size = 128  #最好别变，可以改pic，可以获取角点进行缩放后求H
class HomographyModel(nn.Module):
    def __init__(self):
        super(HomographyModel, self).__init__()
        self.model = Net(patch_size=patch_size)
    def forward(self, a, b):
        return self.model(a, b)
def tensors_to_gif(a, b, name):
    a = a.permute(1, 2, 0).numpy()
    b = b.permute(1, 2, 0).numpy()
    imageio.mimsave(name, [a, b], duration=1)

def h_adjust(orishapea,orishapeb,resizeshapea,resizeshapeb, h): #->h_ori
    # a = original_img.shape[-2] / resized_img.shape[-2]
    # b = original_img.shape[-1] / resized_img.shape[-1]
    a = orishapea / resizeshapea
    b = orishapeb / resizeshapeb
    # the shape of H matrix should be (1, 3, 3)
    h[:, 0, :] = a*h[:, 0, :]
    h[:, :, 0] = (1./a)*h[:, :, 0]
    h[:, 1, :] = b * h[:, 1, :]
    h[:, :, 1] = (1. / b) * h[:, :, 1]
    return h
#################################################

def mse2psnr(mse):
    # 根据Hyper论文中的内容，将MSE->psnr(db)
    # return 10*math.log10(255*255/mse)
    return 10 * math.log10(1/ mse) #???
#psnr calculate
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

################################################################

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, target1,target2):
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        out['mse_loss'] = self.mse(target1, target2)        #end to end
        out['ms_ssim'] = ms_ssim(target1, target2, data_range=1, size_average=False)[0]
        out['psnr'] = mse2psnr(self.mse(target1, target2))
        out['lps'] = loss_fn.forward(target2, target1).mean().item()
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

def save_pic(data,path):
    if osp.exists(path):
        os.system("rm "+path)
        print("rm "+path)
    reimage = data.cpu().clone()
    reimage[reimage < 0.0] = 0.0
    reimage[reimage > 1.0] = 1.0

    reimage = reimage.squeeze(0)
    reimage = transforms.ToPILImage()(reimage)  # PIL格式
    reimage.save(path)

def test_epoch(epoch, test_dataloader,modelhomo, criterion):
    global csvfile
    writer = csv.writer(csvfile)

    modelhomo.eval() #homo
    # model.eval()
    device = next(modelhomo.parameters()).device

    loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    ssim_loss = AverageMeter()
    ssim_loss1 = AverageMeter()
    ssim_loss2 = AverageMeter()

    lps_loss = AverageMeter()
    lps_loss1 = AverageMeter()
    lps_loss2 = AverageMeter()

    psnr1 = AverageMeter()
    psnr2 = AverageMeter()

    test_transforms = transforms.Compose(
        [transforms.ToTensor()])

    with torch.no_grad():
        for d in test_dataloader:
            d1 = d[0].to(device)
            d2 = d[1].to(device)
            
            # ------------------------
            # 1.cv方法
            im1 = np.array(torch.squeeze(d[0]* 255).transpose(0, 2).transpose(0, 1), dtype='uint8') 
            im2 = np.array(torch.squeeze(d[1]* 255).transpose(0, 2).transpose(0, 1), dtype='uint8') 
            resize_scale = 1
            resize_im1 = cv2.resize(im1, (im1.shape[1] // resize_scale, im1.shape[0] // resize_scale))  # W,H
            resize_im2 = cv2.resize(im2, (im2.shape[1] // resize_scale, im2.shape[0] // resize_scale))
            #
            torch.cuda.synchronize()  # 增加同步操作
            start = time.time()
            surf = cv2.xfeatures2d.SURF_create()
            kp1, des1 = surf.detectAndCompute(resize_im1, None)
            kp2, des2 = surf.detectAndCompute(resize_im2, None)
            # 匹配特征点描述子
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # 提取匹配较好的特征点
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            # 通过特征点坐标计算单应性矩阵H
            # （findHomography中使用了RANSAC算法剔初错误匹配）
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # 获取H后，但要放进tensor中的变换
            try:
                h = torch.from_numpy(H.astype(np.float32))  # 否则float64，与网络中的tensor不匹配！
            except:
                h = None
            h_matrix_cv = h.unsqueeze(0).to(device)
            x1_hat_warp_cv = kornia.warp_perspective(d1, h_matrix_cv, (d1.size()[-2], d1.size()[-1]))
            torch.cuda.synchronize()  # 增加同步操作
            end = time.time()
            time_cv = end - start
            #+ 计算
            out_criterion_cv = criterion(x1_hat_warp_cv, d2)

            # ------------------------
            # 2. udh
            #通过homo获取h_matrix 注意要加逆变换
            homo_img1 = d[-3].to(device)
            homo_img2 = d[-2].to(device)
            homo_corners = d[-1].to(device)
            # h矩阵
            torch.cuda.synchronize()  # 增加同步操作
            start = time.time()
            homo_corners = homo_corners - homo_corners[:, 0].view(-1, 1, 2)
            delta_hat = modelhomo(homo_img1, homo_img2)
            homo_corners_hat = homo_corners + delta_hat
            h = kornia.get_perspective_transform(homo_corners, homo_corners_hat)
            h_matrix = torch.inverse(h)
            h_matrix = h_adjust(d1.shape[-2], d1.shape[-1], pic_size, pic_size, h_matrix)
            # h变换
            h_matrix = h_matrix.to(device)
            x1_hat_warp_udh = kornia.warp_perspective(d1, h_matrix, (d1.size()[-2], d1.size()[-1]))
            torch.cuda.synchronize()  # 增加同步操作
            end = time.time()
            time_udh = end - start
            #+ 计算
            out_criterion_udh = criterion(x1_hat_warp_udh, d2)
            #=====================================
            # ssim_loss1.update(out_criterion_cv['ms_ssim'])
            # lps_loss1.update(out_criterion_cv['lps'])
            # psnr1.update(out_criterion_cv['psnr'])

            # ssim_loss2.update(out_criterion_udh['ms_ssim'])
            # lps_loss2.update(out_criterion_udh['lps'])
            # psnr2.update(out_criterion_udh['psnr'])


            #当前图片
            writeRow = [float(out_criterion_cv['psnr']), float(out_criterion_cv['ms_ssim']),float(out_criterion_cv['lps']),time_cv,\
                "","","",\
                float(out_criterion_udh['psnr']),float(out_criterion_udh['ms_ssim']),float(out_criterion_udh['lps']),time_udh,\
                str(d[3]).split("'")[1]]

            writer.writerow(writeRow) #final close remember!
            ##save pic
            save_pic(x1_hat_warp_cv, osp.join(left_save_path, str(d[3]).split("'")[1]))
            save_pic(x1_hat_warp_udh, osp.join(right_save_path, str(d[3]).split("'")[1]))
            print(str(d[3]).split("'")[1])
            ####
    csvfile.close()

    return loss.avg


def save_checkpoint(state, is_best, filename='second_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'second_checkpoint_best_loss.pth.tar')


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Example training script')
    # yapf: disable
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        help='Training dataset')
    parser.add_argument(
        '-e',
        '--epochs',
        default=100,
        type=int,
        help='Number of epochs (default: %(default)s)')
    parser.add_argument(
        '-lr',
        '--learning-rate',
        default=1e-4,
        type=float,
        help='Learning rate (default: %(default)s)')
    parser.add_argument(
        '-n',
        '--num-workers',
        type=int,
        default= 3,
        help='Dataloaders threads (default: %(default)s)')
    parser.add_argument(
        '--lambda',
        dest='lmbda',
        type=float,
        default=1e-2,
        help='Bit-rate distortion parameter (default: %(default)s)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: %(default)s)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=64,
        help='Test batch size (default: %(default)s)')
    parser.add_argument(
        '--aux-learning-rate',
        default=1e-3,
        help='Auxiliary loss learning rate (default: %(default)s)')
    parser.add_argument(
        '--patch-size',
        type=int,
        nargs=2,
        default=(256, 256),
        help='Size of the patches to be cropped (default: %(default)s)')
    parser.add_argument(
        '--cuda',
        type=int,
        default=-1,
        help='Use cuda')
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save model to disk')
    parser.add_argument(
        '--logfile',
        type=str,
        default="train_log.txt",
        help='logfile_name')
    parser.add_argument(
        '--seed',
        type=float,
        help='Set random seed for reproducibility')
    parser.add_argument(
        '--homopath',
        default="homo_best.pth.tar",
        type=str,
        help='homo model path')
    # yapf: enable
    args = parser.parse_args(argv)
    return args


def main(argv):
    global loss_fn
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # train_transforms = transforms.Compose(
    #     [transforms.RandomCrop(args.patch_size),
    #      transforms.ToTensor()])
    #
    # test_transforms = transforms.Compose(
    #     [transforms.CenterCrop(args.patch_size),
    #      transforms.ToTensor()])
    train_transforms = transforms.Compose(
        [transforms.ToTensor()])

    test_transforms = transforms.Compose(
        [transforms.ToTensor()])

    train_dataset = ImageFolder(args.dataset,
                                split='train',
                                patch_size=args.patch_size,
                                transform=train_transforms,
                                need_file_name = True)
    test_dataset = ImageFolder(args.dataset,
                               split='test',
                               patch_size=args.patch_size,
                               transform=test_transforms,
                               need_file_name = True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  pin_memory=False)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=False)


    device = 'cuda' if (torch.cuda.is_available() and args.cuda!=-1) else 'cpu'
    print(device)
    if device=='cuda':
        torch.cuda.set_device(args.cuda)
    print('temp gpu device number:')
    print(torch.cuda.current_device())
    #net assign
    # with torch.autograd.set_detect_anomaly(True): #for debug gradient
    # net = DSIC(N=128,M=192,F=21,C=32,K=5) #(N=128,M=192,F=21,C=32,K=5)
    ##homo
    nethomo = HomographyModel()
    # net = HSIC(N=128, M=192, K=5)
    #也可用GMM_together() 调用一个网络包括整体  分开调用方便测试溶解效果

    # 加载最新模型继续训练
    if os.path.exists(args.homopath):
        model = torch.load(args.homopath, map_location=lambda storage, loc: storage)
        model.keys()
        # net.load_state_dict(torch.load('path/params.pkl'))
        nethomo.load_state_dict(model['state_dict'])
        print("load homo model ok")
    else:
        print("homo from none")

    nethomo = nethomo.to(device)
    # net = net.to(device)
    #lpips
    loss_fn = loss_fn.to(device)


    print("lambda:", args.lmbda)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    for epoch in [0]:  # 只跑一次
        loss = test_epoch(epoch, test_dataloader,nethomo, criterion)

if __name__ == '__main__':
    main(sys.argv[1:])