# 注意此代码没有加cqe部分！
# test2.py PSNR计算后再平均
# 单gpu版  和训练代码一样，只是把训练部分注释
#python MASIC_save_jg_codec.py -d "/home/yangwenzhe/database/aftercut512"  --seed 0 --cuda 0 --patch-size 512 512 --batch-size 1 --test-batch-size 1
#cpu版
#python MASIC_save_jg_codec.py -d "/home/ywz/database/aftercut512"  --seed 0  --patch-size 512 512 --batch-size 1 --test-batch-size 1

#net defination
from MASIC import *


import argparse
import math
import random
import shutil
import os.path as osp
import glob
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

##去随机--2021.10.29
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

out_root_path = "out_pic2"
if not os.path.exists(out_root_path):
    print("not ex")
    os.system("mkdir "+out_root_path)
left_save_path = osp.join(out_root_path,"left")
right_save_path = osp.join(out_root_path,"right")
if not os.path.exists(left_save_path):
    os.system("mkdir " + left_save_path)
if not os.path.exists(right_save_path):
    os.system("mkdir " + right_save_path)

#file
out_root_path_file = open(osp.join(out_root_path,"details.txt"),'w')

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

    def forward(self, output, target1,target2):
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

########################################
        # 计算误差
        # out['bpp_loss'] = sum(
        #     (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        #     for likelihoods in output['likelihoods'].values())
        # # out['mse_loss'] = self.mse(output['x1_hat'], target1) + self.mse(output['x2_hat'], target2)        #end to end
        # out['bpp1'] = (torch.log(output['likelihoods']['y1']).sum() / (-math.log(2) * num_pixels)) + (
        #             torch.log(output['likelihoods']['z1']).sum() / (-math.log(2) * num_pixels))
        # out['bpp2'] = (torch.log(output['likelihoods']['y2']).sum() / (-math.log(2) * num_pixels)) + (
        #             torch.log(output['likelihoods']['z2']).sum() / (-math.log(2) * num_pixels))

        # out['loss'] = self.lmbda * 255**2 * out['mse_loss'] + out['bpp_loss']
        # out['ms_ssim1'] = ms_ssim(output['x1_hat'], target1, data_range=1, size_average=False)[0]  # (N,)
        # out['ms_ssim2'] = ms_ssim(output['x2_hat'], target2, data_range=1, size_average=False)[0]
        # out['ms_ssim'] = (out['ms_ssim1'] + out['ms_ssim2']) / 2

########################################
        out['psnr1'] = mse2psnr(self.mse(output['x1_hat'], target1))
        out['psnr2'] = mse2psnr(self.mse(output['x2_hat'], target2))

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


def train_epoch(epoch, train_dataloader, model, criterion, optimizer,
                aux_optimizer,log_file = "log.txt"):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        # print("dataloader::"+str(i))
        d1 = d[0].to(device)  #load to gpu/cpu
        d2 = d[1].to(device)  # load to gpu/cpu
        h_matrix = d[2].to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d1,d2,h_matrix)

        out_criterion = criterion(out_net, d1,d2)
        out_criterion['loss'].backward()
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            log_data = f'Train epoch {epoch}: ['\
                  f'{i*len(d)}/{len(train_dataloader.dataset)}'\
                  f' ({100. * i / len(train_dataloader):.0f}%)]'\
                  f'\tLoss: {out_criterion["loss"].item():.3f} |'\
                  f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'\
                  f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'\
                  f'\tAux loss: {aux_loss.item():.2f}'

            print(log_data)
            f = open(log_file,'a')
            f.write(log_data)
            f.write("\n")
            f.close()

def save_pic(data,path):
    if osp.exists(path):
        os.system("rm "+path)
        print("rm "+path)
    reimage = data.cpu().clone()
    reimage[reimage > 1.0] = 1.0

    reimage = reimage.squeeze(0)
    reimage = transforms.ToPILImage()(reimage)  # PIL格式
    reimage.save(path)

def test_epoch(epoch, test_dataloader, model, criterion):
    global out_root_path_file

    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()

    psnr1 = AverageMeter()
    psnr2 = AverageMeter()
    realbpp = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            # for ii in range(len(d)):
            #     print(d[ii])
            # raise  ValueError("stop")
            d1 = d[0].to(device)
            d2 = d[1].to(device)
            h_matrix = d[2].to(device)

            #====================
            #codec
            print("start codec")
            name = str(d[3]).split("'")[1].split(".")[0]
            print(name)

            out_net = model(d1, d2, h_matrix)
            # print("network z1hat",out_net['z1_hat'])
            # print("network y1hat",out_net['y1_hat'])

            output = out_net
            out={}
            N, _, H, W = d1.size()
            out = {}
            num_pixels = N * H * W
            out['bpp_loss'] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output['likelihoods'].values())
            # out['mse_loss'] = self.mse(output['x1_hat'], target1) + self.mse(output['x2_hat'], target2)        #end to end
            out['bpp1'] = (torch.log(output['likelihoods']['y1']).sum() / (-math.log(2) * num_pixels)) + (
                    torch.log(output['likelihoods']['z1']).sum() / (-math.log(2) * num_pixels))
            out['bpp2'] = (torch.log(output['likelihoods']['y2']).sum() / (-math.log(2) * num_pixels)) + (
                    torch.log(output['likelihoods']['z2']).sum() / (-math.log(2) * num_pixels))
            print("bpp_loss",out['bpp_loss'])
            print("bpp1", out['bpp1'])
            print("bpp2", out['bpp2'])
            ###################################################################################################################
            #encode
            out_net_en = model.compress(d1,d2,h_matrix, name,output_path=out_root_path,device=device)
            print("bpp",out_net_en['bpp_real'])
            print("=============enc ok=============")
            # raise ValueError("enc ok!")
            #decode
            out_net = model.decompress(d1,d2,h_matrix, name,output_path=out_root_path,device=device)
            # print(out_net_en)
            # print(out_net)
            # print("compare:",bool(out_net_en==out_net))
            # raise ValueError("stop")
            print("=============dec ok=============")
            # print("dec y1_hat",out_net['y1_hat'])
            save_pic(out_net['x1_hat'], "out_test_0.jpg")
            save_pic(out_net['x2_hat'], "out_test_1.jpg")
            raise ValueError("success!")
            #====================

            out_criterion = criterion(out_net, d1,d2)

            psnr1.update(out_criterion['psnr1'])
            psnr2.update(out_criterion['psnr2'])
            realbpp.update(out_net_en['bpp_real'])

            psnr1_val = out_criterion['psnr1']
            psnr2_val = out_criterion['psnr2']
            realbpp_val = out_net_en['bpp_real']
            # bpp1_val = out_criterion1['bpp1']
            # bpp2_val = out_criterion1['bpp2']

            print_context = (str(d[3]).split("'")[1] +
                             f'\tPSNR (dB): {(psnr1_val + psnr2_val) / 2:.3f} |'  # 平均一张图的PSNR
                             f'\tReal_bpp: {realbpp_val:.3f} |'
                             f'\tPSNR1: {psnr1_val:.3f} |'
                             f'\tPSNR2: {psnr2_val:.3f} \n')

            out_root_path_file.write(print_context)
            print(print_context)

            ##save pic
            save_pic(out_net['x1_hat'], osp.join(left_save_path, str(d[3]).split("'")[1]))
            save_pic(out_net['x2_hat'], osp.join(right_save_path, str(d[3]).split("'")[1]))
            print(str(d[3]).split("'")[1])
            ####

    out_root_path_file.close()
    print(f'Test epoch {epoch}: Average losses:'
          f'\tTime: {time.strftime("%Y-%m-%d %H:%M:%S")} |'
          f'\tReal_bpp: {realbpp.avg:.3f} |'
          f'\tPSNR (dB): {(psnr1.avg + psnr2.avg) / 2:.3f} |'  # 平均一张图的PSNR
          f'\tPSNR1: {psnr1.avg:.3f} |'
          f'\tPSNR2: {psnr2.avg:.3f} \n'
          )

    return loss.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoint_best_loss.pth.tar')


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
    # yapf: enable
    args = parser.parse_args(argv)
    return args

def main(argv):
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


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda' if (torch.cuda.is_available() and args.cuda != -1) else 'cpu'
    print(device)
    if device=='cuda':
        torch.cuda.set_device(args.cuda)
        ##去随机--2021.10.29
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('temp gpu device number:')
    print(torch.cuda.current_device())
    #net assign
    # with torch.autograd.set_detect_anomaly(True): #for debug gradient
    # net = DSIC(N=128,M=192,F=21,C=32,K=5) #(N=128,M=192,F=21,C=32,K=5)
    net = HSIC(N=128,M=192,K=5)
    # net = HSIC(N=128, M=192, K=15)
    #加载最新模型继续训练
    if os.path.exists("checkpoint_best_loss.pth.tar"):
        model = torch.load('checkpoint_best_loss.pth.tar', map_location=lambda storage, loc: storage)
        model.keys()

        # 非严格加载
        new_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in model['state_dict'].items() if k in new_dict}#filter out unnecessary keys
        new_dict.update(pretrained_dict)
        net.load_state_dict(new_dict)

        # 严格加载
        # net.load_state_dict(model['state_dict'])

        print("load model ok")
    else:
        print("train from none")


    net.entropy_bottleneck1.update()
    net.entropy_bottleneck2.update()

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    aux_optimizer = optim.Adam(net.aux_parameters(), lr=args.aux_learning_rate)
    print("lambda:",args.lmbda)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    best_loss = 1e10
    # for epoch in range(args.epochs):
    #     train_epoch(epoch, train_dataloader, net, criterion, optimizer,
    #                 aux_optimizer,log_file=args.logfile)
    for epoch in [0]:  # 只跑一次
        # try:
        #验证集
        loss = test_epoch(epoch, test_dataloader, net, criterion)

        # is_best = loss < best_loss
        # best_loss = min(loss, best_loss)
        # if args.save:
        #     save_checkpoint(
        #         {
        #             'epoch': epoch + 1,
        #             'state_dict': net.state_dict(),
        #             'loss': loss,
        #             'optimizer': optimizer.state_dict(),
        #             'aux_optimizer': aux_optimizer.state_dict(),
        #         }, is_best)
        # except:
        #     print("val error")
        #     if args.save:
        #         state = {
        #                 'epoch': epoch + 1,
        #                 'state_dict': net.state_dict(),
        #                 'loss': 'none',
        #                 'optimizer': optimizer.state_dict(),
        #                 'aux_optimizer': aux_optimizer.state_dict(),
        #             }
        #         torch.save(state, 'checkpoint.pth.tar')

if __name__ == '__main__':
    main(sys.argv[1:])


