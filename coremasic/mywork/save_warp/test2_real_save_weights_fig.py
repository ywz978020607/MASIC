# test2.py PSNR计算后再平均
# 单gpu版  和训练代码一样，只是把训练部分注释
#python test2.py -d "/home/ywz/database/aftercut512"  --seed 0  --patch-size 512 512 --batch-size 1 --test-batch-size 1 --cuda 0
#cpu版
#python test2.py -d "/home/ywz/database/aftercut512"  --seed 0  --patch-size 512 512 --batch-size 1 --test-batch-size 1
import argparse
import math
import random
import shutil
import os
import os.path as osp
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
import kornia, imageio

#net defination

out_root_path = "out_pic"
if not os.path.exists(out_root_path):
    print("not ex")
    os.system("mkdir "+out_root_path)
y1_save_path = osp.join(out_root_path,"y1_nom")
y2_save_path = osp.join(out_root_path,"y2_nom")
z2_save_path = osp.join(out_root_path,"z2_nom")
overlap_save_path = osp.join(out_root_path,"overlap")
if not os.path.exists(y1_save_path):
    os.system("mkdir " + y1_save_path)
if not os.path.exists(y2_save_path):
    os.system("mkdir " + y2_save_path)
if not os.path.exists(z2_save_path):
    os.system("mkdir " + z2_save_path)
if not os.path.exists(overlap_save_path):
    os.system("mkdir " + overlap_save_path)


# out_root_path_1 = "out_pic_1"
# if not os.path.exists(out_root_path_1):
#     print("not ex")
#     os.system("mkdir "+out_root_path_1)
# left_save_path_1 = osp.join(out_root_path_1,"left")
# right_save_path_1 = osp.join(out_root_path_1,"right")
# if not os.path.exists(left_save_path_1):
#     os.system("mkdir " + left_save_path_1)
# if not os.path.exists(right_save_path_1):
#     os.system("mkdir " + right_save_path_1)

# out_root_path_2 = "out_pic_2"
# if not os.path.exists(out_root_path_2):
#     print("not ex")
#     os.system("mkdir "+out_root_path_2)
# left_save_path_2 = osp.join(out_root_path_2,"left")
# right_save_path_2 = osp.join(out_root_path_2,"right")
# if not os.path.exists(left_save_path_2):
#     os.system("mkdir " + left_save_path_2)
# if not os.path.exists(right_save_path_2):
#     os.system("mkdir " + right_save_path_2)

def save_pic(data,path):
    if osp.exists(path):
        os.system("rm "+path)
        print("rm "+path)
    # print(data.shape)
    # raise ValueError('stop')
    data[:,:,0,0] = data[:,:,1,1]
    max = torch.max(data)
    im = data/max*1
    reimage = im.cpu().clone()
    reimage[reimage > 1.0] = 1.0

    reimage = reimage.squeeze(0)
    reimage = transforms.ToPILImage()(reimage)  # PIL格式
    reimage.save(path)

from MASIC import *
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

    def forward(self, output, target1,target2):
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp_loss'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'].values())
        out['mse_loss'] = self.mse(output['x1_hat'], target1) + self.mse(output['x2_hat'], target2)        #end to end
        out['bpp1'] = (torch.log(output['likelihoods']['y1']).sum() / (-math.log(2) * num_pixels)) + (
                torch.log(output['likelihoods']['z1']).sum() / (-math.log(2) * num_pixels))
        out['bpp2'] = (torch.log(output['likelihoods']['y2']).sum() / (-math.log(2) * num_pixels)) + (
                torch.log(output['likelihoods']['z2']).sum() / (-math.log(2) * num_pixels))

        out['loss'] = self.lmbda * 255**2 * out['mse_loss'] + out['bpp_loss']

        out['ms_ssim1'] = ms_ssim(output['x1_hat'], target1, data_range=1, size_average=False)[0]  # (N,)
        out['ms_ssim2'] = ms_ssim(output['x2_hat'], target2, data_range=1, size_average=False)[0]
        out['ms_ssim'] = (out['ms_ssim1']+out['ms_ssim2'])/2
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

    # for i, d in enumerate(train_dataloader):
    #     # print("dataloader::"+str(i))
    #     d1 = d[0].to(device)  #load to gpu/cpu
    #     d2 = d[1].to(device)  # load to gpu/cpu
    #     h_matrix = d[2].to(device)
    #
    #     optimizer.zero_grad()
    #     aux_optimizer.zero_grad()
    #
    #     out_net = model(d1,d2,h_matrix)
    #
    #     out_criterion = criterion(out_net, d1,d2)
    #     out_criterion['loss'].backward()
    #     optimizer.step()
    #
    #     aux_loss = model.aux_loss()
    #     aux_loss.backward()
    #     aux_optimizer.step()
    #
    #     if i % 10 == 0:
    #         log_data = f'Train epoch {epoch}: ['\
    #               f'{i*d[0].size()[0]}/{len(train_dataloader.dataset)}'\
    #               f' ({100. * i / len(train_dataloader):.0f}%)]'\
    #               f'\tLoss: {out_criterion["loss"].item():.3f} |'\
    #               f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'\
    #               f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'\
    #               f'\tAux loss: {aux_loss.item():.2f}'
    #
    #         print(log_data)
    #         f = open(log_file,'a')
    #         f.write(log_data)
    #         f.write("\n")
    #         f.close()


def test_epoch(epoch, test_dataloader,modelhomo, model,criterion):
    modelhomo.eval() #homo
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    ssim_loss = AverageMeter()
    ssim_loss1 = AverageMeter()
    ssim_loss2 = AverageMeter()

    psnr1 = AverageMeter()
    psnr2 = AverageMeter()
    bpp1 = AverageMeter()
    bpp2 = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d1 = d[0].to(device)
            d2 = d[1].to(device)
            # h_matrix = d[2].to(device)
            # 通过homo获取h_matrix 注意要加逆变换
            # print(len(d))
            homo_img1 = d[-3].to(device)
            homo_img2 = d[-2].to(device)
            homo_corners = d[-1].to(device)

            homo_corners = homo_corners - homo_corners[:, 0].view(-1, 1, 2)

            delta_hat = modelhomo(homo_img1, homo_img2)
            homo_corners_hat = homo_corners + delta_hat
            h = kornia.get_perspective_transform(homo_corners, homo_corners_hat)
            h_matrix = torch.inverse(h)
            h_matrix = h_adjust(d1.shape[-2], d1.shape[-1], pic_size, pic_size, h_matrix)

            out_net = model(d1,d2,h_matrix)
            out_criterion = criterion(out_net, d1,d2)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion['bpp_loss'])
            loss.update(out_criterion['loss'])
            mse_loss.update(out_criterion['mse_loss'])
            ssim_loss.update(out_criterion['ms_ssim'])  # 已除2
            ssim_loss1.update(out_criterion['ms_ssim1'])  # 已除2
            ssim_loss2.update(out_criterion['ms_ssim2'])  # 已除2

            psnr1.update(out_criterion['psnr1'])
            psnr2.update(out_criterion['psnr2'])
            bpp1.update(out_criterion['bpp1'])
            bpp2.update(out_criterion['bpp2'])
            save_pic(out_net['x1_mask_R'],osp.join(overlap_save_path, str(d[3]).split("'")[1]))
            print(out_net['x1_mask_R'])
            save_pic(out_net['mask_weights'][:,0:1,:,:]*255,osp.join(z2_save_path, str(d[3]).split("'")[1]))
            print(out_net['mask_weights'][:,0:1,:,:])
            save_pic(out_net['mask_weights'][:,1:2,:,:]*255,osp.join(y2_save_path, str(d[3]).split("'")[1]))
            print(out_net['mask_weights'][:,1:2,:,:])
            save_pic(out_net['mask_weights'][:,2:3,:,:]*255,osp.join(y1_save_path, str(d[3]).split("'")[1]))
            print(out_net['mask_weights'][:,2:3,:,:])
    print(f'Test epoch {epoch}: Average losses:'
          f'\tTime: {time.strftime("%Y-%m-%d %H:%M:%S")} |'
          f'\tLoss: {loss.avg:.3f} |'
          f'\tMSE loss: {mse_loss.avg:.4f} |'
          f'\tPSNR (dB): {(psnr1.avg+psnr2.avg)/2:.3f} |'  #平均一张图的PSNR
          f'\tBpp loss: {bpp_loss.avg/2:.4f} |'
          f'\tMS-SSIM: {ssim_loss.avg:.4f} |'  #已除2，相加时候便除了2
          f'\tMS-SSIM1: {ssim_loss1.avg:.4f} |'
          f'\tMS-SSIM2: {ssim_loss2.avg:.4f} |'
          f'\tPSNR1: {psnr1.avg:.3f} |'
          f'\tPSNR2: {psnr2.avg:.3f} |'
          f'\tBPP1: {bpp1.avg:.3f} |'
          f'\tBPP2: {bpp2.avg:.3f} |'
          
          f'\tAux loss: {aux_loss.avg:.2f}\n')

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
                                transform=train_transforms)
    test_dataset = ImageFolder(args.dataset,
                               split='test',
                               patch_size=args.patch_size,
                               transform=test_transforms,
                               need_file_name=True,
                               need_H=False)

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
    net = HSIC(N=128,M=192,K=5)
    #加载最新模型继续训练
    # 加载最新模型继续训练
    if os.path.exists("homo_best.pth.tar"):
        model = torch.load('homo_best.pth.tar', map_location=lambda storage, loc: storage)
        model.keys()
        # net.load_state_dict(torch.load('path/params.pkl'))
        nethomo.load_state_dict(model['state_dict'])
        print("load homo model ok")
    else:
        print("homo from none")

    if os.path.exists("checkpoint_best_loss.pth.tar"):
        model = torch.load('checkpoint_best_loss.pth.tar', map_location=lambda storage, loc: storage)
        model.keys()
        # net.load_state_dict(torch.load('path/params.pkl'))
        net.load_state_dict(model['state_dict'])
        print("load model ok")
    else:
        print("train from none")
    nethomo = nethomo.to(device)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    aux_optimizer = optim.Adam(net.aux_parameters(), lr=args.aux_learning_rate)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    # best_loss = 1e10
    # for epoch in range(args.epochs):
    for epoch in [0]:#只跑一次
        # train_epoch(epoch, train_dataloader, net, criterion, optimizer,
        #             aux_optimizer,log_file=args.logfile)

        # try:
        #验证集
        loss = test_epoch(epoch, test_dataloader,nethomo, net, criterion)

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


