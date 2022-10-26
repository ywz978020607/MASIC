import kornia
import torch
import numpy as np

# def mask(im1, H_inv):
#     # # mask1 = torch.ones_like(im1)
#     # mask1 = kornia.warp_perspective(im1, H_inv, (im1.shape[-2], im1.shape[-1]))
#
#     im1 = torch.ones_like(im1)
#     mask1 = kornia.warp_perspective(im1, H_inv, (im1.shape[-2], im1.shape[-1]))
#
#     mask1[np.nonzero(mask1.numpy())] = 1.
#     mask2 = kornia.warp_perspective(mask1, torch.inverse(H_inv), (im1.shape[-2], im1.shape[-1]))
#     mask2[np.nonzero(mask2.numpy())] = 1.
#     return mask1, mask2

###########################################################################
# torch.where(a!=0,1,0)
# def mask(im1, H_inv):
#     # # mask1 = torch.ones_like(im1)
#     # mask1 = kornia.warp_perspective(im1, H_inv, (im1.shape[-2], im1.shape[-1]))
#
#     # # batch_size > 1时的情况要兼容
#     # print(H_inv.shape)
#     # raise ValueError("stop h shape")
#
#     # im1 = torch.ones_like(im1)
#     im1 = torch.ones([im1.shape[0],1,im1.shape[-2],im1.shape[-1]], dtype=im1.dtype, layout=im1.layout, device=im1.device)
#     mask1 = kornia.warp_perspective(im1, H_inv, (im1.shape[-2], im1.shape[-1]))
#
#     # mask1[torch.nonzero(mask1)] = 1.
#     torch.where(mask1!=0,1,0)
#     mask2 = kornia.warp_perspective(mask1, torch.inverse(H_inv), (im1.shape[-2], im1.shape[-1]))
#     # mask2[torch.nonzero(mask2)] = 1.
#     torch.where(mask2 != 0, 1, 0)
#     return mask1, mask2

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
    # torch.where(mask1!=0,1,0)
    torch.where(mask1 != 0, torch.Tensor([1]).to(im1.device), torch.Tensor([0]).to(im1.device))

    mask2 = kornia.warp_perspective(mask1, torch.inverse(H_inv), (im1.shape[-2], im1.shape[-1]))  # 换算回左目的mask
    # mask2[torch.nonzero(mask2)] = 1.
    # torch.where(mask2 != 0, 1, 0)
    torch.where(mask2 != 0, torch.Tensor([1]).to(im1.device), torch.Tensor([0]).to(im1.device))

    return mask1, mask2