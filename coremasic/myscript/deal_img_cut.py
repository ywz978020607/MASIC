#deal_img_cut: 扩增数据集，降低重复区域


# 将2kInstereo 由1024,860 ->1024,832   832/16 = 52 ; 52/4 = 13
import os,glob
import os.path as osp

import cv2

#根目录 可以是移动硬盘，也可以是本地
root_path = "/media/ywz/disk/SIC/mm21/datasets/Flickr1024"
# root_path = "/home/sharklet/database"
#out
out_path = "/home/ywz/database/flickr"
#不存在则创建文件夹
if not osp.exists(out_path):
    os.system("mkdir "+out_path)
    print(out_path)
################################################################
def cut_pic(ori_file,out_file,W,H): #eg:W=1024,H=832
    image = cv2.imread(ori_file)  # numpy数组格式（H,W,C=3），通道顺序（B,G,R) H<W
    image = image[0:H,0:W] #H=832,W=1024
    if image.shape[0]<H or image.shape[1]<W:
        # print(image.shape[0])
        # print(image.shape[1])
        image = cv2.resize(image,(W,H)) #先宽后高
        # print(image.shape[0])
        # print(image.shape[1])
        # raise ValueError("stop")
    cv2.imwrite(out_file,image)

def resize_pic(ori_file,out_file,W,H):#eg:W=1024,H=832
    image = cv2.imread(ori_file)  # numpy数组格式（H,W,C=3），通道顺序（B,G,R) H<W
    bigger_size = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
    image = image[0:bigger_size, 0:bigger_size]  # 切成正方形--无效操作
    image = cv2.resize(image,(W,H))
    cv2.imwrite(out_file, image)

######################################
#test
# path2 =osp.join(root_path,"Test") #原数据文件夹
# path2_out = osp.join(out_path,"test")
#train
# path2 =osp.join(root_path,"Train") #原数据文件夹
# path2_out = osp.join(out_path,"train")
#val
path2 =osp.join(root_path,"Validation") #原数据文件夹
path2_out = osp.join(out_path,"validation")


path2_list = sorted(glob.glob(osp.join(path2,"*")))

if not osp.exists(path2_out):
    os.system("mkdir "+path2_out)
if not osp.exists(osp.join(path2_out, "left")):
    os.system("mkdir " + osp.join(path2_out, "left"))
if not osp.exists(osp.join(path2_out, "right")):
    os.system("mkdir " + osp.join(path2_out, "right"))
#aftercut/test/left/1.png
for path3 in path2_list:
    print(path3)
    path_out = path3.split("_")[-1].split(".")[0]
    if path_out=='L':
        path_out = "left"
    elif path_out=='R':
        path_out = "right"
    else:
        print(path_out)
        raise ValueError("stop")

    basename_out = osp.basename(path3) #112_R.png
    basename_out = basename_out.split("_")[0]
    basename_out = (str)((int)(basename_out))
    print(basename_out)

    cut_pic(path3,osp.join(path2_out,path_out,basename_out+".png"),640,960)

    # #left
    # # path3_base = osp.basename(path3)
    # try:
    #     # cut_pic(osp.join(path3, "left.png"), osp.join(path2_out, "left", str(count) + ".png"), 1024, 832)
    #     resize_pic(osp.join(path3, "left.png"), osp.join(path2_out, "left", str(count) + ".png"), 256)
    #     resize_pic(osp.join(path3, "right.png"), osp.join(path2_out, "right", str(count) + ".png"), 256)
    #     # print("\r"+path3,end='') #只占一行
    #
    # except:
    #     print(path3)
    #     raise ValueError('cannot find')
print("done.")

