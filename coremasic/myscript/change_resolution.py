#调整数据集的分辨率
#change_imgreso

import os,glob
import os.path as osp
import cv2
import sys

min_reso = 64 #64整数倍

#根目录 可以是移动硬盘，也可以是本地
# root_path = "/home/ywz/database/flickr"
# root_path = "/home/yangwenzhe/database/aftercut512"
root_path = "/home/ywz/database/sPeking"
# root_path = "/home/sharklet/database"
if len(sys.argv)>1:
    root_path = str(sys.argv[1])
print("root_path:", root_path)

#out
out_path = "/home/ywz/database/sPekingmini2"
if len(sys.argv)>2:
    out_path = str(sys.argv[2])
print("out_path:", out_path)


#不存在则创建文件夹
if not osp.exists(out_path):
    os.system("mkdir "+out_path)
    print(out_path)
################################################################

def deal_min_reso(temp_reso):
    if temp_reso%min_reso==0:
        return temp_reso
    else:
        import math
        new_reso = min_reso * math.ceil(temp_reso/min_reso)
        return new_reso
def change_imgreso(ori_file,out_file):
    image = cv2.imread(ori_file)  # numpy数组格式（H,W,C=3），通道顺序（B,G,R) H<W
    H = image.shape[0]
    W = image.shape[1]
    new_W = deal_min_reso(W//2)
    new_H = deal_min_reso(H//2)

    image = cv2.resize(image, (new_W, new_H))  # 先宽后高
    cv2.imwrite(out_file, image)

######################################
#test
for mode in  ["test", "train", "validation"]:
    path2 =osp.join(root_path,mode) #原数据文件夹
    if not osp.exists(path2):
        print("not exists:" + path2)
        continue

    path2_out = osp.join(out_path,mode)

    path2_list = sorted(glob.glob(osp.join(path2,"left","*")))

    if not osp.exists(path2_out):
        os.system("mkdir "+path2_out)
    if not osp.exists(osp.join(path2_out, "left")):
        os.system("mkdir " + osp.join(path2_out, "left"))
    if not osp.exists(osp.join(path2_out, "right")):
        os.system("mkdir " + osp.join(path2_out, "right"))

    #aftercut/test/left/1.png
    for path3 in path2_list:
        print(path3)

        basename_out = osp.basename(path3) #1.png
        print(basename_out)

        change_imgreso(path3, osp.join(path2_out, "left", basename_out))
        change_imgreso(path3.replace("/left/","/right/"), osp.join(path2_out, "right", basename_out))

print("done")







