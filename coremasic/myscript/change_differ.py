#调整数据集的视差
# python change_differ.py 1 0 /home/ywz/database/aftercut512
"""
dtn="sPekingmini3_6"
python change_differ.py 1 0 /home/ywz/database/$dtn
python change_differ.py 1 1 /home/ywz/database/$dtn
python change_differ.py 2 0 /home/ywz/database/$dtn
python change_differ.py 2 1 /home/ywz/database/$dtn
python change_differ.py 3 0 /home/ywz/database/$dtn
python change_differ.py 3 1 /home/ywz/database/$dtn
"""


import os,glob
import os.path as osp
import cv2
import sys

differ = 1 #differ -> 64*differ cut-off
if len(sys.argv)>1:
    differ = int(sys.argv[1])
    print("differ", str(differ))

mode = "test" #"test" "train" "validation"
if len(sys.argv)>2:
    mode = ["test", "train", "validation"][int(sys.argv[2])]
    print("mode", mode)

#根目录 可以是移动硬盘，也可以是本地
# root_path = "/home/ywz/database/flickr"
# root_path = "/home/yangwenzhe/database/aftercut512"
root_path = "/home/ywz/database/aftercut512"
# root_path = "/home/sharklet/database"
if len(sys.argv)>3:
    root_path = str(sys.argv[3])
print("root_path:", root_path)



#out
out_path = root_path+"_"+str(differ) #去掉differ%
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

def change_imgdiffer(ori_file,out_file,scale,mode="left"):
    image = cv2.imread(ori_file)  # numpy数组格式（H,W,C=3），通道顺序（B,G,R) H<W
    H = image.shape[0]
    W = image.shape[1]
    # H_little = (int)(H * scale)
    # if mode=="left":
    #     image = image[0:H_little, 0:W]  # H=832,W=1024
    # elif mode=="right":
    #     image = image[(H-H_little):H, 0:W]  # H=832,W=1024
    # else:
    #     raise ValueError("stop--not left or right in change_differ.")
    W_little = (int)(W - scale*64) #64/unit
    if mode == "left":
        image = image[0:H, 0:W_little]  # H=832,W=1024
    elif mode == "right":
        image = image[0:H, (W-W_little):W]  # H=832,W=1024
    else:
        raise ValueError("stop--not left or right in change_differ.")
    # #拉伸回原大小
    # image = cv2.resize(image, (W, H))  # 先宽后高
    cv2.imwrite(out_file, image)

######################################
#test
path2 =osp.join(root_path,mode) #原数据文件夹
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
    # path_out = path3.split("_")[-1].split(".")[0]

    basename_out = osp.basename(path3) #1.png
    print(basename_out)

    # change_imgdiffer(path3, osp.join(path2_out, "left", basename_out), (1 - differ / 100),"left")
    # change_imgdiffer(path3, osp.join(path2_out, "right", basename_out),  (1 - differ / 100),"right")
    change_imgdiffer(path3, osp.join(path2_out, "left", basename_out), differ, "left")
    change_imgdiffer(path3.replace("/left/","/right/"), osp.join(path2_out, "right", basename_out), differ, "right")

print("done")







