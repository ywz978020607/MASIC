# 画图脚本-放大图-主观图
# python auto.py 4.5 1 "right" "sPekingmini3_4" 47.png "508,289,584,366"

from tkinter.tix import Tree
import cv2
import os,glob,sys
import numpy as np

out_path = "out_contrast"

beishu = 4.5
if len(sys.argv) > 1:
    beishu = (float)(sys.argv[1])

direction = 1
if len(sys.argv) > 2:
    direction = (int)(sys.argv[2])

folder = "right"
if len(sys.argv) > 3:
    folder = (sys.argv[3])

root_name = "aftercut512"
if len(sys.argv) > 4:
    root_name = (sys.argv[4])

database_dir = [root_name,root_name+'_Coarse_2',root_name+'_MASIC']
# database_dir = ['','aftercut512_Coarse_2','aftercut512_MASIC']

png_name = "0.png"
if len(sys.argv) > 5:
    png_name = sys.argv[5]

[x1,y1,x2,y2] = [292,330,347,385]
if len(sys.argv) > 6:
    lambda_list = sys.argv[6].split(",")
    x1 = (int)(lambda_list[0])
    y1 = (int)(lambda_list[1])
    x2 = (int)(lambda_list[2])
    y2 = (int)(lambda_list[3])

only_once = True
# x是宽坐标，y是高坐标
def enlarge(png_file,path,x1,y1,x2,y2,beishu,direction=3): #direction:0123:左上、右上、左下、右下
    global only_once

    img = cv2.imread(os.path.join(path,folder,png_file))
    # img.shape[0] img.shape[1] #先高后宽
    crop_img = img[y1:y2,x1:x2]
    #裁剪
    # gray_square = np.ones((u1,u2),dtype=np.uint8)
    height,width=crop_img.shape[:2]
    crop_enlarge=cv2.resize(crop_img,(int(beishu*width),int(beishu*height)),interpolation=cv2.INTER_CUBIC)
    #显示放大图

    #显示待放大
    cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
    if direction == 0:
        left_up_corner = (0,0)
        right_down_corner = (int(beishu*width),int(beishu*height))
    elif direction == 1:
        left_up_corner = (img.shape[1]-int(beishu*width),0)
        right_down_corner = (img.shape[1],int(beishu*height))
    elif direction == 2:
        left_up_corner = (0,img.shape[0]-int(beishu*height))
        right_down_corner = (int(beishu*width),img.shape[0])
    elif direction == 3:
        left_up_corner = (img.shape[1]-int(beishu*width),img.shape[0]-int(beishu*height))
        right_down_corner = (img.shape[1],img.shape[0])
    # print(left_up_corner)
    # print(right_down_corner)
    img[left_up_corner[1]:right_down_corner[1], left_up_corner[0]:right_down_corner[0]] = crop_enlarge
    cv2.rectangle(img, left_up_corner, right_down_corner, (0, 0,255), 2)

    out_file_path = os.path.join(out_path,folder,png_file.split(".")[0]+"_"+path+"."+png_file.split(".")[-1])
    print(out_file_path)

    cv2.imwrite(out_file_path, img)

    if only_once:
        cv2.imshow(out_file_path, img)
        cv2.waitKey(0)
        only_once=False


# if __name__=="__main__":
#     enlarge("1.png","",30,100,50,160,4.5)

for path in database_dir:
    enlarge(png_name,path,x1,y1,x2,y2,beishu,direction)


"""
Instereo2k:
#67 82
python auto.py 4.5 1 "right" "aftercut512" 8.png "120,163,187,245"
python auto.py 4.5 1 "left" "aftercut512" 8.png "135,167,202,249"

#46 67
python auto.py 4.5 1 "right" "aftercut512" 26.png "219,318,265,385"
python auto.py 4.5 1 "left" "aftercut512" 26.png "255,321,301,388"

故宫:
#56 66
python auto.py 4.5 1 "right" "sPekingmini3_4" 47.png "518,290,574,356"
##python auto.py 4.5 1 "left" "sPekingmini3_4" 47.png "774,285,830,351"

#49 83
python auto.py 4.5 1 "right" "sPekingmini3_4" 8.png "410,7,459,90"
python auto.py 4.5 1 "left" "sPekingmini3_4" 8.png "690,10,739,93"

#
python auto.py 4.5 1 "right" "sPekingmini3_4" 28.png "8,333,180,375"
python auto.py 4.5 1 "left" "sPekingmini3_4" 28.png "274,329,446,371"

"""