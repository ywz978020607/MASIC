import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool

pic_num_list = list(range(0,50))
root_x1 = '/home/dyf/database/Peking/test/left/'
root_x2 = '/home/dyf/database/Peking/test/right/'
root_x1warp_udh = '/home/dyf/dev/masic/dyf/attn_0.005/out_pic2/left/'
root_x1warp_opencv = '/home/dyf/dev/masic/dyf/attn_0.005/out_pic1/left/'
root_save = '/home/dev/plot/img_contrast/'

def cv2plot(root, i):
    img = cv2.imread(root+str(i)+'.png')
    b,g,r = cv2.split(img)
    img_c = cv2.merge([r,g,b])
    return img_c

def show(pic_num):
    x1 = cv2plot(root_x1, pic_num)
    x2 = cv2plot(root_x2, pic_num)
    x1warp_udh = cv2plot(root_x1warp_udh, pic_num)
    x1warp_opencv = cv2plot(root_x1warp_opencv, pic_num)
    plt.figure(figsize=(32,18))
    plt.subplot(2,2,1)
    plt.imshow(x1)
    plt.title('x1')
    plt.axis('off')
    plt.subplot(2,2,2)
    plt.imshow(x2)
    plt.title('x2')
    plt.axis('off')
    plt.subplot(2,2,3)
    plt.imshow(x1warp_opencv)
    plt.title('x1warp_opencv')
    plt.axis('off')
    plt.subplot(2,2,4)
    plt.imshow(x1warp_udh)
    plt.title('x1warp_udh')
    plt.axis('off')
    plt.savefig(str(pic_num)+'.png')
    print(str(pic_num)+' saved!')


if __name__ == '__main__':
    P = Pool(processes=10)
    P.map(func=show, iterable=pic_num_list)