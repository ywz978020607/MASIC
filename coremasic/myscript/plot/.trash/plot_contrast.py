import xlrd
import sys
# import torch
import matplotlib
import math
import matplotlib.pyplot as plt
from operator import itemgetter

plt.rc('font',family='Times New Roman')
# matplotlib.use('Agg')  # tmux专用  不报错
fontsize=15

## 统一定义
plt.grid()
plt.xlabel("Bitrate",fontsize=fontsize)
plt.ylabel("PSNR (dB)",fontsize=fontsize)
# plt.title('PSNR (Instereo2k) ',fontsize=fontsize)
plt.title('PSNR (KITTI2) ',fontsize=fontsize)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

color = ['#f94144','#ffb703','#90be6d','#219ebc','purple','#3d5a80','grey','#e29578','#028f1e','#cb0162','#370617','#ffc6ff']
signal = ['^','d','p','s','H','v','X','o','o','*'    ,'^','d','p','s','H','v','X','o','o']

## datasheet 读画
wb = xlrd.open_workbook(r'data.xlsx')
#sheet1索引从0开始，得到sheet1表的句柄
sheet1 = wb.sheet_by_index(0)
rowNum = sheet1.nrows
colNum = sheet1.ncols
x1 = []
y1 = []
for ii in range(colNum//2):
    y1 = sheet1.col_values(ii*2)
    x1 = sheet1.col_values(ii*2+1)
    while len(x1)>1 and x1[-1]=='':
        x1.pop()
        y1.pop()

    #plot
    x1, y1 = zip(*sorted(zip(x1[1:], y1[1:]), key=itemgetter(0)))
    # if ii==colNum//2-1:
    #     plt.plot(x1, y1, color=color[ii], marker=signal[ii], markersize='5', lw=1.5,linestyle=':')
    # else:
    if 1:
        plt.plot(x1, y1, color=color[ii], marker=signal[ii], markersize='5', lw=1.5)

legend = sheet1.row_values(0)[0::2] #只留奇数位置
plt.legend(legend,loc='lower right',fontsize=10)
## save
plt.xlim(0, 0.8)
# plt.savefig('contrast_ssim.png')
# plt.savefig('contrast_ssim_db.png')
plt.savefig('contrast.png')
plt.show()

print("ok")