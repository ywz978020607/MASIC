# 正常顺序-溶解实验用

import xlrd #1.2.0
import sys
# import torch
import matplotlib
import math
import matplotlib.pyplot as plt
from operator import itemgetter


item_size = 4 #psnr,bpp,ssim,lpips顺序
y_bias = 0 
if len(sys.argv) > 1:
    y_bias = int(sys.argv[1])
y_name = ['PSNR (dB)','BPP-no-use','MS-SSIM (dB)','LPIPS'][y_bias]
print(y_name)

name_index = 0
if len(sys.argv) > 2:
    name_index = int(sys.argv[2])
database_name = ['Instereo2k', 'KITTI', 'Palace'][name_index]

sheet_index = 0
if len(sys.argv) > 3:
    sheet_index = int(sys.argv[3])


plt.rc('font',family='Times New Roman')
# matplotlib.use('Agg')  # tmux专用  不报错
fontsize=15

## 统一定义
plt.grid()
plt.xlabel("Bitrate (bpp)",fontsize=fontsize)
plt.ylabel(y_name,fontsize=fontsize)
# plt.title('PSNR (Instereo2k) ',fontsize=fontsize)
plt.title(str(database_name),fontsize=fontsize)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

color = ['#f94144','#ffb703','#90be6d','#219ebc','purple','#3d5a80','grey','#e29578','#028f1e','#cb0162','#370617','#ffc6ff','#45c67b']
signal = ['^','d','p','s','H','v','X','o','o','*'    ,'^','d','p','s','H','v','X','o','o','o']

## datasheet 读画
wb = xlrd.open_workbook(r'data.xlsx')
#sheet1索引从0开始，得到sheet1表的句柄
sheet1 = wb.sheet_by_index(sheet_index)
if sheet_index != 0:
    database_name = wb.sheet_names()[sheet_index]
    print("database:",database_name)
    plt.title(str(database_name).replace("_","/").replace("ab",""),fontsize=fontsize)

rowNum = sheet1.nrows
colNum = sheet1.ncols
x1 = []
y1 = []
plot_list = []
for ii in range(colNum//item_size):
    y1 = sheet1.col_values(ii*item_size+y_bias) #y-val
    x1 = sheet1.col_values(ii*item_size+1) #bpp
    while len(x1)>1 and x1[-1]=='':
        x1.pop()
        y1.pop()

    #plot
    x1, y1 = zip(*sorted(zip(x1[1:], y1[1:]), key=itemgetter(0)))
    # if ii==colNum//2-1:
    #     plt.plot(x1, y1, color=color[ii], marker=signal[ii], markersize='5', lw=1.5,linestyle=':')
    # else:
    if y_bias == 2:
        y1 = list(y1)
        # SSIM -> dB
        for y_idx in range(len(y1)):
            y1[y_idx] = 10*math.log10(1/(1-(float)(y1[y_idx])))
    plot_list.append([x1[:],y1[:],color[ii],signal[ii],'5',1.5])
    
    plt.plot(x1, y1, color=color[ii], marker=signal[ii], markersize='5', lw=1.5)

legend_list = sheet1.row_values(0)[0::item_size]
legend = legend_list #只留第一个位置
plt.legend(legend,loc='lower right',fontsize=10)
## save
# plt.xlim(0, 0.8)
# plt.savefig('contrast_ssim.png')
# plt.savefig('contrast_ssim_db.png')
plt.savefig('contrast.png')
plt.show()

print("ok")