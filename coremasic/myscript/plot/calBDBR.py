# 0 1, 2 1, 3,1 #instereo2k
# 0 5,
# 0 9,
from bjontegaard_metric import *
# from pandas.tests.extension.numpy_.test_numpy_nested import np
from operator import itemgetter
import math
import xlrd
import sys

item_size = 4 #psnr,bpp,ssim,lpips顺序
y_bias = 0 
if len(sys.argv) > 1:
    y_bias = int(sys.argv[1])
y_name = ['PSNR','BPP-no-use','SSIM','LPIPS'][y_bias]
print(y_name)

sheet_id = 0
if len(sys.argv) > 2:
    sheet_id = int(sys.argv[2])
# name_index = 0
# database_name = ['Instereo2k', 'KITTI2', 'sPeking'][name_index]


## datasheet 读画
wb = xlrd.open_workbook(r'data.xlsx')
#sheet1索引从0开始，得到sheet1表的句柄
sheet1 = wb.sheet_by_index(sheet_id)
print("sheetname:",wb.sheet_names()[sheet_id])
rowNum = sheet1.nrows
colNum = sheet1.ncols

#基准
x_mark = []
y_mark = []
for ii in range(colNum//item_size):
    name = sheet1.col_values(ii*item_size)[0]
    # if name.strip() == 'Hyper': #基准
    if "Ball" in name.strip() or 'Hyper' in name.strip():
        y_mark = sheet1.col_values(ii*item_size+y_bias) #y-val
        x_mark = sheet1.col_values(ii*item_size+1) #bpp
        while len(x_mark)>1 and x_mark[-1]=='':
            x_mark.pop()
            y_mark.pop()
        #plot
        x_mark, y_mark = zip(*sorted(zip(x_mark[1:], y_mark[1:]), key=itemgetter(0)))

        break

# ###
x1 = []
y1 = []
for ii in range(colNum//item_size):
    name = sheet1.col_values(ii*item_size)[0]
    y1 = sheet1.col_values(ii*item_size+y_bias) #y-val
    x1 = sheet1.col_values(ii*item_size+1) #bpp
    while len(x1)>1 and x1[-1]=='':
        x1.pop()
        y1.pop()

    #plot
    x1, y1 = zip(*sorted(zip(x1[1:], y1[1:]), key=itemgetter(0)))

    print(name)
    print('BD-PSNR: ', BD_PSNR(x_mark, y_mark, x1, y1))
    print('BD-RATE: ', BD_RATE(x_mark, y_mark, x1, y1))
