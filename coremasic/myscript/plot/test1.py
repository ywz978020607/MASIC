# pip install xlrd
# pip install pyexcel-xls -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
import xlrd
wb = xlrd.open_workbook(r'data.xlsx')
#获取所有sheet的名字
print(wb.sheet_names())
# sheet2 = wb.sheet_names()[0]
#sheet1索引从0开始，得到sheet1表的句柄
sheet1 = wb.sheet_by_index(0)
rowNum = sheet1.nrows
colNum = sheet1.ncols
#s = sheet1.cell(1,0).value.encode('utf-8')
s = sheet1.cell(1,0).value #(row,col)
#获取某一个位置的数据
# 1 ctype : 0 empty,1 string, 2 number, 3 date, 4 boolean, 5 error
print(sheet1.cell(1,2).ctype)
print(s)
#print(s.decode('utf-8'))
#获取整行和整列的数据
#第二行数据
row2 = sheet1.row_values(1)
#第二列数据
cols2 = sheet1.col_values(2)

