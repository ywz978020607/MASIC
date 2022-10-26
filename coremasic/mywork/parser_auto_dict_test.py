# 自动测试并写入csv
# python parser_auto.py aftercut512 aftercut512 "512 512" "0.005,0.01,0.025,0.0483,0.0932" 0 yangwenzhe
# python parser_auto.py KITTI2 KITTI2 "320 1216" "0.005,0.01,0.025,0.0483,0.0932" 0 yangwenzhe nocqe
# ablation
# python parser_auto.py KITTI2 KITTI2 "320 1216" "0.005,0.01,0.025,0.0483,0.0932" 0 yangwenzhe cqe "case1"
# python parser_auto.py KITTI2 KITTI2 "320 1216" "0.005,0.01,0.025,0.0483,0.0932" 0 yangwenzhe cqe "case2"

#自动解析

import os
import json 
import csv
import sys
from dict_import import auto_import

arg_dict = {
    #--root-name
    "root_name": "Instereo2k" 
}
args = auto_import(arg_dict, sys.argv[1:])
print(args["root_name"])

# # 模型路径对应的名字
# root_name = "Instereo2k"
# # root_name = "KITTI2"
# if len(sys.argv) > 1:
#     root_name = str(sys.argv[1])
# print("root_name:",root_name)


# # 数据集路径对应的名字 
# database_name = "aftercut512"
# # database_name = "aftercut512_1"
# # database_name = "KITTI2"
# if len(sys.argv) > 2:
#     database_name = str(sys.argv[2])
# print("database_name:",database_name)


# # 分辨率 
# # resolution = "320 1216" #320 1216 , 512 512 512 448 512 384
# resolution = "512 512"
# if len(sys.argv) > 3:
#     resolution = str(sys.argv[3])
# print("resolution:",resolution)

# # lambda_list
# lambda_list = [0.005,0.01,0.025,0.0483]
# if len(sys.argv) > 4:
#     lambda_list = sys.argv[4].split(",")
#     for ii in range(len(lambda_list)):
#         lambda_list[ii] = (float)(lambda_list[ii])
# print(lambda_list)

# # 显卡号
# cuda_id = 0
# if len(sys.argv) > 5:
#     cuda_id = int(sys.argv[5])
# print("cuda_id:",str(cuda_id))

# # username
# username = "ywz"
# if len(sys.argv) > 6:
#     username = str(sys.argv[6])
# print("username:",str(username))

# # nocqe
# nocqe = False
# if len(sys.argv) > 7:
#     if str(sys.argv[7]) == 'nocqe':
#         nocqe = True
# code_name = 'test3_real.py'
# if nocqe:
#     code_name = 'test2_real.py'
# print(code_name)

# ablation = ""
# if len(sys.argv) > 8:
#     ablation = str(sys.argv[8])
#     print("ablation:",ablation)
#     code_name = code_name.split(".py")[0] + "_" + ablation + ".py"
#     print(code_name)

# csvfile = open(database_name +'.csv', 'w', newline='') #
# writer = csv.writer(csvfile)

# def dealOut(temp_out):
#     temp_out = temp_out.replace("\t"," ")
#     temp_out = temp_out.replace("\n"," ")
#     temp_out = temp_out.split("Test epoch 0:")[-1]
#     temp_list = temp_out.split("|")[1:]
#     print(temp_list) #[loss,mse,psnr,bpp, bpp1,bpp2,ssim,ssim1,ssim2,psnr1,psnr2]
#     res_json = {}
#     for ii in range(len(temp_list)):
#         temp_dict = temp_list[ii].strip().split(":")
#         temp_dict[0] = temp_dict[0].upper()
#         if "PSNR" in temp_dict[0] and "PSNR1" not in temp_dict[0] and "PSNR2" not in temp_dict[0]:
#             temp_dict[0] = "PSNR" #处理PSNR -- PSNR(dB)等格式不统一问题
#         elif "BPP" in temp_dict[0].upper() and "BPP1" not in temp_dict[0].upper() and "BPP2" not in temp_dict[0].upper():
#             temp_dict[0] = "BPP"
#         elif "LPIPS" in temp_dict[0].upper() and "LPIPS1" not in temp_dict[0].upper() and "LPIPS2" not in temp_dict[0].upper():
#             temp_dict[0] = "LPIPS"

#         res_json[temp_dict[0].strip()] = float(temp_dict[1].strip())
#     print(res_json)
#     return res_json

# ######################
# for ii in lambda_list:
#     os.system("cp ~/database/models/MASIC/"+root_name+"/"+"lambda_"+str(ii)+"/* ./")
#     if ablation:
#         os.system("cp ~/database/models/MASIC_{}/".format(ablation)+root_name+"/"+"lambda_"+str(ii)+"/* ./")
#     temp_cmd = """python -W ignore {} -d "/home/{}/database/""".format(code_name, username)+ database_name + """\"  --seed 0 --cuda """ + str(cuda_id) + """ --patch-size """+resolution+""" --batch-size 1 --test-batch-size 1"""
#     #重定向
#     resout = os.popen(temp_cmd)
#     temp_cmd_out = resout.read()
#     resout.close()
#     #解析
#     print(temp_cmd_out)
#     temp_res_json = dealOut(temp_cmd_out)
#     #写表
#     writeRow = [temp_res_json['PSNR'],temp_res_json['BPP']," ",temp_res_json['MS-SSIM'],temp_res_json['BPP']," ",temp_res_json['LPIPS'],temp_res_json['BPP']," "
#     ,temp_res_json["PSNR1"],temp_res_json["PSNR2"],temp_res_json["MS-SSIM1"],temp_res_json["MS-SSIM2"],temp_res_json["BPP1"],temp_res_json["BPP2"],temp_res_json["LPIPS1"],temp_res_json["LPIPS2"]]
#     writer.writerow(writeRow) #final close remember!

# csvfile.close()


