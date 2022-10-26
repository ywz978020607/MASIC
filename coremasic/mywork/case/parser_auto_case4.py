# 自动测试并写入csv
# python parser_auto.py aftercut512 aftercut512 "512 512" "0.005,0.01,0.025,0.0483,0.0932" 0 yangwenzhe
# python parser_auto.py KITTI2 KITTI2 "320 1216" "0.005,0.01,0.025,0.0483,0.0932" 0 yangwenzhe nocqe
# ablation
# python parser_auto_case4.py --root-name "aftercut512" --database-name "aftercut512" --resolution "512 512" --lambda-list "0.005,0.01,0.025,0.0483" --ablation case4 --K 1

#自动解析

import os
import json 
import csv
import sys
from dict_import import auto_import

#默认参数
arg_dict = {
    "root_name": "Instereo2k", #model
    "database_name": "aftercut512",
    "resolution": "512 512",
    "lambda_list": "0.005,0.01,0.025,0.0483",
    "cuda_id": "0",
    "username": "ywz",
    "code_name": "test3_real.py",
    "ablation": "",
    "K": "5",
}
args = auto_import(arg_dict, sys.argv[1:])
# print(args["root_name"])
# for key in args:
#     expr = key + "= %s" %(args[key])
#     exec(expr)


# args["lambda_list"]
args["lambda_list"] = args["lambda_list"].split(",")
for ii in range(len(args["lambda_list"])):
    args["lambda_list"][ii] = (float)(args["lambda_list"][ii])
print(args["lambda_list"])

if args["ablation"]:
    print("ablation:",args["ablation"])
    args["code_name"] = args["code_name"].split(".py")[0] + "_" + args["ablation"] + ".py"
    print(args["code_name"])

csvfile = open(args["database_name"] +"_"+args["ablation"]+"_"+args["K"]+'.csv', 'w', newline='') #
writer = csv.writer(csvfile)

def dealOut(temp_out):
    temp_out = temp_out.replace("\t"," ")
    temp_out = temp_out.replace("\n"," ")
    temp_out = temp_out.split("Test epoch 0:")[-1]
    temp_list = temp_out.split("|")[1:]
    print(temp_list) #[loss,mse,psnr,bpp, bpp1,bpp2,ssim,ssim1,ssim2,psnr1,psnr2]
    res_json = {}
    for ii in range(len(temp_list)):
        temp_dict = temp_list[ii].strip().split(":")
        temp_dict[0] = temp_dict[0].upper()
        if "PSNR" in temp_dict[0] and "PSNR1" not in temp_dict[0] and "PSNR2" not in temp_dict[0]:
            temp_dict[0] = "PSNR" #处理PSNR -- PSNR(dB)等格式不统一问题
        elif "BPP" in temp_dict[0].upper() and "BPP1" not in temp_dict[0].upper() and "BPP2" not in temp_dict[0].upper():
            temp_dict[0] = "BPP"
        elif "LPIPS" in temp_dict[0].upper() and "LPIPS1" not in temp_dict[0].upper() and "LPIPS2" not in temp_dict[0].upper():
            temp_dict[0] = "LPIPS"

        res_json[temp_dict[0].strip()] = float(temp_dict[1].strip())
    print(res_json)
    return res_json

######################
for ii in args["lambda_list"]:
    os.system("cp ~/database/models/MASIC/"+args["root_name"]+"/"+"lambda_"+str(ii)+"/* ./")
    if args["ablation"]:
        os.system("cp ~/database/models/MASIC_{}_{}/".format(args["ablation"],args["K"])+args["root_name"]+"/"+"lambda_"+str(ii)+"/* ./")
    temp_cmd = """python -W ignore {} -d "/home/{}/database/""".format(args["code_name"], args["username"])+ args["database_name"] + """\"  --seed 0 --cuda """ + str(args["cuda_id"]) + """ --patch-size """+args["resolution"]+""" --batch-size 1 --test-batch-size 1 --K {}""".format(args["K"])
    #重定向
    resout = os.popen(temp_cmd)
    temp_cmd_out = resout.read()
    resout.close()
    #解析
    print(temp_cmd_out)
    temp_res_json = dealOut(temp_cmd_out)
    #写表
    writeRow = [temp_res_json['PSNR'],temp_res_json['BPP']," ",temp_res_json['MS-SSIM'],temp_res_json['BPP']," ",temp_res_json['LPIPS'],temp_res_json['BPP']," "
    ,temp_res_json["PSNR1"],temp_res_json["PSNR2"],temp_res_json["MS-SSIM1"],temp_res_json["MS-SSIM2"],temp_res_json["BPP1"],temp_res_json["BPP2"],temp_res_json["LPIPS1"],temp_res_json["LPIPS2"]]
    writer.writerow(writeRow) #final close remember!

csvfile.close()


