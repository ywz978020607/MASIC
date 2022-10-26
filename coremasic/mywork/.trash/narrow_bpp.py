# narrow bpp $> python3 narrow_bpp.py
# 为了尽可能兼容不同ft的项目 采用顶层训练调整 os管道重定向调用训练 需要每个批次训练使其基本稳定收敛作为一个circle_epoch

# 需要适当设定circle_epoch
# notice: 提前放入起始lmbda对应的稳定的模型

import os
import json 
import csv

####################################
circle_epoch = 10 #起始每轮的epoch--后续可以分析上一epoch与target_bpp的数量，适当调整
circle_batch_size = 3 #ft时的batch-size
lmda = 0.005 #起始
lmda_step = lmda   #建议初始step为已经能够明确估计的lambda两点之差
last_lmda_step_flag = None #记忆上一次增还是减 只有到拐点才学习率减半
#bppft版
target_bpp = 0.148 #最终目标
target_endure = target_bpp*0.001 #千分之一误差
####################################
cuda_id = 0 #显卡号


# 数据集路径对应的名字 
# database_name = "aftercut512"
database_name = "sPekingmini3_4"
# database_name = "KITTI2"

# 分辨率 
resolution = "640 832" #320 1216 , 512 512 512 448 512 384


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
        res_json[temp_dict[0].strip()] = float(temp_dict[1].strip())
    print(res_json)
    return res_json

######################
while 1:
    #---------测试------------
    print("-------test once-------")
    temp_cmd = """python -W ignore test3_real.py -d "/home/ywz/database/"""+ database_name + """\"  --seed 0 --cuda """+str(cuda_id)+""" --patch-size """+resolution+""" --batch-size 1 --test-batch-size 1"""
    #重定向
    resout = os.popen(temp_cmd)
    temp_cmd_out = resout.read()
    resout.close()
    #解析
    print(temp_cmd_out)
    temp_res_json = dealOut(temp_cmd_out)
    # #写表
    # writeRow = [temp_res_json['PSNR'],temp_res_json['BPP']," ",temp_res_json['MS-SSIM'],temp_res_json['BPP']," ",temp_res_json["PSNR1"],\
    #     temp_res_json["PSNR2"],temp_res_json["MS-SSIM1"],temp_res_json["MS-SSIM2"],temp_res_json["BPP1"],temp_res_json["BPP2"]]
    #---------调整------------
    if abs(temp_res_json['BPP']-target_bpp)<target_endure:
        print("Done, target lambda is {}, please save the model in time!".format(str(lmda)))
        break
    if temp_res_json['BPP']<target_bpp:
        lmda += lmda_step
        temp_lmda_step_flag = 1 #+=step
    else:
        lmda -= lmda_step
        temp_lmda_step_flag = -1 #-=step
    print("next lambda:",str(lmda),"temp step:",str(lmda_step))
    if lmda<0:
        print("next lambda<0 break.")
        break
    # 只有折半的时候才/2 降低学习率
    if last_lmda_step_flag!=None and last_lmda_step_flag!=temp_lmda_step_flag:
        lmda_step /= 2
    last_lmda_step_flag = temp_lmda_step_flag
    #---------训练------------
    train_cmd = """python newtrain_codec_real.py -d "/home/ywz/database/"""+ database_name + """\"  --seed 0 --cuda """+str(cuda_id)+""" --patch-size """+resolution+""" --test-patch-size """+resolution+""" --batch-size """+str(circle_batch_size)+""" --test-batch-size 1  --save --lambda """+str(lmda)+""" -e """+str(circle_epoch)
    # narrow-bpp时 最后再训CQE
    # train_cmd = """python newtrain6_real.py -d "/home/ywz/database/"""+ database_name + """\"  --seed 0 --cuda """+str(cuda_id)+""" --patch-size """+resolution+""" --batch-size """+str(circle_batch_size)+""" --test-batch-size 1  --save --lambda """+str(lmda)+""" -e """+str(circle_epoch)
    os.system(train_cmd)
    #------------------------------------------------









## debug -- no use
# temp_cmd = """python test3_real.py -d "/home/ywz/database/"""+ database_name + """\"  --seed 0 --cuda 3 --patch-size """+resolution+""" --batch-size 1 --test-batch-size 1"""
# #重定向
# resout = os.popen(temp_cmd)
# temp_cmd_out = resout.read()
# resout.close()
# #解析
# print(temp_cmd_out)
# temp_res_json = dealOut(temp_cmd_out)

# print(type(temp_res_json["PSNR1"]))
# print(temp_res_json['PSNR1'],",",temp_res_json['BPP1'])



