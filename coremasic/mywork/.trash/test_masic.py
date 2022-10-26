# 自动测试MASIC并写入csv

# 自动解析
# python test_masic.py -d "KITTI2" -m "KITTI2" --lambda 0.001 0.005 0.01 0.025 0.0483 -r "320 1216" --cuda 3

import os
import json
import csv
import argparse
import sys

# cuda_id = 0

# # 模型路径对应的名字
# root_name = "Instereo2k"
# # root_name = "KITTI2"

# # 数据集路径对应的名字
# database_name = "aftercut512"
# # database_name = "aftercut512_1"
# # database_name = "KITTI2"

# # 分辨率
# resolution = "512 512"  # 320 1216 , 512 512 512 448 512 384


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Example test script')
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        help='Testing dataset'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help='model root'
    )
    parser.add_argument(
        '-r',
        '--resolution',
        type=str,
        default="256 256",
        help='resolution of image'
    )
    parser.add_argument(
        '--lambda',
        dest='lmbda',
        type=float,
        nargs='+',
        help='lambda'
    )
    parser.add_argument(
        '--cuda',
        type=int,
        default=0,
        help='cuda number'
    )
    args = parser.parse_args(argv)
    return args





def dealOut(temp_out):
    temp_out = temp_out.replace("\t", " ")
    temp_out = temp_out.replace("\n", " ")
    temp_out = temp_out.split("Test epoch 0:")[-1]
    temp_list = temp_out.split("|")[1:]
    # [loss,mse,psnr,bpp, bpp1,bpp2,ssim,ssim1,ssim2,psnr1,psnr2]
    print(temp_list)
    res_json = {}
    for ii in range(len(temp_list)):
        temp_dict = temp_list[ii].strip().split(":")
        temp_dict[0] = temp_dict[0].upper()
        if "PSNR" in temp_dict[0] and "PSNR1" not in temp_dict[0] and "PSNR2" not in temp_dict[0]:
            temp_dict[0] = "PSNR"  # 处理PSNR -- PSNR(dB)等格式不统一问题
        elif "BPP" in temp_dict[0].upper() and "BPP1" not in temp_dict[0].upper() and "BPP2" not in temp_dict[0].upper():
            temp_dict[0] = "BPP"
        res_json[temp_dict[0].strip()] = float(temp_dict[1].strip())
    print(res_json)
    return res_json

def main(argv):
    args = parse_args(argv)
    root_name = args.model
    resolution = args.resolution
    database_name = args.dataset
    cuda_id = args.cuda
    lambda_list = args.lmbda

    csvfile = open(database_name + '.csv', 'w', newline='')
    writer = csv.writer(csvfile)
######################
    for ii in lambda_list:
        os.system("cp ~/database/models/MASIC/"+root_name+"/" +
                "lambda_"+str(ii)+"/checkpoint_best_loss.pth.tar ./")
        os.system("cp ~/database/models/MASIC/"+root_name+"/" +
                "lambda_"+str(ii)+"/second_checkpoint_best_loss.pth.tar ./")


        temp_cmd = """python -W ignore test3_real.py -d "/home/yangwenzhe/database/""" + database_name + \
            """\"  --seed 0 --cuda """ + \
            str(cuda_id) + """ --patch-size """+resolution + \
            """ --batch-size 1 --test-batch-size 1"""
        # 重定向
        resout = os.popen(temp_cmd)
        temp_cmd_out = resout.read()
        resout.close()
        # 解析
        print(temp_cmd_out)
        temp_res_json = dealOut(temp_cmd_out)
        # 写表
        writeRow = [temp_res_json['PSNR'], temp_res_json['BPP'], " ", temp_res_json['MS-SSIM'], temp_res_json['BPP'], " ", temp_res_json["PSNR1"],
                    temp_res_json["PSNR2"], temp_res_json["MS-SSIM1"], temp_res_json["MS-SSIM2"], temp_res_json["BPP1"], temp_res_json["BPP2"]]
        writer.writerow(writeRow)  # final close remember!

    csvfile.close()

if __name__ == '__main__':
    main(sys.argv[1:])