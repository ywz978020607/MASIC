from logging import root
import os
import sys

# python auto_train_case4.py --root-name "aftercut512" --database-name "aftercut512" --resolution "512 512" --lambda-list "0.005,0.01,0.025,0.0483" --ablation case4  --train-batch-size 5 --epoch 20 --K 1
from dict_import import auto_import

#默认参数
arg_dict = {
    "root_name": "aftercut512", #model
    "database_name": "aftercut512",
    "resolution": "512 512",
    "lambda_list": "0.005,0.01,0.025,0.0483",
    "cuda_id": "0",
    "ablation": "",
    "K": "5",
    "train_batch_size": "5",
    "train_patch_size": "",
    "epoch": "20",
}
args = auto_import(arg_dict, sys.argv[1:])


# lambda_list
args["lambda_list"] = args["lambda_list"].split(",")
for ii in range(len(args["lambda_list"])):
    args["lambda_list"][ii] = (float)(args["lambda_list"][ii])
print(args["lambda_list"])

if not args["train_patch_size"]:
    args["train_patch_size"] = args["resolution"]
print("train patch size", args["train_patch_size"])

for ii in args["lambda_list"]:
    os.system("cp ~/database/models/MASIC/"+args["root_name"]+"/"+"lambda_"+str(ii)+"/* ./")
    
    os.system("""python newtrain_codec_real_case4.py -d "/home/ywz/database/"""+ args["database_name"] + """\"  --seed 0 --cuda """+str(args["cuda_id"])+""" --patch-size """+ args["train_patch_size"] +""" --test-patch-size """+ args["resolution"] +""" --batch-size """+ args["train_batch_size"] +""" --test-batch-size 1  --save --lambda """+ str(ii) +""" -e {}""".format(args["epoch"]) + " --K {} ".format(args["K"]) )
    # os.system("""python newtrain_cqe_real.py -d "/home/ywz/database/"""+ database_name + """\"  --seed 0 --cuda """+str(cuda_id)+""" --patch-size """+ train_patch_size +""" --test-patch-size """+ args["resolution"] +""" --batch-size 1 --test-batch-size 1  --save --lambda """+ str(ii) +""" -e 25""" + homopath)

    os.system("""mkdir ~/database/models/MASIC_case4_{}/""".format(args["K"])+ args["database_name"] +"""/lambda_""" + str(ii) + """/""")
    #save
    os.system("""cp checkpoint_best_loss.pth.tar ~/database/models/MASIC_case4_{}/""".format(args["K"])+ args["database_name"] +"""/lambda_""" + str(ii) + """/""")
    # os.system("""cp second_checkpoint_best_loss.pth.tar ~/database/models/MASIC/"""+ database_name +"""/lambda_""" + str(ii) + """/""")
