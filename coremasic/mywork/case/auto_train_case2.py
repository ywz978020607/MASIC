from logging import root
import os
import sys

# python auto_train.py sPekingmini3_6 sPekingmini3_6 "640 704" "0.001,0.005,0.01,0.025,0.0483,0.0932" 0 "homo_best_sPekingmini3_6.pth.tar"
# python auto_train.py sPeking_8 sPeking_8 "1216 1664" "0.001,0.005,0.01,0.025,0.0483,0.0932" 0 "homo_best_sPeking_8.pth.tar" 5 "512 512"
#model-path
root_name = "Instereo2k"
if len(sys.argv) > 1:
    root_name = str(sys.argv[1])
print("root_name:",root_name)

database_name = "aftercut512"
if len(sys.argv) > 2:
    database_name = str(sys.argv[2])
print("database_name:",database_name)

# 分辨率 
# resolution = "320 1216" #320 1216 , 512 512 512 448 512 384
resolution = "512 512"
if len(sys.argv) > 3:
    resolution = str(sys.argv[3])
print("resolution:",resolution)

# lambda_list
lambda_list = [0.005,0.01,0.025,0.0483,0.0932]
if len(sys.argv) > 4:
    lambda_list = sys.argv[4].split(",")
    for ii in range(len(lambda_list)):
        lambda_list[ii] = (float)(lambda_list[ii])
print(lambda_list)

# 显卡号
cuda_id = 0
if len(sys.argv) > 5:
    cuda_id = int(sys.argv[5])
print("cuda_id:",str(cuda_id))

homopath = ""
if len(sys.argv) > 6:
    homopath = str(sys.argv[6])
    print("homopath:",str(homopath))
if homopath:
    homopath = " --homopath " + homopath

train_batch_size = 5
if len(sys.argv) > 7:
    train_batch_size = str(sys.argv[7])
    print("train batch size", train_batch_size)
train_patch_size = resolution
if len(sys.argv) > 8:
    train_patch_size = str(sys.argv[8])
    print("train patch size", train_patch_size)

for ii in lambda_list:
    os.system("cp ~/database/models/MASIC/"+database_name+"/"+"lambda_"+str(ii)+"/* ./")
    # os.system("cp ~/database/models/MASIC/"+database_name+"/"+"lambda_" + str(ii) + "/checkpoint_best_loss.pth.tar ./")

    # os.system("""python newtrain_codec_real.py -d "/home/ywz/database/"""+ database_name + """\"  --seed 0 --cuda """+str(cuda_id)+""" --patch-size """+ train_patch_size +""" --test-patch-size """+ resolution +""" --batch-size """+ train_batch_size +""" --test-batch-size 1  --save --lambda """+ str(ii) +""" -e 100""" + homopath)
    os.system("""python newtrain_cqe_real_case2.py -d "/home/ywz/database/"""+ database_name + """\"  --seed 0 --cuda """+str(cuda_id)+""" --patch-size """+ train_patch_size +""" --test-patch-size """+ resolution +""" --batch-size 1 --test-batch-size 1  --save --lambda """+ str(ii) +""" -e 5""" + homopath)

    os.system("""mkdir ~/database/models/MASIC_case2/"""+ database_name +"""/lambda_""" + str(ii) + """/""")
    #save
    # os.system("""cp checkpoint_best_loss.pth.tar ~/database/models/MASIC/"""+ database_name +"""/lambda_""" + str(ii) + """/""")
    os.system("""cp second_checkpoint_best_loss.pth.tar ~/database/models/MASIC_case2/"""+ database_name +"""/lambda_""" + str(ii) + """/""")
