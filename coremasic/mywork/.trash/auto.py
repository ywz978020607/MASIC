import os

for ii in [0.005,0.01,0.025,0.0483,0.0932]:
    os.system("cp ~/database/models/MASIC/lambda_" + str(ii) + "/checkpoint_best_loss.pth.tar ./")
    os.system("""python test2_real.py -d "/home/yangwenzhe/database/aftercut512"  --seed 0 --cuda 1 --patch-size 512 512 --batch-size 1 --test-batch-size 1""")