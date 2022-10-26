# Project for MASIC

## Palace Datasets
Palace: https://drive.google.com/file/d/1X6w7P8EEo7RBX7Ev9NTuYPnIxIoM8AJX/view?usp=sharing

## Install
```
#pytorch-cuda11:
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install -e . 
pip install opencv-contrib-python==3.4.2.17 
pip install kornia==0.5.0
pip install imageio
pip install range_coder
# https://pytorch.org/get-started/previous-versions/
```

## Code
In coremasic/, the defination is under mywork/MASIC.py, and the validation is under myscript/.

### train
```
eg:
# train udh(seperately, you can see readme in udh/udh/)
cd udh/udh
python train.py path/to/train/ path/to/valid/

# train the codec part
cd coremasic/mywork/
python newtrain_codec_real.py -d "/home/yangwenzhe/database/aftercut512"  --seed 0 --cuda 0 --patch-size 512 512 --test-patch-size 512 512 --batch-size 3 --test-batch-size 1  --save --lambda 0.001 -e 200

# train the cqe part
cd coremasic/mywork/
python newtrain_cqe_real.py -d "/home/yangwenzhe/database/Peking"  --seed 0 --cuda 1 --patch-size 512 896 --test-patch-size 1216 2176 --batch-size 2 --test-batch-size 1  --save --homopath "homo_peking.pth.tar" --lambda 0.005

```

### test
```
# final test
cd coremasic/mywork/
python test3_real.py -d "/home/yangwenzhe/database/aftercut512"  --seed 0  --patch-size 512 512 --batch-size 1 --test-batch-size 1 --cuda 1 --homopath "homo_best.pth.tar"

# test only before cqe
cd coremasic/mywork/
python test2_real.py -d "/home/yangwenzhe/database/aftercut512"  --seed 0  --patch-size 512 512 --batch-size 1 --test-batch-size 1 --cuda 1 --homopath "homo_best.pth.tar"

```

### PS.benchmark for codec
Execute the coremasic/mywork/MASIC_save_jg_codec.py.


