import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

if opt.train:
    os.system("python train.py \
        --data_root ../data/enlightengan \
        --name enlighten01 \
        --patchD \
        --patch_vgg \
        --n_patchD 5 \
        --n_layers_D 5 \
        --n_layers_patchD 4 \
        --fineSize 256 \
        --patchSize 32 \
        --batch_size 24 \
        --self_attention \
        --use_norm \
        --use_wgan 0 \
        --use_ragan \
        --only_lsgan \
        --mult_attention \
        --vgg_choose conv5_1 \
        --residual_skip 1.0 \
        --vgg_choose conv5_1 \
        --display_port=" + opt.port)
elif opt.predict:
    os.system("python predict.py \
        --data_root ../data/enlightengan \
        --name enlighten01 \
        --use_norm \
        --use_wgan 0 \
        --self_attention \
        --mult_attention")
