#!/usr/bin/env bash

ROOT=$PWD

# python3 main.py --image_path=$ROOT/examples/001.png_crop_16.png \
# --model_path=$ROOT/examples/best_trial_2_date_2021-07-13-19-43-53.pth.tar \
# --dataset_name=ADP --aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/CRC/ \
--model_path=/ssd2/HistoKT/results/ImageNet_weights/new-ImageNet-pretraining-checkpoint/None/CRC_transformed/best_trial_1_date_2021-07-16-15-52-18.pth.tar \
--dataset_name=CRC_transformed \
--output_path=output/CRC_baseline_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/CRC/ \
--model_path=/ssd2/HistoKT/results/ADPL3_ImageNet_norm_ImageNet/ADP-Release1_norm_ImageNet_color_aug_None_ADPL3/CRC_transformed/AdamP/checkpoint/deep_tuning/lr-0.0005/best_trial_1_date_2021-07-19-07-02-36.pth.tar \
--dataset_name=CRC_transformed \
--output_path=output/CRC_ADPpost_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True
