#!/usr/bin/env bash

ROOT=$PWD

# python3 main.py --image_path=$ROOT/examples/001.png_crop_16.png \
# --model_path=$ROOT/examples/best_trial_2_date_2021-07-13-19-43-53.pth.tar \
# --dataset_name=ADP --aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/PCam/ \
--model_path=/ssd2/HistoKT/results/ImageNet_weights/new-ImageNet-pretraining-checkpoint/None/PCam_transformed/best_trial_1_date_2021-07-16-15-56-24.pth.tar \
--dataset_name=PCam_transformed \
--output_path=output/PCam_baseline_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/PCam/ \
--model_path=/ssd2/HistoKT/results/post_training_without_color_aug_ImageNet_norm_ImageNet/CRC_transformed_norm_ImageNet_color_aug_None_ImageNet/PCam_transformed/AdamP/checkpoint/deep_tuning/lr-0.001/best_trial_1_date_2021-07-18-14-37-31.pth.tar \
--dataset_name=PCam_transformed \
--output_path=output/PCam_CRCpost_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/PCam/ \
--model_path=/ssd2/HistoKT/results/ADPL3_ImageNet_norm_ImageNet/ADP-Release1_norm_ImageNet_color_aug_None_ADPL3/PCam_transformed/AdamP/checkpoint/deep_tuning/lr-0.001/best_trial_0_date_2021-07-19-00-09-10.pth.tar \
--dataset_name=PCam_transformed \
--output_path=output/PCam_ADPpost_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True
