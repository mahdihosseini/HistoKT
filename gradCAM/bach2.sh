#!/usr/bin/env bash

ROOT=$PWD

# python3 main.py --image_path=$ROOT/examples/001.png_crop_16.png \
# --model_path=$ROOT/examples/best_trial_2_date_2021-07-13-19-43-53.pth.tar \
# --dataset_name=ADP --aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=/ssd2/HistoKT/results/ImageNet_weights/new-ImageNet-pretraining-checkpoint/None/BACH_transformed/best_trial_1_date_2021-07-16-15-52-18.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH_baseline_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=/ssd2/HistoKT/results/post_training_without_color_aug_ImageNet_norm_ImageNet/CRC_transformed_norm_ImageNet_color_aug_None_ImageNet/BACH_transformed/AdamP/checkpoint/deep_tuning/lr-0.00005/best_trial_1_date_2021-07-18-09-46-33.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH_CRCpost_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=/ssd2/HistoKT/results/ADPL3_ImageNet_norm_ImageNet/ADP-Release1_norm_ImageNet_color_aug_None_ADPL3/BACH_transformed/AdamP/checkpoint/deep_tuning/lr-0.00005/best_trial_1_date_2021-07-19-00-09-10.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH_ADPpost_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True
