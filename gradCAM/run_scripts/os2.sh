#!/usr/bin/env bash

ROOT=$PWD

# python3 main.py --image_path=$ROOT/examples/001.png_crop_16.png \
# --model_path=$ROOT/examples/best_trial_2_date_2021-07-13-19-43-53.pth.tar \
# --dataset_name=ADP --aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/OS/ \
--model_path=/ssd2/HistoKT/results/ImageNet_weights/new-ImageNet-pretraining-checkpoint/None/OSDataset_transformed/best_trial_0_date_2021-07-16-15-48-24.pth.tar \
--dataset_name=OSDataset_transformed \
--output_path=output/OS_baseline_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/OS/ \
--model_path=/ssd2/HistoKT/results/post_training_without_color_aug_ImageNet_norm_ImageNet/CRC_transformed_norm_ImageNet_color_aug_None_ImageNet/OSDataset_transformed/AdamP/checkpoint/deep_tuning/lr-0.0001/best_trial_1_date_2021-07-19-04-25-54.pth.tar \
--dataset_name=OSDataset_transformed \
--output_path=output/OS_CRCpost_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/OS/ \
--model_path=/ssd2/HistoKT/results/ADPL3_ImageNet_norm_ImageNet/ADP-Release1_norm_ImageNet_color_aug_None_ADPL3/OSDataset_transformed/AdamP/checkpoint/deep_tuning/lr-0.0001/best_trial_1_date_2021-07-19-06-50-19.pth.tar \
--dataset_name=OSDataset_transformed \
--output_path=output/OS_ADPpost_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True
