#!/usr/bin/env bash

ROOT=$PWD
WEIGHT_SHARING=/ssd2/HistoKT/results/weight_sharing



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

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=$WEIGHT_SHARING/None/ADP-Release1_AJ-Lymph_BACH_BCSS_CRC_GlaS_MHIST_OSDataset_PCam_combined_norm_target_domain_color_aug_None/BACH_transformed/AdamP/checkpoint/deep_tuning/lr-0.0002/best_trial_0_date_2021-07-21-02-01-27.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH_ws_all_nocoloraug_none \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=$WEIGHT_SHARING/None/ADP-Release1_CRC_combined_norm_target_domain_color_aug_None/BACH_transformed/AdamP/checkpoint/deep_tuning/lr-0.0005/best_trial_0_date_2021-07-20-18-57-20.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH_ws_ADP+CRC_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=$WEIGHT_SHARING/None/ADP-Release1_BCSS_combined_norm_target_domain_color_aug_None/BACH_transformed/AdamP/checkpoint/deep_tuning/lr-0.0002/best_trial_0_date_2021-07-21-07-01-42.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH_ws_ADP+BCSS_nocoloraug_none \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=$WEIGHT_SHARING/None/ADP-Release1_CRC_OSDataset_combined_norm_target_domain_color_aug_None/BACH_transformed/AdamP/checkpoint/deep_tuning/lr-0.0001/best_trial_1_date_2021-07-21-07-17-15.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH_ws_ADP+CRC+OS_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=$WEIGHT_SHARING/None/ADP-Release1_CRC_BCSS_combined_norm_target_domain_color_aug_None/BACH_transformed/AdamP/checkpoint/deep_tuning/lr-0.0001/best_trial_1_date_2021-07-20-22-00-49.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH_ws_ADP+CRC+BCSS_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=$WEIGHT_SHARING/None/ADP-Release1_CRC_OSDataset_BCSS_combined_norm_target_domain_color_aug_None/BACH_transformed/AdamP/checkpoint/deep_tuning/lr-0.0002/best_trial_2_date_2021-07-21-01-06-57.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH_ws_ADP+CRC+OS+BCSS_nocoloraug_imagenet \
--aug_smooth=True --eigen_smooth=True