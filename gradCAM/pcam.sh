#!/usr/bin/env bash

ROOT=/mnt/d/ADP/HistoKT/gradCAM

# python3 main.py --image_path=$ROOT/examples/001.png_crop_16.png \
# --model_path=$ROOT/examples/best_trial_2_date_2021-07-13-19-43-53.pth.tar \
# --dataset_name=ADP --aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/PCam/ \
--model_path=$ROOT/examples/pretrain_colordistortion/PCam_transformed/best_trial_0_date_2021-07-07-11-05-11.pth.tar \
--dataset_name=PCam_transformed \
--output_path=output/PCam \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/PCam/ \
--model_path=$ROOT/examples/posttrain_ADP/PCam_transformed/best_trial_1_date_2021-07-07-23-44-18.pth.tar \
--dataset_name=PCam_transformed \
--output_path=output/PCam_ADPpost \
--aug_smooth=True --eigen_smooth=True
