#!/usr/bin/env bash

ROOT=/mnt/d/ADP/HistoKT/gradCAM

# python3 main.py --image_path=$ROOT/examples/001.png_crop_16.png \
# --model_path=$ROOT/examples/best_trial_2_date_2021-07-13-19-43-53.pth.tar \
# --dataset_name=ADP --aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/OS/ \
--model_path=$ROOT/examples/pretrain_colordistortion/OSDataset_transformed/best_trial_1_date_2021-07-07-11-05-11.pth.tar \
--dataset_name=OSDataset_transformed \
--output_path=output/OS \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/OS/ \
--model_path=$ROOT/examples/posttrain_ADP/OSDataset_transformed/best_trial_2_date_2021-07-08-11-07-21.pth.tar \
--dataset_name=OSDataset_transformed \
--output_path=output/OS_ADPpost \
--aug_smooth=True --eigen_smooth=True
