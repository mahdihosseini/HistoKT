#!/usr/bin/env bash

ROOT=/mnt/d/ADP/HistoKT/gradCAM

# python3 main.py --image_path=$ROOT/examples/001.png_crop_16.png \
# --model_path=$ROOT/examples/best_trial_2_date_2021-07-13-19-43-53.pth.tar \
# --dataset_name=ADP --aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=$ROOT/examples/pretrain_colordistortion/BACH_transformed/best_trial_1_date_2021-07-07-11-05-11.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/BACH/ \
--model_path=$ROOT/examples/posttrain_ADP/BACH_transformed/best_trial_1_date_2021-07-07-22-48-15.pth.tar \
--dataset_name=BACH_transformed \
--output_path=output/BACH_ADPpost \
--aug_smooth=True --eigen_smooth=True
