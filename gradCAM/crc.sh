#!/usr/bin/env bash

ROOT=/mnt/d/ADP/HistoKT/gradCAM

# python3 main.py --image_path=$ROOT/examples/001.png_crop_16.png \
# --model_path=$ROOT/examples/best_trial_2_date_2021-07-13-19-43-53.pth.tar \
# --dataset_name=ADP --aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/CRC/ \
--model_path=$ROOT/examples/pretrain_colordistortion/CRC_transformed/best_trial_0_date_2021-07-07-16-50-22.pth.tar \
--dataset_name=CRC_transformed \
--output_path=output/CRC \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/CRC/ \
--model_path=$ROOT/examples/posttrain_ADP/CRC_transformed/best_trial_0_date_2021-07-08-10-47-22.pth.tar \
--dataset_name=CRC_transformed \
--output_path=output/CRC_ADPpost \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/CRC/ \
--model_path=$ROOT/examples/pretrain_nocoloraug/CRC_transformed/best_trial_0_date_2021-07-13-19-43-35.pth.tar \
--dataset_name=CRC_transformed \
--output_path=output/CRC_nocoloraug \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/CRC/ \
--model_path=$ROOT/examples/posttrain_ADP_nocoloraug/CRC_transformed/best_trial_0_date_2021-07-16-13-48-03.pth.tar \
--dataset_name=CRC_transformed \
--output_path=output/CRC_ADPpost_nocoloraug \
--aug_smooth=True --eigen_smooth=True

python3 main.py --image_path=$ROOT/examples/CRC/ \
--model_path=$ROOT/examples/posttrain_ADP_nocoloraug_imagenet/CRC_transformed/best_trial_0_date_2021-07-16-13-48-03.pth.tar \
--dataset_name=CRC_transformed \
--output_path=output/CRC_ADPpost_nocoloraug \
--aug_smooth=True --eigen_smooth=True
