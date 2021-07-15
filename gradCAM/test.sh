#!/usr/bin/env bash

ROOT=/mnt/d/ADP/HistoKT/gradCAM

python3 main.py --image_path=$ROOT/examples/001.png_crop_16.png \
--model_path=$ROOT/examples/best_trial_2_date_2021-07-13-19-43-53.pth.tar \
--dataset_name=ADP --aug_smooth=True --eigen_smooth=True

