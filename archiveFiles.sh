#!/bin/bash

ROOT_PATH="/home/zhan8425/scratch/HistoKTdata"

for DATASET in AIDPATH_transformed AJ-Lymph_transformed BACH_transformed CRC_transformed GlaS_transformed MHIST_transformed OSDataset_transformed PCam_transformed
do
  echo $ROOT_PATH/$DATASET.tar
  tar cf $ROOT_PATH/$DATASET.tar $ROOT_PATH/$DATASET/*
done