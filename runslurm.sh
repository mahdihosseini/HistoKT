#!/bin/bash

for DATASET in ADP-Release1 AIDPATH_transformed AJ-Lymph_transformed BACH_transformed CRC_transformed GlaS_transformed MHIST_transformed OSDataset_transformed PCam_transformed
do
  sbatch run${DATASET}-ycbcr.sh
  sleep 2
done
