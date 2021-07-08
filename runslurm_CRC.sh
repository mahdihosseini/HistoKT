#!/bin/bash

for DATASET in ADP-Release1 AJ-Lymph_transformed BACH_transformed GlaS_transformed MHIST_transformed OSDataset_transformed PCam_transformed BCSS_transformed
do
  for LR in 0.001 0.0005 0.0002 0.0001 0.00005
  do
    sbatch run${DATASET}-AdamP-lr-${LR}-ADP.sh
    sleep 2
  done
done
