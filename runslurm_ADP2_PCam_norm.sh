#!/bin/bash

for DATASET in PCam_transformed
do
  for LR in 0.001 0.0005 0.0002 0.0001 0.00005
  do
    sbatch run${DATASET}-AdamP-lr-${LR}-ADP_trained_on_ImageNet-norm-PCam_transformed.sh
    sleep 2
  done
done
