#!/bin/bash

for DATASET in ADP-Release1
do
  for LR in 0.001 0.0005 0.0002 0.0001 0.00005
  do
    sbatch run${DATASET}-AdamP-lr-${LR}-ImageNet-norm-ImageNet.sh
    sleep 2
  done
done
