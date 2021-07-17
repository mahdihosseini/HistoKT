#!/bin/bash

for DATASET in ADP-Release1 AJ-Lymph_transformed BACH_transformed CRC_transformed GlaS_transformed MHIST_transformed OSDataset_transformed PCam_transformed BCSS_transformed
do
  sh runslurm${DATASET}.sh
done
