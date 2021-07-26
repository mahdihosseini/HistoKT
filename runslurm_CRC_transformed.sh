#!/bin/bash
sbatch runCRC_transformed-AdamP-lr-0.0005-ADP-Release1-norm-ImageNet-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-BCSS_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-OSDataset_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-AJ-Lymph_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-BACH_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-GlaS_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-MHIST_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-PCam_transformed-norm-ImageNet-None.sh
sleep 2
