#!/bin/bash
sbatch runOSDataset_transformed-AdamP-lr-0.0001-ADP-Release1-norm-ImageNet-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-BCSS_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-CRC_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-AJ-Lymph_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-BACH_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-GlaS_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-MHIST_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-PCam_transformed-norm-ImageNet-None.sh
sleep 2
