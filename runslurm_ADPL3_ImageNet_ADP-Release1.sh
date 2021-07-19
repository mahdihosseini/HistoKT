#!/bin/bash
sbatch runADP-Release1-AdamP-lr-0.0001-BCSS_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-OSDataset_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-CRC_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-AJ-Lymph_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-BACH_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-GlaS_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-MHIST_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-PCam_transformed-norm-ImageNet-None.sh
sleep 2
