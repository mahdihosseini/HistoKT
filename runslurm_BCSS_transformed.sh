#!/bin/bash
sbatch runBCSS_transformed-AdamP-lr-0.001-ADP-Release1-norm-ImageNet-None.sh
sleep 2
sbatch runBCSS_transformed-AdamP-lr-0.001-OSDataset_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runBCSS_transformed-AdamP-lr-0.001-CRC_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runBCSS_transformed-AdamP-lr-0.001-AJ-Lymph_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runBCSS_transformed-AdamP-lr-0.001-BACH_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runBCSS_transformed-AdamP-lr-0.001-GlaS_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runBCSS_transformed-AdamP-lr-0.001-MHIST_transformed-norm-ImageNet-None.sh
sleep 2
sbatch runBCSS_transformed-AdamP-lr-0.001-PCam_transformed-norm-ImageNet-None.sh
sleep 2
