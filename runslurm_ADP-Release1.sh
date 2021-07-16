#!/bin/bash
sbatch runADP-Release1-AdamP-lr-0.0001-BCSS_transformed-norm-BCSS_transformed-Color-Distortion-0.3.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-OSDataset_transformed-norm-OSDataset_transformed-Color-Distortion-0.3.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-CRC_transformed-norm-CRC_transformed-Color-Distortion-0.3.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-AJ-Lymph_transformed-norm-AJ-Lymph_transformed-Color-Distortion-0.3.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-BACH_transformed-norm-BACH_transformed-Color-Distortion-0.3.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-GlaS_transformed-norm-GlaS_transformed-Color-Distortion-0.3.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-MHIST_transformed-norm-MHIST_transformed-Color-Distortion-0.3.sh
sleep 2
sbatch runADP-Release1-AdamP-lr-0.0001-PCam_transformed-norm-PCam_transformed-Color-Distortion-0.3.sh
sleep 2
