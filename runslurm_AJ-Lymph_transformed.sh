#!/bin/bash
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-BCSS_transformed-norm-BCSS_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-OSDataset_transformed-norm-OSDataset_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-CRC_transformed-norm-CRC_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-BACH_transformed-norm-BACH_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-GlaS_transformed-norm-GlaS_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-MHIST_transformed-norm-MHIST_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-PCam_transformed-norm-PCam_transformed-Color-Distortion-0.1.sh
sleep 2
