#!/bin/bash
sbatch runOSDataset_transformed-AdamP-lr-0.0001-BCSS_transformed-norm-BCSS_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-CRC_transformed-norm-CRC_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-AJ-Lymph_transformed-norm-AJ-Lymph_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-BACH_transformed-norm-BACH_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-GlaS_transformed-norm-GlaS_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-MHIST_transformed-norm-MHIST_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-PCam_transformed-norm-PCam_transformed-Color-Distortion-0.1.sh
sleep 2
