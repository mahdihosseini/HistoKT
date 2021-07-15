#!/bin/bash
sbatch runMHIST_transformed-AdamP-lr-0.0005-BCSS_transformed-norm-BCSS_transformed-Color-Distortion-0.4.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.0005-OSDataset_transformed-norm-OSDataset_transformed-Color-Distortion-0.4.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.0005-CRC_transformed-norm-CRC_transformed-Color-Distortion-0.4.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.0005-AJ-Lymph_transformed-norm-AJ-Lymph_transformed-Color-Distortion-0.4.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.0005-BACH_transformed-norm-BACH_transformed-Color-Distortion-0.4.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.0005-GlaS_transformed-norm-GlaS_transformed-Color-Distortion-0.4.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.0005-PCam_transformed-norm-PCam_transformed-Color-Distortion-0.4.sh
sleep 2
