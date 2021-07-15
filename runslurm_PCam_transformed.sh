#!/bin/bash
sbatch runPCam_transformed-AdamP-lr-0.001-BCSS_transformed-norm-BCSS_transformed-Color-Distortion-0.2.sh
sleep 2
sbatch runPCam_transformed-AdamP-lr-0.001-OSDataset_transformed-norm-OSDataset_transformed-Color-Distortion-0.2.sh
sleep 2
sbatch runPCam_transformed-AdamP-lr-0.001-CRC_transformed-norm-CRC_transformed-Color-Distortion-0.2.sh
sleep 2
sbatch runPCam_transformed-AdamP-lr-0.001-AJ-Lymph_transformed-norm-AJ-Lymph_transformed-Color-Distortion-0.2.sh
sleep 2
sbatch runPCam_transformed-AdamP-lr-0.001-BACH_transformed-norm-BACH_transformed-Color-Distortion-0.2.sh
sleep 2
sbatch runPCam_transformed-AdamP-lr-0.001-GlaS_transformed-norm-GlaS_transformed-Color-Distortion-0.2.sh
sleep 2
sbatch runPCam_transformed-AdamP-lr-0.001-MHIST_transformed-norm-MHIST_transformed-Color-Distortion-0.2.sh
sleep 2
