#!/bin/bash
sbatch runBACH_transformed-AdamP-lr-0.00005-BCSS_transformed-norm-BCSS_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.00005-OSDataset_transformed-norm-OSDataset_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.00005-CRC_transformed-norm-CRC_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.00005-AJ-Lymph_transformed-norm-AJ-Lymph_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.00005-GlaS_transformed-norm-GlaS_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.00005-MHIST_transformed-norm-MHIST_transformed-Color-Distortion-0.1.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.00005-PCam_transformed-norm-PCam_transformed-Color-Distortion-0.1.sh
sleep 2
