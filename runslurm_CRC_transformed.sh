#!/bin/bash
sbatch runCRC_transformed-AdamP-lr-0.0005-ADP-Release1-norm-ADP-Release1-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-BCSS_transformed-norm-BCSS_transformed-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-OSDataset_transformed-norm-OSDataset_transformed-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-AJ-Lymph_transformed-norm-AJ-Lymph_transformed-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-BACH_transformed-norm-BACH_transformed-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-GlaS_transformed-norm-GlaS_transformed-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-MHIST_transformed-norm-MHIST_transformed-None.sh
sleep 2
sbatch runCRC_transformed-AdamP-lr-0.0005-PCam_transformed-norm-PCam_transformed-None.sh
sleep 2
