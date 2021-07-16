#!/bin/bash
sbatch runOSDataset_transformed-AdamP-lr-0.0001-ADP-Release1-norm-ADP-Release1-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-BCSS_transformed-norm-BCSS_transformed-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-CRC_transformed-norm-CRC_transformed-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-AJ-Lymph_transformed-norm-AJ-Lymph_transformed-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-BACH_transformed-norm-BACH_transformed-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-GlaS_transformed-norm-GlaS_transformed-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-MHIST_transformed-norm-MHIST_transformed-None.sh
sleep 2
sbatch runOSDataset_transformed-AdamP-lr-0.0001-PCam_transformed-norm-PCam_transformed-None.sh
sleep 2
