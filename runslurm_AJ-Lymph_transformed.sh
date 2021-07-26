#!/bin/bash
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-ADP-Release1-norm-ADP-Release1-None.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-BCSS_transformed-norm-BCSS_transformed-None.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-OSDataset_transformed-norm-OSDataset_transformed-None.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-CRC_transformed-norm-CRC_transformed-None.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-BACH_transformed-norm-BACH_transformed-None.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-GlaS_transformed-norm-GlaS_transformed-None.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-MHIST_transformed-norm-MHIST_transformed-None.sh
sleep 2
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-PCam_transformed-norm-PCam_transformed-None.sh
sleep 2
