#!/bin/bash
sbatch runGlaS_transformed-AdamP-lr-0.001-ADP-Release1-norm-ADP-Release1-None.sh
sleep 2
sbatch runGlaS_transformed-AdamP-lr-0.001-BCSS_transformed-norm-BCSS_transformed-None.sh
sleep 2
sbatch runGlaS_transformed-AdamP-lr-0.001-OSDataset_transformed-norm-OSDataset_transformed-None.sh
sleep 2
sbatch runGlaS_transformed-AdamP-lr-0.001-CRC_transformed-norm-CRC_transformed-None.sh
sleep 2
sbatch runGlaS_transformed-AdamP-lr-0.001-AJ-Lymph_transformed-norm-AJ-Lymph_transformed-None.sh
sleep 2
sbatch runGlaS_transformed-AdamP-lr-0.001-BACH_transformed-norm-BACH_transformed-None.sh
sleep 2
sbatch runGlaS_transformed-AdamP-lr-0.001-MHIST_transformed-norm-MHIST_transformed-None.sh
sleep 2
sbatch runGlaS_transformed-AdamP-lr-0.001-PCam_transformed-norm-PCam_transformed-None.sh
sleep 2
