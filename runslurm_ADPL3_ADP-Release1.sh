#!/bin/bash
sbatch runAJ-Lymph_transformed-AdamP-lr-0.0002-ADP-Release1-norm-ADP-Release1-None.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.00005-ADP-Release1-norm-ADP-Release1-None.sh
sleep 2
sbatch runGlaS_transformed-AdamP-lr-0.001-ADP-Release1-norm-ADP-Release1-None.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.0005-ADP-Release1-norm-ADP-Release1-None.sh
sleep 2
sbatch runPCam_transformed-AdamP-lr-0.001-ADP-Release1-norm-ADP-Release1-None.sh
sleep 2
