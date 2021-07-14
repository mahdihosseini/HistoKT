#!/bin/bash
sbatch runAJ-Lymph_transformed-None.sh
sleep 2
sbatch runBACH_transformed-None.sh
sleep 2
sbatch runGlaS_transformed-None.sh
sleep 2
sbatch runMHIST_transformed-None.sh
sleep 2
sbatch runPCam_transformed-None.sh
sleep 2
