#!/bin/bash
sbatch runBACH_transformed-AdamP-lr-5e-05-ADP_level_1-norm-ADP-Release1-Color-Distortion-0.1.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-5e-05-ADP_level_1-norm-ADP-Release1-Color-Distortion-0.2.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-5e-05-ADP_level_1-norm-ADP-Release1-Color-Distortion-0.3.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-5e-05-ADP_level_1-norm-ADP-Release1-Color-Distortion-0.4.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-5e-05-ADP_level_1-norm-ADP-Release1-Color-Distortion-0.5.sh
sleep 2
