#!/bin/bash
sbatch runBACH_transformed-AdamP-lr-0.001-ADP_trained_on_ImageNet-norm-no_norm.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.0005-ADP_trained_on_ImageNet-norm-no_norm.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.0002-ADP_trained_on_ImageNet-norm-no_norm.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.0001-ADP_trained_on_ImageNet-norm-no_norm.sh
sleep 2
sbatch runBACH_transformed-AdamP-lr-0.00005-ADP_trained_on_ImageNet-norm-no_norm.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.001-ADP_trained_on_ImageNet-norm-no_norm.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.0005-ADP_trained_on_ImageNet-norm-no_norm.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.0002-ADP_trained_on_ImageNet-norm-no_norm.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.0001-ADP_trained_on_ImageNet-norm-no_norm.sh
sleep 2
sbatch runMHIST_transformed-AdamP-lr-0.00005-ADP_trained_on_ImageNet-norm-no_norm.sh
sleep 2