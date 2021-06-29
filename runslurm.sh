#!/bin/bash

for NUM in 0.00003 0.00001 0.0000005
do
  sbatch runMHIST_transformed-AdaM-lr-${NUM}.sh
  sleep 2
done

for NUM in 0.01 0.005 0.003
do
  sbatch runMHIST_transformed-SGDP-lr-${NUM}.sh
  sleep 2
done

for NUM in 0.07 0.1 0.3
do
  sbatch runMHIST_transformed-Adas-lr-${NUM}.sh
  sleep 2
done
