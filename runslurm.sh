#!/bin/bash

for NUM in 0.001 0.0005 0.0003 0.0001 0.00005
do
  sbatch runMHIST_transformed-AdaM-lr-${NUM}.sh
  sleep 2
done

for NUM in 0.5 0.3 0.1 0.05 0.03
do
  sbatch runMHIST_transformed-SGDP-lr-${NUM}.sh
  sleep 2
done

for NUM in 3.0 1.0 0.5 0.3 0.1
do
  sbatch runMHIST_transformed-SAM-lr-${NUM}.sh
  sleep 2
done
