#!/bin/bash

for NUM in 0.05 0.03 0.01 0.005 0.003
do
  sbatch runMHIST_transformed-Adas-lr-${NUM}.sh
  sleep 2
done
