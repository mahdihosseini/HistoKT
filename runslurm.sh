#!/bin/bash

for NUM in 0.05 0.03 0.01 0.005
do
  sbatch runMHIST_transformed-SAM-lr-${NUM}.sh
  sleep 2
done
