#!/bin/bash

for NUM in 0.0005 0.001 0.002 0.005 0.01 0.02 0.05
do
  sbatch runMHIST_transformed-AdamP-lr-${NUM}.sh
  sleep 2
done
