#!/bin/bash

for NUM in 0.0002 0.0001 0.00005
do
  sbatch runMHIST_transformed-AdamP-lr-${NUM}.sh
  sleep 2
done
