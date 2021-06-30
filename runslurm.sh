#!/bin/bash

for NUM in 0.00001 0.000005
do
  sbatch runMHIST_transformed-AdaM-lr-${NUM}.sh
  sleep 2
done
