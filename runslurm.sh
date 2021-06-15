#!/bin/bash

for DATASET in CRC_transformed PCam_transformed
do
  for NUM in 100 200 300 500 1000
  do
    echo "sbatch run${DATASET}_transformed_${NUM}_per_class.sh"
    sleep 2
  done
done
