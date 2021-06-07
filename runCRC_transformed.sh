#!/bin/bash

### GPU OPTIONS:
### CEDAR: v100l, p100
### BELUGA: *no option, just use --gres=gpu:*COUNT*
### GRAHAM: v100, t4
### see https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm

#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-plato
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config src/adas/HistoKTconfigs/CRC_transformed-configAdas.yaml --output .Adas-output/CRC_transformed --checkpoint .Adas-checkpoint/CRC_transformed --data /home/zhan8425/scratch/HistoKTdata