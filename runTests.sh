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
#SBATCH --mem-per-cpu=32000
#SBATCH --account=def-plato
#SBATCH --time=24:0:0

source ~/projects/def-plato/zhan8425/HistoKTENV/bin/activate
python src/adas/train.py --config src/adas/configSGD.yaml --output .SGD-output/ADP --checkpoint .SGD-checkpoint/ADP
