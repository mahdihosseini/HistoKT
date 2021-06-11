#!/bin/bash

### GPU OPTIONS:
### CEDAR: v100l, p100
### BELUGA: *no option, just use --gres=gpu:*COUNT*
### GRAHAM: v100, t4
### see https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --account=def-plato
#SBATCH --time=2:0:0

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate
python src/adas/process_datasets.py
### python src/adas/get_norm.py
