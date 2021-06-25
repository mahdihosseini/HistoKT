#!/bin/bash

### GPU  OPTIONS:
### CEDAR: v100l , p100
### BELUGA: *no option , just  use --gres=gpu:*COUNT*
### GRAHAM: v100 , t4

#SBATCH  --gres=gpu:v100l:1
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --cpus-per-task=10
#SBATCH  --mem=32000M
#SBATCH  --account=def-plato
#SBATCH  --time=10:00:00
#SBATCH  --output=%x-%j.out

source ~/projects/def-plato/stephy/HistoKT/env/bin/activate
python ~/HistoKTdata/CLR/SimCLR/main.py --dataset CIFAR10
