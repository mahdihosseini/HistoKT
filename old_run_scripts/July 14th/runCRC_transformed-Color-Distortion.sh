#!/bin/bash

### GPU OPTIONS:
### CEDAR: v100l, p100
### BELUGA: *no option, just use --gres=gpu:*COUNT*
### GRAHAM: v100, t4
### see https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm

#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000M
#SBATCH --account=def-plato
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# prepare data

echo "transferring data"
echo ""
date
tar xf /home/zhan8425/scratch/HistoKTdata/CRC_transformed_2000_per_class.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""
date

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config ~/projects/def-plato/zhan8425/HistoKT/PretrainingConfigs/CRC_transformed-Color-Distortion-configAdas.yaml --output pretraining-output/Color-Distortion/CRC_transformed --checkpoint pretraining-checkpoint/Color-Distortion/CRC_transformed --data $SLURM_TMPDIR --save-freq 200
