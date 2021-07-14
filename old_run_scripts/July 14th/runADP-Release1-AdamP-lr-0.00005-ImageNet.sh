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
#SBATCH --time=23:00:00
#SBATCH --output=%x-%j.out

# prepare data

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/ADP\ V1.0\ Release.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/ADP-Release1_testing/AdamP/lr-0.00005-config-AdamP.yaml --output ImageNet_post_trained/ADP-Release1/AdamP/output/deep_tuning --checkpoint ImageNet_post_trained/ADP-Release1/AdamP/checkpoint/deep_tuning/lr-0.00005 --data $SLURM_TMPDIR --pretrained_model ImageNet --freeze_encoder False --save-freq 200
