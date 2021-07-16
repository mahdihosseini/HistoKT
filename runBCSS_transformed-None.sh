#!/bin/bash

### GPU OPTIONS:
### CEDAR: v100l, p100
### BELUGA: *no option, just use --gres=gpu:*COUNT*
### GRAHAM: v100, t4
### see https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm

#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000M
#SBATCH --account=def-plato
#SBATCH --time=23:00:00
#SBATCH --output=%x-%j.out

# prepare data

echo "transferring data"
echo ""
date
tar xf /scratch/stephy/HistoKTdata/BCSS_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""
date

source /home/zhan8425/projects/def-msh/zhan8425/HistoKT/ENV/bin/activate

python src/adas/train.py --config /home/zhan8425/projects/def-msh/zhan8425/HistoKT/NewPretrainingConfigs/BCSS_transformed-None-configAdas.yaml --output new-ImageNet-pretraining-output/None/BCSS_transformed --checkpoint new-ImageNet-pretraining-checkpoint/None/BCSS_transformed --data $SLURM_TMPDIR --pretrained_model ImageNet_weights/ResNet18ImageNet.pth --freeze_encoder False --save-freq 200 --norm_vals ImageNet

