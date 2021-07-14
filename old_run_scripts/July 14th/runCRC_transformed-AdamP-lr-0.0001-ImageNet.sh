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
tar xf /home/zhan8425/scratch/HistoKTdata/CRC_transformed_2000_per_class.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/CRC_transformed_testing/AdamP/lr-0.0001-config-AdamP.yaml --output ImageNet_post_trained/CRC_transformed/AdamP/output/fine_tuning --checkpoint ImageNet_post_trained/CRC_transformed/AdamP/checkpoint/fine_tuning/lr-0.0001 --data $SLURM_TMPDIR --pretrained_model ImageNet --freeze_encoder True --save-freq 200
python src/adas/train.py --config PostTrainingConfigs/CRC_transformed_testing/AdamP/lr-0.0001-config-AdamP.yaml --output ImageNet_post_trained/CRC_transformed/AdamP/output/deep_tuning --checkpoint ImageNet_post_trained/CRC_transformed/AdamP/checkpoint/deep_tuning/lr-0.0001 --data $SLURM_TMPDIR --pretrained_model ImageNet --freeze_encoder False --save-freq 200
