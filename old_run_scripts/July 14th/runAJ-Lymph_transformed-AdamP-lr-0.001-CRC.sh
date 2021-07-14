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
#SBATCH --time=11:00:00
#SBATCH --output=%x-%j.out

# prepare data

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/AJ-Lymph_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/AJ-Lymph_transformed_testing/AdamP/lr-0.001-config-AdamP.yaml --output CRC_post_trained/AJ-Lymph_transformed/AdamP/output/fine_tuning --checkpoint CRC_post_trained/AJ-Lymph_transformed/AdamP/checkpoint/fine_tuning/lr-0.001 --data $SLURM_TMPDIR --pretrained_model /project/6060173/zhan8425/HistoKT/pretraining-checkpoint/Color-Distortion/CRC_transformed/best_trial_2_date_2021-07-07-16-50-22.pth.tar --freeze_encoder True --save-freq 200
python src/adas/train.py --config PostTrainingConfigs/AJ-Lymph_transformed_testing/AdamP/lr-0.001-config-AdamP.yaml --output CRC_post_trained/AJ-Lymph_transformed/AdamP/output/deep_tuning --checkpoint CRC_post_trained/AJ-Lymph_transformed/AdamP/checkpoint/deep_tuning/lr-0.001 --data $SLURM_TMPDIR --pretrained_model /project/6060173/zhan8425/HistoKT/pretraining-checkpoint/Color-Distortion/CRC_transformed/best_trial_2_date_2021-07-07-16-50-22.pth.tar --freeze_encoder False --save-freq 200
