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
tar xf /home/zhan8425/scratch/HistoKTdata/BACH_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/BACH_transformed_testing/AdamP/lr-0.0005-config-AdamP.yaml --output ADP_trained_on_ImageNet_post_trained_norm_no_norm_aug_Color-Distortion/BACH_transformed/AdamP/output/fine_tuning --checkpoint ADP_trained_on_ImageNet_post_trained_norm_no_norm_aug_Color-Distortion/BACH_transformed/AdamP/checkpoint/fine_tuning/lr-0.0005 --data $SLURM_TMPDIR --pretrained_model /project/6060173/zhan8425/HistoKT/ImageNet_post_trained/ADP-Release1/AdamP/checkpoint/deep_tuning/lr-0.0001/best_trial_2_date_2021-07-10-23-24-04.pth.tar --freeze_encoder True --save-freq 200 --color_aug Color-Distortion --norm_vals no_norm

python src/adas/train.py --config PostTrainingConfigs/BACH_transformed_testing/AdamP/lr-0.0005-config-AdamP.yaml --output ADP_trained_on_ImageNet_post_trained_norm_no_norm_aug_Color-Distortion/BACH_transformed/AdamP/output/deep_tuning --checkpoint ADP_trained_on_ImageNet_post_trained_norm_no_norm_aug_Color-Distortion/BACH_transformed/AdamP/checkpoint/deep_tuning/lr-0.0005 --data $SLURM_TMPDIR --pretrained_model /project/6060173/zhan8425/HistoKT/ImageNet_post_trained/ADP-Release1/AdamP/checkpoint/deep_tuning/lr-0.0001/best_trial_2_date_2021-07-10-23-24-04.pth.tar --freeze_encoder False --save-freq 200 --color_aug Color-Distortion --norm_vals no_norm

