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
tar xf /home/zhan8425/scratch/HistoKTdata/PCam_transformed_1000_per_class.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/PCam_transformed_testing/AdamP/lr-0.00005-config-AdamP.yaml --output ADP_trained_on_ImageNet_norm_ImageNet_post_trained_norm_PCam_transformed_aug_Color-Distortion/PCam_transformed/AdamP/output/fine_tuning --checkpoint ADP_trained_on_ImageNet_norm_ImageNet_post_trained_norm_PCam_transformed_aug_Color-Distortion/PCam_transformed/AdamP/checkpoint/fine_tuning/lr-0.00005 --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/ImageNet_post_trained_norm_ImageNet/ADP-Release1/AdamP/checkpoint/deep_tuning/lr-0.0002/best_trial_2_date_2021-07-12-09-19-32.pth.tar --freeze_encoder True --save-freq 200 --color_aug Color-Distortion --norm_vals PCam_transformed

python src/adas/train.py --config PostTrainingConfigs/PCam_transformed_testing/AdamP/lr-0.00005-config-AdamP.yaml --output ADP_trained_on_ImageNet_norm_ImageNet_post_trained_norm_PCam_transformed_aug_Color-Distortion/PCam_transformed/AdamP/output/deep_tuning --checkpoint ADP_trained_on_ImageNet_norm_ImageNet_post_trained_norm_PCam_transformed_aug_Color-Distortion/PCam_transformed/AdamP/checkpoint/deep_tuning/lr-0.00005 --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/ImageNet_post_trained_norm_ImageNet/ADP-Release1/AdamP/checkpoint/deep_tuning/lr-0.0002/best_trial_2_date_2021-07-12-09-19-32.pth.tar --freeze_encoder False --save-freq 200 --color_aug Color-Distortion --norm_vals PCam_transformed

