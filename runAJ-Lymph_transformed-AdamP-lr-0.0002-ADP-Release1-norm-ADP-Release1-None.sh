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
#SBATCH --time=11:00:00
#SBATCH --output=%x-%j.out

# prepare data

echo "transferring data"
echo ""
date
tar xf /scratch/stephy/HistoKTdata/AJ-Lymph_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""
date

source /home/zhan8425/projects/def-msh/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config /home/zhan8425/projects/def-msh/zhan8425/HistoKT/NewPostTrainingConfigs/AJ-Lymph_transformed/AdamP/None-config.yaml --output ADP-Release1_norm_ADP-Release1_color_aug_None/AJ-Lymph_transformed/AdamP/output/deep_tuning/ --checkpoint ADP-Release1_norm_ADP-Release1_color_aug_None/AJ-Lymph_transformed/AdamP/checkpoint/deep_tuning/lr-0.0002 --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-msh/zhan8425/HistoKT/best-pretraining-checkpoint/None/ADP-Release1/best_trial_2_date_2021-07-13-19-43-53.pth.tar --freeze_encoder False --save-freq 200 --norm_vals ADP-Release1 

