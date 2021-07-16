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
python src/adas/train.py --config /home/zhan8425/projects/def-msh/zhan8425/HistoKT/NewPostTrainingConfigs/BCSS_transformed/AdamP/Color-Distortion-0.1-config.yaml --output AJ-Lymph_transformed_norm_AJ-Lymph_transformed/BCSS_transformed/AdamP/output/deep_tuning/Color-Distortion/distortion-0.1 --checkpoint AJ-Lymph_transformed_norm_AJ-Lymph_transformed/BCSS_transformed/AdamP/checkpoint/deep_tuning/Color-Distortion/distortion-0.1/lr-0.001 --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-msh/zhan8425/HistoKT/best-pretraining-checkpoint/None/AJ-Lymph_transformed/best_trial_1_date_2021-07-14-04-50-12.pth.tar --freeze_encoder False --save-freq 200 --color_aug Color-Distortion --norm_vals AJ-Lymph_transformed 

