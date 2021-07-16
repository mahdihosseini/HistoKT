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
tar xf /scratch/stephy/HistoKTdata/PCam_transformed_1000_per_class.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""
date

source /home/zhan8425/projects/def-msh/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config /home/zhan8425/projects/def-msh/zhan8425/HistoKT/NewPostTrainingConfigs/PCam_transformed/AdamP/None-config.yaml --output OSDataset_transformed_norm_OSDataset_transformed_color_aug_None/PCam_transformed/AdamP/output/deep_tuning/ --checkpoint OSDataset_transformed_norm_OSDataset_transformed_color_aug_None/PCam_transformed/AdamP/checkpoint/deep_tuning/lr-0.001 --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-msh/zhan8425/HistoKT/best-pretraining-checkpoint/None/OSDataset_transformed/best_trial_2_date_2021-07-13-19-35-33.pth.tar --freeze_encoder False --save-freq 200 --norm_vals OSDataset_transformed 

