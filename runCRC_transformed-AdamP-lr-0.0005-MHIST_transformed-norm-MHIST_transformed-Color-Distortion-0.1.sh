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
tar xf /scratch/stephy/HistoKTdata/CRC_transformed_2000_per_class.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""
date

source /home/zhan8425/projects/def-msh/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config /home/zhan8425/projects/def-msh/zhan8425/HistoKT/NewPostTrainingConfigs/CRC_transformed/AdamP/Color-Distortion-0.1-config.yaml --output MHIST_transformed_norm_MHIST_transformed/CRC_transformed/AdamP/output/deep_tuning/Color-Distortion/distortion-0.1 --checkpoint MHIST_transformed_norm_MHIST_transformed/CRC_transformed/AdamP/checkpoint/deep_tuning/Color-Distortion/distortion-0.1/lr-0.0005 --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-msh/zhan8425/HistoKT/best-pretraining-checkpoint/None/MHIST_transformed/best_trial_0_date_2021-07-14-05-06-15.pth.tar --freeze_encoder False --save-freq 200 --color_aug Color-Distortion --norm_vals MHIST_transformed 

