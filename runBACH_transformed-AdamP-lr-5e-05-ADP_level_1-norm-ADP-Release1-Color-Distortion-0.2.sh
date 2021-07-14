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

echo "transferring data"
echo ""
date
tar xf /home/zhan8425/scratch/HistoKTdata/BACH_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""
date

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config ~/projects/def-plato/zhan8425/HistoKT/NewPostTrainingConfigs/BACH_transformed/AdamP/Color-Distortion-0.2-config.yaml --output ADP_level_1_norm_ADP-Release1/BACH_transformed/AdamP/output/fine_tuning/Color-Distortion/distortion-0.2 --checkpoint ADP_level_1_norm_ADP-Release1/BACH_transformed/AdamP/checkpoint/fine_tuning/Color-Distortion/distortion-0.2/lr-5e-05 --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/pretrained_weights/ADP-Release1/level_1/best_trial_2_date_2021-07-13-19-43-53.pth.tar --freeze_encoder True --save-freq 200 --color_aug Color-Distortion --norm_vals ADP-Release1

python src/adas/train.py --config ~/projects/def-plato/zhan8425/HistoKT/NewPostTrainingConfigs/BACH_transformed/AdamP/Color-Distortion-0.2-config.yaml --output ADP_level_1_norm_ADP-Release1/BACH_transformed/AdamP/output/deep_tuning/Color-Distortion/distortion-0.2 --checkpoint ADP_level_1_norm_ADP-Release1/BACH_transformed/AdamP/checkpoint/deep_tuning/Color-Distortion/distortion-0.2/lr-5e-05 --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/pretrained_weights/ADP-Release1/level_1/best_trial_2_date_2021-07-13-19-43-53.pth.tar --freeze_encoder False --save-freq 200 --color_aug Color-Distortion --norm_vals ADP-Release1

