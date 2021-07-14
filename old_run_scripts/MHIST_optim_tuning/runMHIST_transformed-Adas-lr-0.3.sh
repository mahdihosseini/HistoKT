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
#SBATCH --time=5:00:00
#SBATCH --output=%x-%j.out

# prepare data

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/MHIST_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/MHIST_testing/Adas/lr-0.3-config-Adas.yaml --output ADP_post_trained/MHIST_transformed/Adas/output/fine_tuning --checkpoint ADP_post_trained/MHIST_transformed/Adas/checkpoint/fine_tuning/lr-0.3 --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar --freeze_encoder True
python src/adas/train.py --config PostTrainingConfigs/MHIST_testing/Adas/lr-0.3-config-Adas.yaml --output ADP_post_trained/MHIST_transformed/Adas/output/deep_tuning --checkpoint ADP_post_trained/MHIST_transformed/Adas/checkpoint/deep_tuning/lr-0.3 --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar --freeze_encoder False
