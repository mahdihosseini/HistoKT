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
#SBATCH --mem=32000M
#SBATCH --account=def-plato
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# prepare data

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/AIDPATH_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/AIDPATH_transformed-configAdas.yaml --output ADP_post_trained/AIDPATH_transformed/output --checkpoint .Adas-checkpoint/AIDPATH_transformed/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/AJ-Lymph_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/AJ-Lymph_transformed-configAdas.yaml --output ADP_post_trained/AJ-Lymph_transformed/output --checkpoint .Adas-checkpoint/AJ-Lymph_transformed/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/BACH_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/BACH_transformed-configAdas.yaml --output ADP_post_trained/BACH_transformed/output --checkpoint .Adas-checkpoint/BACH_transformed/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/CRC_transformed_.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/CRC_transformed_-configAdas.yaml --output ADP_post_trained/CRC_transformed_/output --checkpoint .Adas-checkpoint/CRC_transformed_/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/GlaS_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/GlaS_transformed-configAdas.yaml --output ADP_post_trained/GlaS_transformed/output --checkpoint .Adas-checkpoint/GlaS_transformed/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/MHIST_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/MHIST_transformed-configAdas.yaml --output ADP_post_trained/MHIST_transformed/output --checkpoint .Adas-checkpoint/MHIST_transformed/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/OSDataset_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/OSDataset_transformed-configAdas.yaml --output ADP_post_trained/OSDataset_transformed/output --checkpoint .Adas-checkpoint/OSDataset_transformed/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/PCam_transformed_500_per_class.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/PCam_transformed_500_per_class-configAdas.yaml --output ADP_post_trained/PCam_transformed_500_per_class/output --checkpoint .Adas-checkpoint/PCam_transformed_500_per_class/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar