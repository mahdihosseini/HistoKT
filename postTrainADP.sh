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
tar xf /home/zhan8425/scratch/HistoKTdata/PCam_transformed_500_per_class.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/PCam_transformed-configAdas.yaml --output ADP_post_trained/PCam_transformed_500_per_class/output --checkpoint ADP_post_trained/PCam_transformed_500_per_class/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/CRC_transformed_500_per_class.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/CRC_transformed-configAdas.yaml --output ADP_post_trained/CRC_transformed_500_per_class/output --checkpoint ADP_post_trained/CRC_transformed_500_per_class/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar