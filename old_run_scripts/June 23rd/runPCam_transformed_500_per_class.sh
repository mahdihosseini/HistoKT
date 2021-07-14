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
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out

# prepare data

echo "transferring data"
date
echo ""
tar -xf /home/zhan8425/scratch/HistoKTdata/PCam_transformed_500_per_class.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config HistoKTconfigs/PCam_transformed-configAdas.yaml --output .Adas-output/PCam_transformed/500_per_class --checkpoint .Adas-checkpoint/PCam_transformed/500_per_class --data $SLURM_TMPDIR