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
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

# prepare data

echo "transferring data"
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata2/AIDPATH_transformed.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config src/adas/HistoKTconfigs/AIDPATH_transformed-configAdas.yaml --output .Adas-output/AIDPATH_transformed --checkpoint .Adas-checkpoint/AIDPATH_transformed --data $SLURM_TMPDIR