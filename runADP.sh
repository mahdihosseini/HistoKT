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
#SBATCH --time=18:00:00
#SBATCH --output=%x-%j.out

# prepare data

echo "transferring data"
echo ""
mkdir $SLURM_TMPDIR/data
tar xf /home/zhan8425/scratch/HistoKTdata/ADP\ V1.0\ Release.tar -C $SLURM_TMPDIR/ADP\ V1.0\ Release
echo "Finished transferring"
echo ""

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config src/adas/HistoKTconfigs/ADP-configAdas.yaml --output .Adas-output/ADP --checkpoint .Adas-checkpoint/ADP --data $SLURM_TMPDIR