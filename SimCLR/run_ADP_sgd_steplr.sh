#!/bin/bash

### GPU  OPTIONS:
### CEDAR: v100l , p100
### BELUGA: *no option , just  use --gres=gpu:*COUNT*
### GRAHAM: v100 , t4

#SBATCH  --gres=gpu:v100l:1
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --cpus-per-task=10
#SBATCH  --mem=32000M
#SBATCH  --account=def-plato
#SBATCH  --time=23:00:00
#SBATCH  --output=%x-%j.out

echo "transferring data"
echo ""
mkdir $SLURM_TMPDIR/data
tar xf ~/HistoKTdata/ADP\ V1.0\ Release.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""

sed -i "s/slurm_tempdir/'${SLURM_TMPDIR}'" ./config/config_ADP_sgd_steplr.yaml

source ~/projects/def-plato/stephy/HistoKT/SimCLR/env/bin/activate
python ~/projects/def-plato/stephy/HistoKT/SimCLR/main.py --config ./config/config_ADP_sgd_steplr.yaml
