import os.path


def main(root):

    for dataset in ["CRC_transformed",
                    "PCam_transformed"]:
        for num in [100, 200, 300, 500, 1000]:
            with open(os.path.join(root, f"run{dataset}_{num}_per_class.sh"), "w") as write_file:
                data = f"""#!/bin/bash

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
tar -xf /home/zhan8425/scratch/HistoKTdata/{dataset}_{num}_per_class.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config HistoKTconfigs/{dataset}-configAdas.yaml --output .Adas-output/{dataset}/{num}_per_class --checkpoint .Adas-checkpoint/{dataset}/{num}_per_class --data $SLURM_TMPDIR"""

                write_file.write(data)


if __name__ == "__main__":
    root_dir = ""
    main(root_dir)
