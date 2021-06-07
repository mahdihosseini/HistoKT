import os.path


def main(root):

    for dataset in ["AIDPATH_transformed",
                  "AJ-Lymph_transformed",
                  "BACH_transformed",
                  "CRC_transformed",
                  "GlaS_transformed",
                  "MHIST_transformed",
                  "OSDataset_transformed",
                  "PCam_transformed"]:
        with open(os.path.join(root, f"run{dataset}.sh"), "w") as write_file:
            data = f"""#!/bin/bash

### GPU OPTIONS:
### CEDAR: v100l, p100
### BELUGA: *no option, just use --gres=gpu:*COUNT*
### GRAHAM: v100, t4
### see https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm

#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-plato
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py --config src/adas/HistoKTconfigs/{dataset}-configAdas.yaml --output .Adas-output/{dataset} --checkpoint .Adas-checkpoint/{dataset} --data /home/zhan8425/scratch/HistoKTdata"""

            write_file.write(data)


if __name__ == "__main__":
    root_dir = ""
    main(root_dir)
