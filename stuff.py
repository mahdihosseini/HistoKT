import os.path


def main(root):

    info_list = []
    with open(os.path.join(root, f"postTrainADP.sh"), "w") as write_file:

        for dataset in [
                  "OSDataset_transformed",
                  "PCam_transformed"]:
            if dataset == "PCam_transformed":
                dataset_file = "PCam_transformed_500_per_class.tar"
            elif dataset == "CRC_transformed":
                dataset_file = "CRC_transformed_500_per_class.tar"
            else:
                dataset_file = dataset
            info_list.append(f"""echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/{dataset_file}.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

python src/adas/train.py --config PostTrainingConfigs/{dataset}-configAdas.yaml --output ADP_post_trained/{dataset_file}/output --checkpoint ADP_post_trained/{dataset_file}/checkpoint --data $SLURM_TMPDIR --pretrained_model /home/zhan8425/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint/ADP/best_trial_2.pth.tar""")

        latter_bit = "\n\n".join(info_list)
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
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# prepare data

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate

"""

        write_file.write(data+latter_bit)


if __name__ == "__main__":
    root_dir = ""
    main(root_dir)
