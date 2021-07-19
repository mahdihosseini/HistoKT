import os


best_lrs = {
    "ADP-Release1": 0.0001,
    "BCSS_transformed": 0.001,
    "OSDataset_transformed": 0.0001,
    "CRC_transformed": 0.0005,
    "AJ-Lymph_transformed": 0.0002,
    "BACH_transformed": "0.00005",
    "GlaS_transformed": 0.001,
    "MHIST_transformed": 0.0005,
    "PCam_transformed": 0.001,
}


def run_baselines(root, CC=True, node="cedar"):
    runscripts = []
    colour_aug = "None"
    # pretrained_model = "ImageNet_weights/resnet18ImageNet.pth"
    pretrained_model = "ImageNet"
    normalization = "ImageNet"

    account = "def-plato"
    # account = "def-msh"

    if node == "cedar":
        gpu = "v100l"
    elif node == "beluga":
        gpu = "v100"
    elif node == "graham":
        gpu = "p100"

    gpu_start = 3

    for dataset in [
        "ADP-Release1",
        # "BCSS_transformed",
        # "OSDataset_transformed",
        # "CRC_transformed",
        # "AJ-Lymph_transformed",
        # "BACH_transformed",
        # "GlaS_transformed",
        # "MHIST_transformed",
        # "PCam_transformed",
    ]:

        if CC:
            if node == "cedar":
                data_storage = "/home/zhan8425/scratch/HistoKTdata"
                env_root = "/home/zhan8425/projects/def-plato/zhan8425/HistoKT"
            elif node == "beluga":
                data_storage = "/scratch/stephy/HistoKTdata"
                env_root = "/home/zhan8425/projects/def-msh/zhan8425/HistoKT"
            env_name = "ENV"
            data_dir = "$SLURM_TMPDIR"
        else:
            data_storage = "None"
            env_root = "/ssd2/HistoKT/source"
            env_name = "env"
            data_dir = f"/ssd{gpu_start + 1}/users/mhosseini/datasets/"

        with open(os.path.join(root, f"NewPretrainingConfigs/{dataset}-{colour_aug}-configAdas.yaml"),
                  "w") as write_file:
            if "ADP" in dataset or "BCSS_transformed" in dataset:
                loss_fn = "MultiLabelSoftMarginLoss"
            else:
                loss_fn = "cross_entropy"
            data = f"""###### Application Specific ######
dataset: '{dataset}' # options: CIFAR100, CIFAR10, ImageNet, ADP-Release1, MHIST
# To use these datasets, please transform them using standardize_datasets.py
# AIDPATH_transformed
# AJ-Lymph_transformed
# BACH_transformed
# CRC_transformed
# GlaS_transformed
# MHIST_transformed
# OSDataset_transformed
# PCam_transformed
# BCSS_transformed
network: 'ResNet18' # AlexNet, DenseNet201, DenseNet169, DenseNet161, DenseNet121, GoogLeNet
# InceptionV3, MNASNet_0_5, MNASNet_0_75, MNASNet_1, MNASNet_1_3, MobileNetV2, ResNet18
# ResNet34, ResNet50, ResNet101, ResNet152, ResNext50, ResNext101, WideResNet50, WideResNet101
# ShuffleNetV2_0_5, ShuffleNetV2_1, ShuffleNetV2_1_5, ShuffleNetV2_2, SqueezeNet_1,
# SqueezeNet_1_1, VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN,
# EfficientNetB4
# ResNet101CIFAR, ResNet50CIFAR, ResNet34CIFAR, ResNet18CIFAR, ResNeXtCIFAR, EfficientNetB0CIFAR, VGG16CIFAR, DenseNet121CIFAR
optimizer: 'Adas' # options: SGD, AdaM, AdaGrad, RMSProp, AdaDelta
scheduler: 'None' # options: AdaS (with SGD), StepLR, CosineAnnealingWarmRestarts, OneCycleLR
# ADP level
level : 'L3' #L1, L2, L3, L3Only

###### Augmentation Methods ######
# ADP ONLY
color_kwargs:
    augmentation: '{colour_aug}' # options: YCbCr, HSV, RGB-Jitter,
                               # Color-Distortion (Color-Jittering followed by color drop),
                               # None
    distortion: 0.0 #options: 0.3 for Color-Distortion
                            # 0.1 for YCbCr-Light and HSV-Light
                            # 1.0 for YCbCr-Strong and HSV-Strong
                            # 1.0 for RBGJitter
degree_of_rotation: 45  # angle for random image rotation for torchvision.transforms.RandomAffine
vertical_flipping: 0.5  #the probability of an image being vertically flipped
horizontal_flipping: 0.5 #the probability of an image being horizontally flipped
horizontal_shift: 0.1 #HorizontalTranslation
vertical_shift: 0.1 #VerticalTranslation

# ALL Datasets
gaussian_blur: False # controlling the use of Gblur
kernel_size: 9 #kernel size for the Gaussian Blur filter
variance: 0.1 #variance of the Gaussian Blur Filter
cutout: False # might change
n_holes: 1
cutout_length: 16

###### Suggested Tune ######
init_lr: 0.03
early_stop_threshold: -1 # set to -1 if you wish not to use early stop, or equally, set to a high value. Set to -1 if not using AdaS
optimizer_kwargs:
  momentum: 0.9
  weight_decay: 5e-4
  beta: 0.98
  linear: False
  gamma: 0.5
  step_size: 25
scheduler_kwargs: {{}}

###### Suggested Default ######
p: 1 # options: 1, 2.
start_trial: 0 # trial to start at, indexing from zero, default 0
n_trials: 3 #increase to more to see more results
num_workers: 4
max_epochs: 250
mini_batch_size: 64
loss: '{loss_fn}' # options: cross_entropy, MultiLabelSoftMarginLoss
early_stop_patience: 10 # epoch window to consider when deciding whether to stop"""
            write_file.write(data)

        runscripts.append(f"run{dataset}-{colour_aug}.sh")

        with open(f"run{dataset}-{colour_aug}-ImageNet.sh", "w") as outfile:
            time_taken = "11:00:00"
            if "CRC" in dataset:
                datafile = "CRC_transformed_2000_per_class"
                time_taken = "23:00:00"
            elif "PCam" in dataset:
                datafile = "PCam_transformed_1000_per_class"
            elif "ADP" in dataset:
                datafile = "ADP\\ V1.0\\ Release"
                time_taken = "23:00:00"
            elif "OSDataset_transformed" in dataset:
                time_taken = "23:00:00"
                datafile = dataset
            elif "BCSS_transformed" in dataset:
                datafile = dataset
                time_taken = "23:00:00"
            else:
                datafile = dataset

            dataCC = f"""#!/bin/bash

### GPU OPTIONS:
### CEDAR: v100l, p100
### BELUGA: *no option, just use --gres=gpu:*COUNT*
### GRAHAM: v100, t4
### see https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm

#SBATCH --gres=gpu:{gpu}:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000M
#SBATCH --account={account}
#SBATCH --time={time_taken}
#SBATCH --output=%x-%j.out

# prepare data

echo "transferring data"
echo ""
date
tar xf {data_storage}/{datafile}.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""
date

"""

            data = f"""source {env_root}/{env_name}/bin/activate

python src/adas/train.py \
--config {env_root}/NewPretrainingConfigs/{dataset}-{colour_aug}-configAdas.yaml \
--output new-ADPL3-ImageNet-pretraining-output/{colour_aug}/{dataset} \
--checkpoint new-ADPL3-ImageNet-pretraining-checkpoint/{colour_aug}/{dataset} \
--data {data_dir} \
--freeze_encoder False \
--save-freq 200 \
--norm_vals {normalization} \
{"" if CC else f"--gpu {gpu_start}"} \
--pretrained_model {pretrained_model}

"""

            outfile.write(data if not CC else dataCC + data)

        with open(f"runslurm_ADPL3_baselines.sh", "w") as outfile:

            outlines = [f"sbatch {filestring}\nsleep 2\n" for filestring in runscripts]

            outfile.write("#!/bin/bash\n")
            outfile.write("".join(outlines))


def run_fine_tune(root, CC=True, node="cedar", account="def-msh"):
    runscripts = []
    # account = "def-plato"
    # account = "def-msh"

    if node == "cedar":
        gpu = "v100l"
    elif node == "beluga":
        gpu = "v100"
    elif node == "graham":
        gpu = "p100"

    optimizer = "AdamP"

    freeze_encoders = [
        # "True",
        "False"
    ]

    color_aug = "None"

    pretrained_datasets = [
        "ADP-Release1",
        # "BCSS_transformed",
        # "OSDataset_transformed",
        # "CRC_transformed",
        # "AJ-Lymph_transformed",
        # "BACH_transformed",
        # "GlaS_transformed",
        # "MHIST_transformed",
        # "PCam_transformed",
    ]
    datasets = [
        # "ADP-Release1",
        # "BCSS_transformed",
        # "OSDataset_transformed",
        # "CRC_transformed",
        "AJ-Lymph_transformed",
        "BACH_transformed",
        "GlaS_transformed",
        "MHIST_transformed",
        "PCam_transformed",
    ]
    gpu_start = 0
    for dataset in datasets:

        # choose from no_norm, target_domain, or a dataset name
        if CC:
            if node == "cedar":
                data_storage = "/home/zhan8425/scratch/HistoKTdata"
                env_root = "/home/zhan8425/projects/def-plato/zhan8425/HistoKT"
            elif node == "beluga":
                data_storage = "/scratch/stephy/HistoKTdata"
                env_root = "/home/zhan8425/projects/def-msh/zhan8425/HistoKT"
            elif node == "graham":
                data_storage = "/scratch/stephy/HistoKTdata"
                env_root = "/home/zhan8425/projects/def-msh/zhan8425/HistoKT"
            env_name = "ENV"
            data_dir = "$SLURM_TMPDIR"
        else:
            data_storage = "None"
            env_root = "/ssd2/HistoKT/source"
            env_name = "env"
            data_dir = f"/ssd{gpu_start + 1}/users/mhosseini/datasets/"

        for pretrained_dataset in pretrained_datasets:
            normalization_all = "ADP-Release1"
            pretrained_model_name = pretrained_dataset

            # pretrained_model_root = f"{env_root}/best-pretraining-checkpoint/None/{pretrained_dataset}"
            # files = os.listdir(pretrained_model_root)
            # assert len(files) == 1
            # pretrained_model = os.path.join(pretrained_model_root, files[0])

            # pretrained_model = f"{env_root}/BestImageNet_Weights/{pretrained_dataset}.pth.tar"
            pretrained_model = f"{env_root}/ADPL3_weights/{normalization_all}.pth.tar"

            if dataset == pretrained_dataset:
                continue
            learning_rate = best_lrs[dataset]
            os.makedirs(f"NewPostTrainingConfigs/{dataset}/{optimizer}", exist_ok=True)
            with open(os.path.join(root, f"NewPostTrainingConfigs/{dataset}/{optimizer}/{color_aug}-config.yaml"),
                      "w") as write_file:
                if "ADP" in dataset or "BCSS_transformed" in dataset:
                    loss_fn = "MultiLabelSoftMarginLoss"
                else:
                    loss_fn = "cross_entropy"
                data = f"""###### Application Specific ######
dataset: '{dataset}' # options: CIFAR100, CIFAR10, ImageNet, ADP-Release1, MHIST
# To use these datasets, please transform them using standardize_datasets.py
# AIDPATH_transformed
# AJ-Lymph_transformed
# BACH_transformed
# CRC_transformed
# GlaS_transformed
# MHIST_transformed
# OSDataset_transformed
# PCam_transformed
# BCSS_transformed
network: 'ResNet18' # AlexNet, DenseNet201, DenseNet169, DenseNet161, DenseNet121, GoogLeNet
# InceptionV3, MNASNet_0_5, MNASNet_0_75, MNASNet_1, MNASNet_1_3, MobileNetV2, ResNet18
# ResNet34, ResNet50, ResNet101, ResNet152, ResNext50, ResNext101, WideResNet50, WideResNet101
# ShuffleNetV2_0_5, ShuffleNetV2_1, ShuffleNetV2_1_5, ShuffleNetV2_2, SqueezeNet_1,
# SqueezeNet_1_1, VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN,
# EfficientNetB4
# ResNet101CIFAR, ResNet50CIFAR, ResNet34CIFAR, ResNet18CIFAR, ResNeXtCIFAR, EfficientNetB0CIFAR, VGG16CIFAR, DenseNet121CIFAR
optimizer: '{optimizer}' # options: SGD, AdaM, AdaGrad, RMSProp, AdaDelta
scheduler: 'StepLR' # options: AdaS (with SGD), StepLR, CosineAnnealingWarmRestarts, OneCycleLR
# ADP level
level : 'L3' #L1, L2, L3, L3Only

###### Augmentation Methods ######
# ADP ONLY
color_kwargs:
    augmentation: '{color_aug}' # options: YCbCr, HSV, RGB-Jitter,
                               # Color-Distortion (Color-Jittering followed by color drop),
                               # None
    distortion: 0.0 #options: 0.3 for Color-Distortion
                            # 0.1 for YCbCr-Light and HSV-Light
                            # 1.0 for YCbCr-Strong and HSV-Strong
                            # 1.0 for RBGJitter
degree_of_rotation: 45  # angle for random image rotation for torchvision.transforms.RandomAffine
vertical_flipping: 0.5  #the probability of an image being vertically flipped
horizontal_flipping: 0.5 #the probability of an image being horizontally flipped
horizontal_shift: 0.1 #HorizontalTranslation
vertical_shift: 0.1 #VerticalTranslation

# ALL Datasets
gaussian_blur: False # controlling the use of Gblur
kernel_size: 9 #kernel size for the Gaussian Blur filter
variance: 0.1 #variance of the Gaussian Blur Filter
cutout: False # might change
n_holes: 1
cutout_length: 16

###### Suggested Tune ######
init_lr: {learning_rate}
early_stop_threshold: -1 # set to -1 if you wish not to use early stop, or equally, set to a high value. Set to -1 if not using AdaS
optimizer_kwargs:
    weight_decay: 5e-4

scheduler_kwargs: 
    gamma: 0.5
    step_size: 20

###### Suggested Default ######
p: 1 # options: 1, 2.
start_trial: 0 # trial to start at, indexing from zero, default 0
n_trials: 3 #increase to more to see more results
num_workers: 4
max_epochs: 250
mini_batch_size: 64
loss: '{loss_fn}' # options: cross_entropy, MultiLabelSoftMarginLoss
early_stop_patience: 10 # epoch window to consider when deciding whether to stop"""
                write_file.write(data)

            runscripts.append(f"run{dataset}-{optimizer}-lr-{learning_rate}-{pretrained_model_name}-norm-{normalization_all}-{color_aug}.sh")

            with open(f"run{dataset}-{optimizer}-lr-{learning_rate}-{pretrained_model_name}-norm-{normalization_all}-{color_aug}.sh", "w") as outfile:
                time_taken = "11:00:00"
                if "CRC" in dataset:
                    datafile = "CRC_transformed_2000_per_class"
                    time_taken = "23:00:00"
                elif "PCam" in dataset:
                    datafile = "PCam_transformed_1000_per_class"
                elif "ADP" in dataset:
                    datafile = "ADP\\ V1.0\\ Release"
                    time_taken = "23:00:00"
                elif "OSDataset_transformed" in dataset:
                    time_taken = "23:00:00"
                    datafile = dataset
                elif "BCSS_transformed" in dataset:
                    datafile = dataset
                    time_taken = "23:00:00"
                else:
                    datafile = dataset

                if normalization_all == "target_domain":
                    normalization = dataset
                else:
                    normalization = normalization_all

                dataCC = f"""#!/bin/bash
    
### GPU OPTIONS:
### CEDAR: v100l, p100
### BELUGA: *no option, just use --gres=gpu:*COUNT*
### GRAHAM: v100, t4
### see https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm

#SBATCH --gres=gpu:{gpu}:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000M
#SBATCH --account={account}
#SBATCH --time={time_taken}
#SBATCH --output=%x-%j.out

# prepare data

echo "transferring data"
echo ""
date
tar xf {data_storage}/{datafile}.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""
date

"""

                data = f"""source {env_root}/{env_name}/bin/activate\n"""
                for freeze_encoder in freeze_encoders:
                    run_part = f"""python src/adas/train.py \
--config {env_root}/NewPostTrainingConfigs/{dataset}/{optimizer}/{color_aug}-config.yaml \
--output {pretrained_model_name}_norm_{normalization_all}_color_aug_None_ADPL3/{dataset}/{optimizer}/output/{"fine_tuning" if freeze_encoder == "True" else "deep_tuning"}/ \
--checkpoint {pretrained_model_name}_norm_{normalization_all}_color_aug_None_ADPL3/{dataset}/{optimizer}/checkpoint/{"fine_tuning" if freeze_encoder == "True" else "deep_tuning"}/lr-{learning_rate} \
--data {data_dir} \
--pretrained_model {pretrained_model} \
--freeze_encoder {freeze_encoder} \
--save-freq 200 \
--norm_vals {normalization} \
{"" if CC else f"--gpu {gpu_start}"}

"""
                    data += run_part

                outfile.write(data if not CC else dataCC + data)
        gpu_start += 1

    if CC:
        for dataset in datasets:
            with open(f"runslurm_ADPL3_{normalization_all}_{dataset}.sh", "w") as outfile:

                outlines = [f"sbatch {filestring}\nsleep 2\n" for filestring in runscripts if "run"+dataset in filestring]

                outfile.write("#!/bin/bash\n")
                outfile.write("".join(outlines))
    else:
        for dataset in datasets:
            with open(f"runlambda_ADPL3_{dataset}.sh", "a") as outfile:

                outlines = [f"bash {filestring}\n" for filestring in runscripts if "run"+dataset in filestring]

                outfile.write("#!/bin/bash\n")
                outfile.write("".join(outlines))
    # if CC:
    #     with open(f"runslurm_ADPL3_{normalization_all}.sh", "w") as outfile:
    #         outlines = [f"sbatch {filestring}\nsleep 2\n" for filestring in runscripts]
    #         outfile.write("#!/bin/bash\n")
    #         outfile.write("".join(outlines))
    # else:
    #     with open(f"runlambda_ADPL3_{dataset}.sh", "w") as outfile:
    #         outlines = [f"bash {filestring}\n" for filestring in runscripts]
    #         outfile.write("#!/bin/bash\n")
    #         outfile.write("".join(outlines))


if __name__ == "__main__":
    root_dir = ""
    # run_baselines(root_dir, CC=False, node="beluga")
    run_fine_tune(root_dir, CC=True, node="graham", account="def-msh")
