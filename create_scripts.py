import os


def optim_fine_tuning(root):

    optimizer = "AdamP"
    learning_rates = ["0.001", "0.0005", "0.0002", "0.0001", "0.00005"]
    freeze_encoders = ["True", "False"]
    pretrained_model = "/project/6060173/zhan8425/HistoKT/pretraining-checkpoint/Colour-Distortion/ADP-Release1/best_trial_2_date_2021-07-05-14-35-14.pth.tar"
    # pretrained_model = "ImageNet"

    for dataset in ["AJ-Lymph_transformed",
                    "BACH_transformed",
                    "CRC_transformed",
                    "GlaS_transformed",
                    "MHIST_transformed",
                    "OSDataset_transformed",
                    "PCam_transformed"]:

        for learning_rate in learning_rates:
            os.makedirs(f"PostTrainingConfigs/{dataset}_testing/{optimizer}", exist_ok=True)
            with open(f"PostTrainingConfigs/{dataset}_testing/{optimizer}/lr-{learning_rate}-config-{optimizer}.yaml",
                      "w") as outfile:
                if "ADP" in dataset:
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
network: 'ResNet18' # AlexNet, DenseNet201, DenseNet169, DenseNet161, DenseNet121, GoogLeNet
# InceptionV3, MNASNet_0_5, MNASNet_0_75, MNASNet_1, MNASNet_1_3, MobileNetV2, ResNet18
# ResNet34, ResNet50, ResNet101, ResNet152, ResNext50, ResNext101, WideResNet50, WideResNet101
# ShuffleNetV2_0_5, ShuffleNetV2_1, ShuffleNetV2_1_5, ShuffleNetV2_2, SqueezeNet_1,
# SqueezeNet_1_1, VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN,
# EfficientNetB4
# ResNet101CIFAR, ResNet50CIFAR, ResNet34CIFAR, ResNet18CIFAR, ResNeXtCIFAR, EfficientNetB0CIFAR, VGG16CIFAR, DenseNet121CIFAR
optimizer: '{optimizer}' # options: SGD, AdaM, AdaGrad, RMSProp, AdaDelta
scheduler: 'None' # options: AdaS (with SGD), StepLR, CosineAnnealingWarmRestarts, OneCycleLR
# ADP level
level : 'L3Only' #L1, L2, L3, L3Only

###### Augmentation Methods ######
# ADP ONLY
color_kwargs:
    augmentation: 'Color-Distortion' # options: YCbCr, HSV, RGB-Jitter,
                               # Color-Distortion (Color-Jittering followed by color drop),
                               # None
    distortion: 0.3 #options: 0.3 for Color-Distortion
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
max_epochs: 200
mini_batch_size: 32
loss: '{loss_fn}' # options: cross_entropy, MultiLabelSoftMarginLoss
early_stop_patience: 10 # epoch window to consider when deciding whether to stop"""

                outfile.write(data)
            with open(f"run{dataset}-{optimizer}-lr-{learning_rate}-ADP.sh", "w") as outfile:
                time_taken = "11:00:00"
                if "CRC" in dataset:
                    datafile = "CRC_transformed_2000_per_class"
                    time_taken = "23:00:00"
                elif "PCam" in dataset:
                    datafile = "PCam_transformed_1000_per_class"
                elif "ADP" in dataset:
                    datafile = "ADP\\ V1.0\\ Release"
                else:
                    datafile = dataset
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
#SBATCH --mem=16000M
#SBATCH --account=def-plato
#SBATCH --time={time_taken}
#SBATCH --output=%x-%j.out

# prepare data

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate

echo "transferring data"
date
echo ""
tar xf /home/zhan8425/scratch/HistoKTdata/{datafile}.tar -C $SLURM_TMPDIR
echo "Finished transferring"
date
echo ""

"""
                for freeze_encoder in freeze_encoders:
                    run_part = f"""python src/adas/train.py \
--config PostTrainingConfigs/{dataset}_testing/{optimizer}/lr-{learning_rate}-config-{optimizer}.yaml \
--output ADP_post_trained/{dataset}/{optimizer}/output/{"fine_tuning" if freeze_encoder == "True" else "deep_tuning"} \
--checkpoint ADP_post_trained/{dataset}/{optimizer}/checkpoint/{"fine_tuning" if freeze_encoder == "True" else "deep_tuning"}/lr-{learning_rate} \
--data $SLURM_TMPDIR \
--pretrained_model {pretrained_model} \
--freeze_encoder {freeze_encoder} \
--save-freq 200
"""
                    data += run_part
                outfile.write(data)


def run_baselines(root):
    colour_aug = "Colour-Distortion"
    for dataset in [
                    # "ADP-Release1",
                    # "AIDPATH_transformed",
                    # "AJ-Lymph_transformed",
                    # "BACH_transformed",
                    "CRC_transformed",
                    # "GlaS_transformed",
                    # "MHIST_transformed",
                    # "OSDataset_transformed",
                    # "PCam_transformed"
                                    ]:
        with open(os.path.join(root, f"PretrainingConfigs/{dataset}-{colour_aug}-configAdas.yaml"), "w") as write_file:
            if "ADP" in dataset:
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
level : 'L3Only' #L1, L2, L3, L3Only

###### Augmentation Methods ######
# ADP ONLY
color_kwargs:
    augmentation: '{colour_aug}' # options: YCbCr, HSV, RGB-Jitter,
                               # Color-Distortion (Color-Jittering followed by color drop),
                               # None
    distortion: 0.3 #options: 0.3 for Color-Distortion
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
mini_batch_size: 32
loss: '{loss_fn}' # options: cross_entropy, MultiLabelSoftMarginLoss
early_stop_patience: 10 # epoch window to consider when deciding whether to stop"""
            write_file.write(data)

        with open(f"run{dataset}-{colour_aug}.sh", "w") as outfile:
            if "CRC" in dataset:
                datafile = "CRC_transformed_2000_per_class"
            elif "PCam" in dataset:
                datafile = "PCam_transformed_1000_per_class"
            elif "ADP" in dataset:
                datafile = "ADP\\ V1.0\\ Release"
            else:
                datafile = dataset
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
#SBATCH --mem=16000M
#SBATCH --account=def-plato
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

# prepare data

echo "transferring data"
echo ""
date
tar xf /home/zhan8425/scratch/HistoKTdata/{datafile}.tar -C $SLURM_TMPDIR
echo "Finished transferring"
echo ""
date

source ~/projects/def-plato/zhan8425/HistoKT/ENV/bin/activate
python src/adas/train.py \
--config ~/projects/def-plato/zhan8425/HistoKT/PretrainingConfigs/{dataset}-{colour_aug}-configAdas.yaml \
--output pretraining-output/{colour_aug}/{dataset} --checkpoint pretraining-checkpoint/{colour_aug}/{dataset} \
--data $SLURM_TMPDIR \
--save-freq 200
"""
            outfile.write(data)


if __name__ == "__main__":
    root_dir = ""
    # run_baselines(root_dir)
    optim_fine_tuning(root_dir)
