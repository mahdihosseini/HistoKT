import os
import itertools
#import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from matplotlib.pyplot import cm
# import model
from resnet18 import resnet18
# import datasets
from datasets import TransformedDataset
from datasets import ADPDataset
from datasets import BCSSDataset

# defined in image_transformed/custom_augmentations.py
transformed_norm_weights = {
    'AIDPATH_transformed': {'mean': [0.6032, 0.3963, 0.5897], 'std': [0.1956, 0.2365, 0.1906]},
    'AJ-Lymph_transformed': {'mean': [0.4598, 0.3748, 0.4612], 'std': [0.1406, 0.1464, 0.1176]},
    'BACH_transformed': {'mean': [0.6880, 0.5881, 0.8209], 'std': [0.1632, 0.1841, 0.1175]},
    'CRC_transformed': {'mean': [0.6976, 0.5340, 0.6687], 'std': [0.2272, 0.2697, 0.2247]},
    'GlaS_transformed': {'mean': [0.7790, 0.5002, 0.7765], 'std': [0.1638, 0.2418, 0.1281]},
    'MHIST_transformed': {'mean': [0.7361, 0.6469, 0.7735], 'std': [0.1812, 0.2303, 0.1530]},
    'OSDataset_transformed': {'mean': [0.8414, 0.6492, 0.7377], 'std': [0.1379, 0.2508, 0.1979]},
    'PCam_transformed': {'mean': [0.6970, 0.5330, 0.6878], 'std': [0.2168, 0.2603, 0.1933]},
    'ADP': {'mean': [0.81233799, 0.64032477, 0.81902153], 'std': [0.18129702, 0.25731668, 0.16800649]},
    'BCSS_transformed': {'mean': [0.7107, 0.4878, 0.6726], 'std': [0.1788, 0.2152, 0.1615]},
    'ImageNet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}


def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_features(dataset_name, split, path_to_root, path_to_pth):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    dist = False

    # read the dataset and initialize the data loader
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=transformed_norm_weights[dataset_name]["mean"],
            std=transformed_norm_weights[dataset_name]["std"])])

    train_set = TransformedDataset(transform=transform_train,
                                    root=os.path.join(path_to_root, dataset_name),
                                    split=split)

    if len(train_set.samples) > 500:
        train_set.samples = train_set.samples[0:500]

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set) if dist else None

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=mini_batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=num_workers,
        sampler=train_sampler)

    print("load data successfully")

    text_label = train_loader.dataset.class_to_idx
    # initialize our implementation of ResNet
    num_classes = len(train_loader.dataset.class_to_idx.items())
    model = resnet18(path_to_pth, pretrained=True, device=device, num_classes=num_classes)
    model.to(device)  # moving model to compute device
    model.eval()
    print("load the model successfully")

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None
    tgts = None

    with torch.no_grad():
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if dataset_name == "PCam_transformed":
                y = y.type(torch.LongTensor)
                y = y.flatten()
                y = y.to(device, non_blocking=True)

            output = model(X)
            current_features = output.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features

            curr_labels = y.detach().cpu().numpy()
            if tgts is not None:
                tgts = np.concatenate((tgts, curr_labels))
            else:
                tgts = curr_labels
        #print("targets shape: ", tgts.shape)
        #print("features shape: ", features.shape)
    return features, tgts, text_label


def visualize_tsne_points(tx, ty, labels, text_label, output_filename, plots_dir):
    # for every class, we'll add a scatter plot separately
    color = iter(cm.rainbow(np.linspace(0,1,7)))
    markers = itertools.cycle(('o', 'v', '1', '+', '*', 'X', 'd'))

    last_dataset = ""
    for label in text_label:
        # find the samples of the current class in the data
        # multi-labeled
        if "ADP" in output_filename or "BCSS_transformed" in output_filename:
            indices = [i for i, l in enumerate(labels) if l[text_label[label]] == 1]
        else:
            indices = [i for i, l in enumerate(labels) if l == text_label[label]]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # add a scatter plot with the correponding color and label
        if label.split("_")[-2] != last_dataset:
            c = next(color)
            last_dataset = label.split("_")[-2]
        
        plt.scatter(current_tx, current_ty, label=label, marker = next(markers), alpha=0.3, color=c)

    # build a legend using the labels we set previously
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    #plt.axis('off')
    # save the plot
    plt.savefig(plots_dir + "/" + output_filename + ".png", bbox_inches='tight')
    plt.clf()


def main(dataset_name_list, root, checkpoint, output=None):
    fix_random_seeds()
    features_full = None
    labels_full = None
    text_label_full = None
    for dataset_name in dataset_name_list:
        path_to_dataset_cp = os.path.join(checkpoint, dataset_name)
        for file in os.listdir(path_to_dataset_cp):
            if ".pth" in file and "best_" in file:
                path_to_pth = os.path.join(path_to_dataset_cp, file)

                split = "test"
                features, labels, text_label = get_features(dataset_name, split, root, path_to_pth)
                if features_full is None:
                    features_full = features
                    labels_full = labels
                    text_label_full = {k+"_"+dataset_name: v for k, v in text_label.items()}
                else:
                    features_full = np.concatenate((features_full, features), axis=0)
                    print("feature shape: ", features_full.shape)

                    count = len(text_label_full.keys())
                    labels += count
                    labels_full = np.concatenate((labels_full, labels), axis=0)
                    print(labels_full[:5], labels_full[-5:])

                    for label in text_label:
                        text_label_full[label+"_"+dataset_name] = text_label[label] + count
                    print(text_label_full)

    tsne_full = TSNE(n_components=2).fit_transform(features_full)
    # initialize matplotlib plot
    plt.figure()
    output_filename = "full_7_datasets_diff_makers"
    if output is not None:
        x_min = np.min(tsne_full[:, 0])
        x_max = np.max(tsne_full[:, 0])
        y_min = np.min(tsne_full[:, 1])
        y_max = np.max(tsne_full[:, 1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        ax = plt.gca()  # get the axis handle
        ax.set_aspect((x_max-x_min)/(y_max-y_min))
        visualize_tsne_points(tsne_full[:, 0], tsne_full[:, 1], labels_full, text_label_full, output_filename, plots_dir=output)
    plt.close()

if __name__ == '__main__':
    # batch size 32 or 64, and num_workers 4 on compute canada
    mini_batch_size = 32
    num_workers = 4

    checkpoint = "/ssd2/HistoKT/results/best-pretraining-checkpoint/None"#"/home/zhujiada/projects/def-plato/zhujiada/HistoKT/.adas-checkpoint-baseline"
    root = "/ssd2/HistoKT/datasets"#"/scratch/zhan8425/HistoKTdata"
    output = "/ssd2/HistoKT/test"#"/home/zhujiada/projects/def-plato/zhujiada/output"  # None if same as the checkpoint dir

    dataset_name_list = ["BACH_transformed", "AJ-Lymph_transformed", "GlaS_transformed", "OSDataset_transformed", "MHIST_transformed","CRC_transformed","PCam_transformed"]
    main(dataset_name_list, root, checkpoint, output)
