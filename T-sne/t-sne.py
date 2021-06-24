import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
# import model
from resnet18 import resnet18
# import datasets
from datasets import TransformedDataset
from datasets import ADPDataset


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
    'ADP': {'mean': [0.81233799, 0.64032477, 0.81902153], 'std': [0.18129702, 0.25731668, 0.16800649]}}


def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_features(dataset_name, path_to_root, path_to_pth):
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

    if dataset_name == 'ADP':
        train_set = ADPDataset("L3Only",
                               transform=transform_train,
                               root=path_to_root,
                               split='train')

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set) if dist else None

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            pin_memory=True,
            num_workers=num_workers,
            sampler=train_sampler)
    else:
        train_set = TransformedDataset(transform=transform_train,
                                       root=os.path.join(path_to_root, dataset_name),
                                       split='train')

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set) if dist else None

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            pin_memory=True,
            num_workers=num_workers,
            sampler=train_sampler)
    print("load test data successfully")

    text_label = train_loader.dataset.class_to_idx

    # initialize our implementation of ResNet
    num_classes = len(train_loader.dataset.class_to_idx.items())
    model = resnet18(path_to_pth, pretrained=True, num_classes=num_classes, device=device)
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
            if dataset_name == "ADP":
                y = y.type(torch.LongTensor)
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
        print("targets shape: ", tgts.shape)
        print("features shape: ", features.shape)
    return features, tgts, text_label


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def visualize_tsne_points(tx, ty, labels, text_label, dataset_name, plots_dir):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in text_label:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.savefig(plots_dir + "/" + dataset_name + ".png", bbox_inches = 'tight')
    plt.clf()
    plt.close()


def visualize_tsne(tsne, labels, text_label, dataset_name, plots_dir=None):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels, text_label, dataset_name, plots_dir)


def main(dataset_name_list, root, checkpoint, output=None):
    fix_random_seeds()
    for dataset_name in dataset_name_list:
        path_to_pth_list = list()
        path_to_dataset_cp = os.path.join(checkpoint, dataset_name)
        for file in os.listdir(path_to_dataset_cp):
            print("file/dir: ", file)
            if "per_class" in file:
                temp = os.path.join(path_to_dataset_cp, file)
                print(temp)
                for file2 in os.listdir(temp):
                    if ".pth" in file2 and "best_" in file2:
                        path_to_pth_list.append(os.path.join(temp, file2))
            else:
                if ".pth" in file and "best_" in file:
                    path_to_pth_list.append(os.path.join(path_to_dataset_cp, file))

        for path_to_pth in path_to_pth_list:
            features, labels, text_label = get_features(dataset_name, root, path_to_pth)

            tsne = TSNE(n_components=2).fit_transform(features)

            if output is not None:
                visualize_tsne(tsne, labels, text_label, dataset_name, plots_dir=output)


if __name__ == '__main__':
    # batch size 32 or 64, and num_workers 4 on compute canada
    mini_batch_size = 32
    num_workers = 4

    checkpoint = "/home/zhujiada/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint"
    root = "/scratch/zhan8425/HistoKTdata"
    #root = sys.argv[1]
    output = "/home/zhujiada/projects/def-plato/zhujiada/output"  # None if same as the checkpoint dir

    #dataset_name_list = ["CRC_transformed","PCam_transformed"]
    #dataset_name_list = ["GlaS_transformed", "AJ-Lymph_transformed", "BACH_transformed", "OSDataset_transformed", "MHIST_transformed","AIDPATH_transformed"]
    dataset_name_list = ["MHIST_transformed", "ADP"]
    main(dataset_name_list, root, checkpoint, output)