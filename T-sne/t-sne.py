import os
#import cv2
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
            mean=transformed_norm_weights["ImageNet"]["mean"],
            std=transformed_norm_weights["ImageNet"]["std"])])

    if dataset_name == 'ADP':
        train_set = ADPDataset("L1",
                               transform=transform_train,
                               root=path_to_root,
                               split=split)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set) if dist else None

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            pin_memory=True,
            num_workers=num_workers,
            sampler=train_sampler)
    elif dataset_name == "BCSS_transformed":
        train_set = BCSSDataset(root=os.path.join(path_to_root, dataset_name), split=split, transform=transform_train, multi_labelled=True, class_labels=True)
        
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
                                       split=split)

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
            if dataset_name == "ADP" or dataset_name == "BCSS_transformed":
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


def visualize_tsne_points(tx, ty, labels, text_label, output_filename, plots_dir):
    # for every class, we'll add a scatter plot separately
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
        plt.scatter(current_tx, current_ty, label=label, alpha=0.3)

    # build a legend using the labels we set previously
    #plt.legend(loc='best')
    plt.axis('off')
    # save the plot
    plt.savefig(plots_dir + "/" + output_filename + ".png", bbox_inches='tight')
    plt.clf()


def main(dataset_name_list, root, checkpoint, output=None):
    fix_random_seeds()
    for dataset_name in dataset_name_list:
        path_to_pth_list = list()
        if dataset_name == "ADP":
            path_to_dataset_cp = os.path.join(checkpoint, "ADP-Release1")
        else:
            path_to_dataset_cp = os.path.join(checkpoint, dataset_name)
        
        path_to_dataset_cp = os.path.join(path_to_dataset_cp, "AdamP/checkpoint/deep_tuning")
        if dataset_name == "BACH_transformed":
            path_to_dataset_cp = os.path.join(path_to_dataset_cp, "lr-0.00005")
        if dataset_name == "AJ-Lymph_transformed":
            path_to_dataset_cp = os.path.join(path_to_dataset_cp, "lr-0.0002")
        
        for file in os.listdir(path_to_dataset_cp):
            if ".pth" in file and "best_" in file:
                path_to_pth_list.append(os.path.join(path_to_dataset_cp, file))

        for path_to_pth in path_to_pth_list:
            print("path_to_pth: ", path_to_pth)

            split = "train"
            features_train, labels_train, text_label = get_features(dataset_name, split, root, path_to_pth)
            tsne_train = TSNE(n_components=2).fit_transform(features_train)
            print("text_label", text_label)

            split = "valid"
            features_val, labels_val, _ = get_features(dataset_name, split, root, path_to_pth)
            tsne_val = TSNE(n_components=2).fit_transform(features_val)

            split = "test"
            features_test, labels_test, _ = get_features(dataset_name, split, root, path_to_pth)
            tsne_test = TSNE(n_components=2).fit_transform(features_test)

            x_min = np.min([np.min(tsne_train[:, 0]), np.min(tsne_val[:, 0]), np.min(tsne_test[:, 0])])
            x_max = np.max([np.max(tsne_train[:, 0]), np.max(tsne_val[:, 0]), np.max(tsne_test[:, 0])])
            y_min = np.min([np.min(tsne_train[:, 1]), np.min(tsne_val[:, 1]), np.min(tsne_test[:, 1])])
            y_max = np.max([np.max(tsne_train[:, 1]), np.max(tsne_val[:, 1]), np.max(tsne_test[:, 1])])
            print("x limits: ", (x_min, x_max), " , y limits: ", (y_min, y_max))
            # initialize matplotlib plot
            plt.figure()
            ax = plt.gca()  # get the axis handle
            cp_name = os.path.splitext(os.path.basename(path_to_pth))[0]
            if "per_class" in path_to_pth.split('/')[-2]:
                output_filename = dataset_name + "_" + path_to_pth.split('/')[-2] + "_" + cp_name
            else:
                output_filename = dataset_name + "_" + cp_name
            if output is not None:
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                ax = plt.gca()
                ax.set_aspect((x_max-x_min)/(y_max-y_min))
                visualize_tsne_points(tsne_train[:, 0], tsne_train[:, 1], labels_train, text_label, output_filename+"_train", plots_dir=output)
                
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                ax = plt.gca()
                ax.set_aspect((x_max-x_min)/(y_max-y_min))
                visualize_tsne_points(tsne_val[:, 0], tsne_val[:, 1], labels_val, text_label, output_filename+"_val", plots_dir=output)
                
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                ax = plt.gca()
                ax.set_aspect((x_max-x_min)/(y_max-y_min))
                visualize_tsne_points(tsne_test[:, 0], tsne_test[:, 1], labels_test, text_label, output_filename+"_test", plots_dir=output)
            else:
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                ax = plt.gca()
                ax.set_aspect((x_max-x_min)/(y_max-y_min))
                visualize_tsne_points(tsne_train[:, 0], tsne_train[:, 1], labels_train, text_label, output_filename+"_train", plots_dir=checkpoint)
                
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                ax = plt.gca()
                ax.set_aspect((x_max-x_min)/(y_max-y_min))
                visualize_tsne_points(tsne_val[:, 0], tsne_val[:, 1], labels_val, text_label, output_filename+"_val", plots_dir=checkpoint)
                
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                ax = plt.gca()
                ax.set_aspect((x_max-x_min)/(y_max-y_min))
                visualize_tsne_points(tsne_test[:, 0], tsne_test[:, 1], labels_test, text_label, output_filename+"_test", plots_dir=checkpoint)
            plt.close()



if __name__ == '__main__':
    # batch size 32 or 64, and num_workers 4 on compute canada
    mini_batch_size = 32
    num_workers = 4

    checkpoint = "/ssd2/HistoKT/results/post_training_without_color_aug_ImageNet_norm_ImageNet/CRC_transformed_norm_ImageNet_color_aug_None_ImageNet"#"/home/zhujiada/projects/def-plato/zhujiada/HistoKT/.adas-checkpoint-baseline"
    root = "/ssd2/HistoKT/datasets"#"/scratch/zhan8425/HistoKTdata"
    output = "/ssd2/HistoKT/test/Tsne/post_ImageNet_CRC"#"/home/zhujiada/projects/def-plato/zhujiada/output"  # None if same as the checkpoint dir

    dataset_name_list = ["BACH_transformed", "AJ-Lymph_transformed"]#, "GlaS_transformed", "OSDataset_transformed", "MHIST_transformed","CRC_transformed","PCam_transformed", "BCSS_transformed"]
    main(dataset_name_list, root, checkpoint, output)
