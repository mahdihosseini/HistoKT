import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from datasets import TransformedDataset
from datasets import ADPDataset
from datasets import BCSSDataset
from resnet import resnet18
from torch.utils.data import DataLoader
from sklearn import metrics
from scipy.special import softmax, expit

# batch size 32 or 64, and num_workers 4 on compute canada
mini_batch_size = 32
num_workers = 4

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


def test_results(path_to_pth, test_dataloader, dataset_name, path_to_out_data=None):
    # eg. path_to_pth = "/HistoKT/.Adas-checkpoint/MHIST_transformed/best_trial.pth"

    results = dict()
    num_classes = len(test_dataloader.dataset.class_to_idx.items())
    #print("num_classes L3Only, should be 22: ", num_classes)

    model = resnet18(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # setting default device
    cp = torch.load(path_to_pth, map_location=device)
    print("best epoch: ", cp['epoch'])
    model.load_state_dict(cp['state_dict_network'])
    model.to(device)  # moving model to compute device
    model.eval()
    
    if dataset_name == "ADP" or dataset_name == "BCSS_transformed":
        dataset_size = len(test_dataloader.dataset)
        test_class_counts = np.sum(test_dataloader.dataset.class_labels, axis=0)
        weightsBCE = dataset_size / test_class_counts
        weightsBCE = torch.as_tensor(weightsBCE, dtype=torch.float32).to(device)
        loss_fn = torch.nn.MultiLabelSoftMarginLoss(weight = weightsBCE).cuda(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    size = len(test_dataloader.dataset)
    test_loss, correct = 0, 0
    tgts = list()
    preds = list()
    pred_label = list()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if dataset_name == "PCam_transformed":
                y = y.type(torch.LongTensor)
                y = y.flatten()
                y = y.to(device, non_blocking=True)
            if dataset_name == "ADP" or dataset_name == "BCSS_transformed":
                y = y.type(torch.LongTensor)
                y = y.to(device, non_blocking=True)
            pred = model(X)

            if dataset_name == 'ADP' or dataset_name == "BCSS_transformed":
                m = nn.Sigmoid()
                pred_temp = (m(pred) > 0.5).int()
                targets_all = y.data.int()
                correct += torch.sum(pred_temp == targets_all).double().detach().cpu().item()

                pred_label.extend(pred_temp.detach().cpu().tolist())

            else:
                correct += (pred.argmax(1) == y).type(torch.float).sum().detach().cpu().item()
                
                if num_classes == 2:
                    pred_label.extend(pred.argmax(1).detach().cpu().tolist())
                else:
                    pred_label.extend(pred.argmax(1).detach().cpu().tolist())  #????

            test_loss += loss_fn(pred, y).detach().cpu().item()

            tgts.extend(y.detach().cpu().tolist())  # int eg. 0, 1
            #if i == 0 :
            #    print(tgts[:5])
            #    print(pred_label[:5])

            if num_classes == 2:
                preds.extend(pred[:, 1].detach().cpu().tolist())
        if num_classes == 2:
            fpr, tpr, thresholds = metrics.roc_curve(
                tgts, preds, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            results["auc"] = auc
        
        #spearmanr_corr = stats.spearmanr(tgts, preds)
        #results["spearmanr_corr"] = spearmanr_corr
        #print("spearmanr_corr: ", spearmanr_corr)  # spearmanr_corr:  SpearmanrResult(correlation=0.6003393380606222, pvalue=4.055666004490729e-06)

        try:
            CK_linear = metrics.cohen_kappa_score(tgts, pred_label, weights = "linear")
            results["CK_linear"] = CK_linear
        except ValueError:
            print("value error for CK_linear")
        try:
            CK_quadratic = metrics.cohen_kappa_score(tgts, pred_label, weights = "quadratic")
            results["CK_quadratic"] = CK_quadratic
        except ValueError:
            print("value error for CK_quadratic")

        if num_classes == 2:
            f1 = metrics.f1_score(tgts, pred_label)
        else:
            f1 = metrics.f1_score(tgts, pred_label, average='micro')
        results["f1_score"] = f1

    if dataset_name == 'ADP' or dataset_name == "BCSS_transformed":
        test_loss /= size
        test_acc1 = (correct / (size * num_classes))
    else:
        test_loss /= i+1
        test_acc1 = correct / size
    print("size = ", size, ", i+1 = ", i+1)

    results["loss"] = test_loss
    results["acc1"] = test_acc1
    df = pd.DataFrame(data=results, index=[0])
    #print(df)

    # save test results datesetname_weightname
    cp_name = os.path.splitext(os.path.basename(path_to_pth))[0]
    output_filename = "test_results_" + dataset_name + "_" + cp_name + ".xlsx".replace(' ', '-')
    if path_to_out_data is not None:
        df.to_excel(os.path.join(path_to_out_data, output_filename))
    else:
        cp_dir = os.path.dirname(path_to_pth)
        df.to_excel(os.path.join(cp_dir, output_filename))
    return


def test_main(path_to_root, path_to_checkpoint, dataset_name_list, path_to_output=None):
    # eg. path_to_root = "/HistoKT/.adas-data"
    # eg. path_to_checkpoint = "/HistoKT/.Adas-checkpoint"
    # /MHIST_transformed" which contains files like best_trial_0_date_2021-06-14-22-23-51.pth

    for dataset_name in dataset_name_list:
        print("****************************", dataset_name, "****************************")
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transformed_norm_weights[dataset_name]["mean"],
                std=transformed_norm_weights[dataset_name]["std"])])

        if dataset_name == 'ADP':
            dataset = ADPDataset("L3Only", root=path_to_root, split='test', transform=transform_test)
        elif dataset_name == "BCSS_transformed":
            dataset = BCSSDataset(root=os.path.join(path_to_root, dataset_name), split='test', transform=transform_test, multi_labelled=True, class_labels=True)
        else:
            dataset = TransformedDataset(root=os.path.join(path_to_root, dataset_name), split="test", transform=transform_test)
        ### just for fast testing ###
        #dataset.samples = dataset.samples[0:50]
        test_dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=False, num_workers=num_workers)
        print("load test data successfully")
        
        if dataset_name == "ADP":
            path_to_dataset_cp = os.path.join(path_to_checkpoint, "ADP-Release1")
        else:
            path_to_dataset_cp = os.path.join(path_to_checkpoint, dataset_name)
        path_to_dataset_cp = os.path.join(path_to_dataset_cp, "AdamP/checkpoint")
        for file in os.listdir(path_to_dataset_cp):
            tune = os.path.join(path_to_dataset_cp, file)
            for file2 in os.listdir(tune):
                rate = os.path.join(tune, file2)
                for file3 in os.listdir(rate):                
                    if ".pth" in file3 and "best_" in file3:
                        path_to_pth = os.path.join(rate, file3)
                        print(path_to_pth)
                        if path_to_output is not None:
                            path_to_out_data = os.path.join(path_to_output, dataset_name)
                            path_to_out_data = os.path.join(path_to_out_data, "AdamP")
                            path_to_out_data = os.path.join(path_to_out_data, file)
                            path_to_out_data = os.path.join(path_to_out_data, file2)
                            if not os.path.isdir(path_to_out_data):
                                os.makedirs(path_to_out_data)
                            test_results(path_to_pth, test_dataloader, dataset_name, path_to_out_data)
                        else:
                            test_results(path_to_pth, test_dataloader, dataset_name)
    return


if __name__ == "__main__":
    checkpoint = "/home/zhujiada/projects/def-plato/zhan8425/HistoKT/ImageNet_post_trained"
    root = "/scratch/zhan8425/HistoKTdata"
    #root = sys.argv[1]
    output = "/home/zhujiada/projects/def-plato/zhujiada/output_ImageNet_post"  # None if same as the checkpoint dir

    # ["ADP", "GlaS_transformed", "AJ-Lymph_transformed", "BACH_transformed", "OSDataset_transformed", "MHIST_transformed","CRC_transformed","PCam_transformed"]
    dataset_name_list = ["BCSS_transformed"]
    test_main(root, checkpoint, dataset_name_list, output)
    pass


