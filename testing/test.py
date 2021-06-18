import os
import sys
import torch
import pandas as pd
import torchvision.transforms as transforms
from datasets import TransformedDataset
from resnet import resnet18
from torch.utils.data import DataLoader
from sklearn import metrics
from scipy.special import softmax, expit

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


def test_results(path_to_pth, test_dataloader, dataset_name, path_to_out_data=None):
    # eg. path_to_pth = "/HistoKT/.Adas-checkpoint/MHIST_transformed/best_trial.pth"

    results = dict()
    num_classes = len(test_dataloader.dataset.class_to_idx.items())

    model = resnet18(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # setting default device
    cp = torch.load(path_to_pth, map_location=device)
    model.load_state_dict(cp['state_dict_network'])
    model.to(device)  # moving model to compute device
    model.eval()

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

            pred = model(X)
            test_loss += loss_fn(pred, y).detach().cpu().item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().detach().cpu().item()

            tgts.extend(y.detach().cpu().tolist())  # int eg. 0, 1
            pred_label.extend(pred.argmax(1).detach().cpu().tolist())

            if num_classes == 2:
                preds.extend(pred[:, 1].detach().cpu().tolist())
                #preds.extend(softmax(pred, axis=1)[:, 1].tolist()) 
                #preds.extend(expit(pred[:, 1]).tolist())
        if num_classes == 2:
            fpr, tpr, thresholds = metrics.roc_curve(
                tgts, preds, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            results["auc"] = auc

        #spearmanr_corr = stats.spearmanr(tgts, preds)
        #results["spearmanr_corr"] = spearmanr_corr
        #print("spearmanr_corr: ", spearmanr_corr)  # spearmanr_corr:  SpearmanrResult(correlation=0.6003393380606222, pvalue=4.055666004490729e-06)

        CK_linear = metrics.cohen_kappa_score(tgts, pred_label, weights = "linear")
        results["CK_linear"] = CK_linear
        CK_quadratic = metrics.cohen_kappa_score(tgts, pred_label, weights = "quadratic")
        results["CK_quadratic"] = CK_quadratic
        f1 = metrics.f1_score(tgts, pred_label)
        results["f1_score"] = f1

    test_loss /= i+1
    results["loss"] = test_loss
    results["acc1"] = correct / size
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
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transformed_norm_weights[dataset_name]["mean"],
                std=transformed_norm_weights[dataset_name]["std"])
        ])

        dataset = TransformedDataset(root=os.path.join(path_to_root, dataset_name), split="test", transform=transform_test)
        ### just for fast testing ###
        #dataset.samples = dataset.samples[0:50]
        # TODO perhaps bump up the batch size to 32 or 64, and num_workers = 4 on compute canada
        test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        path_to_dataset_cp = os.path.join(path_to_checkpoint, dataset_name)
        for file in os.listdir(path_to_dataset_cp):
            if ".pth" in file and "best_" in file:
                path_to_pth = os.path.join(path_to_dataset_cp, file)
                if path_to_output is not None:
                    path_to_out_data = os.path.join(path_to_output, dataset_name)
                    if not os.path.isdir(path_to_out_data):
                        os.makedirs(path_to_out_data)
                    test_results(path_to_pth, test_dataloader, dataset_name, path_to_out_data)
                else:
                    test_results(path_to_pth, test_dataloader, dataset_name)
    return


if __name__ == "__main__":
    checkpoint = "/home/zhujiada/projects/def-plato/zhan8425/HistoKT/.Adas-checkpoint"
    #root = "/scratch/zhan8425/HistoKTdata"
    root = sys.argv[1]
    output = "/home/zhujiada/projects/def-plato/zhujiada/output"  # None if same as the checkpoint dir

    dataset_name_list = ["AIDPATH_transformed", "AJ-Lymph_transformed", "BACH_transformed", "GlaS_transformed", "OSDataset_transformed"]
    # "MHIST_transformed"
    test_main(root, checkpoint, dataset_name_list, output)
    pass


