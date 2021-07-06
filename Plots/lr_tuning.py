import os
from collections import defaultdict
from results_plot import get_values_for_file_list
import numpy as np
import pandas as pd


def get_file_list(root):
    files_list = []
    for root, _, files in os.walk(root):
        for file in files:
            files_list.append(os.path.join(root, file))
    return files_list


def get_file_dicts(files_list):
    deep_dict = defaultdict(list)
    fine_dict = defaultdict(list)
    for file_path in files_list:
        fp, name = os.path.split(file_path)
        fp, lr = os.path.split(fp)
        fp, tuning = os.path.split(fp)
        if tuning == "deep_tuning":
            deep_dict[lr].append(file_path)
        elif tuning == "fine_tuning":
            fine_dict[lr].append(file_path)
        else:
            print(file_path)
    return deep_dict, fine_dict


def get_relevant_metrics(file_dict, results_list):
    return {key: get_values_for_file_list(results_list, values)
            for key, values in file_dict.items()}


def get_aggregated_values(in_dict):
    out_dict = {}

    for lr, file_dict in in_dict.items():
        top_values = defaultdict(list)
        for file_path, values_dict in file_dict.items():
            for metric_name, values in values_dict.items():
                top_values[metric_name].append(values)
        out_dict[lr] = {metric_name: np.stack(values, axis=0)
                        for metric_name, values in top_values.items()}
    return out_dict


def get_mean_stddev(in_dict,
                    metric_of_interest="test_auc"):
    rows = []
    out_dict = defaultdict(list)
    for lr, values in in_dict.items():
        top_index = values[metric_of_interest].argmax(axis=1)
        for metric, array in values.items():
            vals = values[metric][np.arange(top_index.shape[0]), top_index]
            average_vals = vals.mean()
            std_vals = vals.std(ddof=1)
            out_dict[metric + "_mean"].append(average_vals)
            out_dict[metric + "_std"].append(std_vals)
        rows.append(lr)

    return pd.DataFrame(data=out_dict, index=rows)


def get_csvs(root, optimizer, dataset="MHIST_transformed"):
    files_list = get_file_list(root)

    deep_dict, fine_dict = get_file_dicts(files_list)

    results_list = ["train_acc1", "train_loss", "test_acc1", "test_loss", "train_auc", "test_auc"]
    deep_dict = get_relevant_metrics(deep_dict, results_list)
    fine_dict = get_relevant_metrics(fine_dict, results_list)

    deep_out = get_aggregated_values(deep_dict)
    fine_out = get_aggregated_values(fine_dict)

    deep_out_df = get_mean_stddev(deep_out)
    fine_out_df = get_mean_stddev(fine_out)

    deep_out_df.to_csv(f"{dataset}-{optimizer}_deep_tuned.csv")
    fine_out_df.to_csv(f"{dataset}-{optimizer}_fine_tuned.csv")


if __name__ == "__main__":
    root = "/home/zhan8425/projects/def-plato/zhan8425/HistoKT/ADP_post_trained"
    optim = "AdamP"

    for dataset in ["AIDPATH_transformed",
                    "AJ-Lymph_transformed",
                    "BACH_transformed",
                    "CRC_transformed",
                    "GlaS_transformed",
                    "MHIST_transformed",
                    "OSDataset_transformed",
                    "PCam_transformed"]:
        root_dir = os.path.join(root, dataset, optim, "output")
        get_csvs(root_dir, optim, dataset)