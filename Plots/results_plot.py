import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_result(result_name, path_to_file):
    # plots the value of interested result over the number of epochs in the excel file
    data = pd.read_excel(path_to_file)
    results = []
    count = int(data.columns[-1].split("_")[-1])
    for col in data.columns:
        if col.split("_")[0] == result_name.split("_")[0] and col.split("_")[1] == result_name.split("_")[1]:
            if np.isnan(data[col].tolist()[0]) == False:
                results.append(data[col].tolist()[0])
    if len(results) == count + 1:
        title = path_to_file.split("/")[-3]
        directory = os.path.dirname(path_to_file)
        file_name = path_to_file.split("/")[-1]
        trial = file_name.split("_")[2]
        plots_dir = os.path.join(directory, "plots_"+trial)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        x = range(0, len(results))
        plt.plot(x, results, 'b-')
        plt.xlabel("epoch")
        if result_name.split("_")[0] == "test":
            result_name = "val_" + result_name.split("_")[1]
        plt.ylabel(result_name)
        plt.title(title+" "+trial)
        if result_name.split("_")[1] == "loss":
            plt.yscale("log")
            
        if result_name.split("_")[1] == "acc1":
            plt.ylim((round(min(results),1)-0.1, 1))
            
            x_max = x[np.argmax(results)]
            y_max = max(results)
            plt.plot([x_max], [y_max], 'o')
            plt.annotate("max: ({:}, {:.3f})".format(x_max, y_max), xy=(x_max, y_max), xytext=(x_max, y_max+0.01), horizontalalignment='right', verticalalignment='top', fontsize=8)
        
        plt.savefig(plots_dir + "/" + result_name + ".png", bbox_inches = 'tight')
        
        plt.clf()


def plot_results_for_dir(result_name_list, path_to_dir):
    # plots the values of interested results for a given model
    
    # Example of result_name:
        #"train_acc1"
        #"train_acc5"
        #"train_loss"
        #"in_S"
        #"out_S"
        #"fc_S"
        #"in_rank"
        #"out_rank"
        #"fc_rank"
        #"in_condition"
        #"out_condition"
        #"train_auc"
        #"rank_velocity"
        #"learning_rate"
        #"test_auc"
        #"test_acc1"
        #"test_acc5"
        #"test_loss"

    # Example of path_to_dir:
        # "/HistoKT/BACH_transformed/lr-0.03"
        # which contains file like results_date=2021-06-10-22-36-53_trial=0_ResNet18_BACH_transformed_Adasmomentum=0.9_weight_decay=0.0005_beta=0.98_linear=0.0_gamma=0.5_step_size=25.0_None_LR=0.03.xlsx
    plt.figure()
    for file in os.listdir(path_to_dir):
        if file.endswith(".xlsx"):
            for result_name in result_name_list:
                plot_result(result_name, os.path.join(path_to_dir, file))
    plt.close()
    
path_to_dir = "C:/Users/ryanr/Desktop/Summer_Research/HistoKT/.adas-data/results"
result_name_list = ["train_acc1", "train_loss", "test_acc1", "test_loss"]
plot_results_for_dir(result_name_list, path_to_dir)
