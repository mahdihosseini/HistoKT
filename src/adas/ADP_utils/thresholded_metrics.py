import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .classesADP import classesADP
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

class Thresholded_Metrics:
    
    def __init__(self, targets, predictions, level, network, epoch):
        
        self.target = targets.numpy()
        self.predictions = predictions.numpy()
        
        #class names
        self.class_names = classesADP[level]['classesNames']
        #path
        cur_path = os.path.abspath(os.path.curdir)
        self.eval_dir = os.path.join(cur_path, 'eval')
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        #sess_id
        self.sess_id = 'adp_' + str(network) + '_' + str(level) + '_Epoch_' + str(epoch + 1) + '_Release1_1um_bicubic'
        
        #Get optimal class thresholds
        self.class_thresholds, self.class_fprs, self.class_tprs, self.auc_measures = self.get_optimal_thresholds()
        
        #Get thresholded class accuracies
        self.metric_tprs, self.metric_fprs, self.metric_tnrs, self.metric_fnrs, self.metric_accs, self.metric_f1s = self.get_thresholded_metrics()
        
        #self.auc_measures_U = [ self.auc_measures[i] for i in self.unaugmented_class_inds]
        #self.auc_measures_U.append(self.auc_measures[-1])
        
        #Plot ROC curves
        self.plot_rocs()
        
        #Write metrics to excel
        self.write_to_excel()
        
    def get_optimal_thresholds(self):
        
        def get_opt_thresh(tprs, fprs, thresholds):
            return thresholds[np.argmin(abs(tprs - (1 - fprs)))]
        
        class_fprs = []
        class_tprs = []
        class_thresholds = []
        auc_measures = []
        thresh_rng = [1/3,1]
        
        for iter_class in range(self.predictions.shape[1]):
            fprs, tprs, thresholds = \
                    roc_curve(self.target[:, iter_class], self.predictions[:, iter_class])
            auc_measure = auc(fprs, tprs)
            opt_thresh = min(max(get_opt_thresh(tprs, fprs, thresholds), thresh_rng[0]), thresh_rng[1])
            class_thresholds.append(opt_thresh)
            class_fprs.append(fprs)
            class_tprs.append(tprs)
            auc_measures.append(auc_measure)
        auc_measures.append(sum(np.sum(self.target, 0) * auc_measures)/np.sum(self.target))
        return class_thresholds, class_fprs, class_tprs, auc_measures
    
    def get_thresholded_metrics(self):      
        predictions_thresholded = self.predictions >= self.class_thresholds
        with np.errstate(divide = 'ignore', invalid = 'ignore'):  
            #Obtain Metrics
            cond_positive = np.sum(self.target == 1, 0)
            cond_negative = np.sum(self.target == 0, 0)
            true_positive = np.sum((self.target == 1) & (predictions_thresholded == 1), 0)
            false_positive = np.sum((self.target == 0) & (predictions_thresholded == 1), 0)
            true_negative = np.sum((self.target == 0) & (predictions_thresholded == 0), 0)
            false_negative = np.sum((self.target == 1) & (predictions_thresholded == 0), 0)
            class_tprs = true_positive / cond_positive
            class_fprs = false_positive / cond_negative
            class_tnrs = true_negative / cond_negative
            class_fnrs = false_negative / cond_positive
            class_accs = np.sum(self.target == predictions_thresholded, 0) / predictions_thresholded.shape[0]
            class_f1s = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
            
            #
            cond_positive_T = np.sum(self.target == 1)
            cond_negative_T = np.sum(self.target == 0)
            true_positive_T = np.sum((self.target == 1) & (predictions_thresholded == 1))
            false_positive_T = np.sum((self.target == 0) & (predictions_thresholded == 1))
            true_negative_T = np.sum((self.target == 0) & (predictions_thresholded == 0))
            false_negative_T = np.sum((self.target == 1) & (predictions_thresholded == 0))
            tpr_T = true_positive_T / cond_positive_T
            fpr_T = false_positive_T / cond_negative_T
            tnr_T = true_negative_T / cond_negative_T
            fnr_T = false_negative_T / cond_positive_T
            acc_T = np.sum(self.target == predictions_thresholded) / np.prod(predictions_thresholded.shape)
            f1_T = (2 * true_positive_T) / (2 * true_positive_T + false_positive_T + false_negative_T)
        
            #
            class_tprs = np.append(class_tprs, tpr_T)
            class_fprs = np.append(class_fprs, fpr_T)
            class_tnrs = np.append(class_tnrs, tnr_T)
            class_fnrs = np.append(class_fnrs, fnr_T)
            class_accs = np.append(class_accs, acc_T)
            class_f1s  = np.append(class_f1s, f1_T)
        
        return class_tprs, class_fprs, class_tnrs, class_fnrs, class_accs, class_f1s

    def plot_rocs(self):
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        for iter_class in range(len(self.class_names)):
            plt.plot(self.class_fprs[iter_class], self.class_tprs[iter_class], label=self.class_names[iter_class])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(os.path.join(self.eval_dir, 'ROC_' + self.sess_id + '.png'), bbox_inches='tight')
        plt.close()
        
    def write_to_excel(self):
        sess_xlsx_path = os.path.join(self.eval_dir, 'metrics_' + self.sess_id + '.xlsx')
        df = pd.DataFrame({'HTT': self.class_names + ['Average'],
                           'TPR': list(self.metric_tprs),
                           'FPR': list(self.metric_fprs),
                           'TNR': list(self.metric_tnrs),
                           'FNR': list(self.metric_fnrs),
                           'ACC': list(self.metric_accs),
                           'F1': list(self.metric_f1s),
                           'AUC': self.auc_measures}, columns=['HTT', 'TPR', 'FPR', 'TNR', 'FNR', 'ACC', 'F1', 'AUC'])
        df.to_excel(sess_xlsx_path)
