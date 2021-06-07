"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini and Mathieu Tuli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
from typing import Dict, Union, List

import pstats
import sys
import torch


def get_is_module():
    mod_name = vars(sys.modules[__name__])
    return True if 'adas.' in mod_name['__name__'] else False


def safe_float_cast(str_number: str) -> float:
    try:
        number = float(str_number)
    except ValueError:
        number = float('nan')
    return number


def pstats_to_dict(stats: pstats.Stats) -> List[Dict[str, Union[str, float]]]:
    formatted_stats = list()
    stats = 'ncalls'+stats.split('ncalls')[-1]
    stats = [line.rstrip().split(None, 5) for line in
             stats.split('\n')]
    for stat in stats[1:]:
        stats_dict = dict()
        if len(stat) >= 5:
            stats_dict['n_calls'] = stat[0]
            stats_dict['tot_time'] = stat[1]
            stats_dict['per_call1'] = stat[2]
            stats_dict['cum_time'] = stat[3]
            stats_dict['per_call2'] = stat[4]
            name = stat[5].split(':')
            stats_dict['name'] = \
                f"{name[0].split('/')[-1]}_line(function)_{name[1]}"
            formatted_stats.append(stats_dict)
    return formatted_stats


def smart_string_to_float(
    string: str,
        e: str = 'could not convert string to float') -> float:
    try:
        ret = float(string)
        return ret
    except ValueError:
        raise ValueError(e)


def smart_string_to_int(
    string: str,
        e: str = 'could not convert string to int') -> int:
    try:
        ret = int(string)
        return ret
    except ValueError:
        raise ValueError(e)
    return float('inf')


def parse_config(
    config: Dict[str, Union[str, float, int]]) -> Dict[
        str, Union[str, float, int]]:
    valid_dataset = ['CIFAR10', 'CIFAR100',
                     'ImageNet', 'TinyImageNet',
                     'ADP-Release1', 'MHIST',
                     "AIDPATH_transformed",
                     "AJ-Lymph_transformed",
                     "BACH_transformed",
                     "CRC_transformed",
                     "GlaS_transformed",
                     "MHIST_transformed",
                     "OSDataset_transformed",
                     "PCam_transformed"]
    if config['dataset'] not in valid_dataset:
        raise ValueError(
            f"config.yaml: unknown dataset {config['dataset']}. " +
            f"Must be one of {valid_dataset}")
    valid_models = {
        'AlexNet', ', DenseNet201', 'DenseNet169', 'DenseNet161',
        'DenseNet121', 'GoogLeNet', 'InceptionV3', 'MNASNet_0_5',
        'MNASNet_0_75', 'MNASNet_1', 'MNASNet_1_3', 'MobileNetV2',
        'MobileNetV2CIFAR', 'SENet18CIFAR', 'ShuffleNetV2CIFAR',
        'ResNet18', 'ResNet34', 'ResNet34CIFAR', 'ResNet50', 'ResNet50CIFAR',
        'ResNet101', 'ResNet101CIFAR', 'ResNet152',
        'ResNext50', 'ResNext101', 'ResNeXtCIFAR', 'WideResNet50',
        'WideResNet101',
        'ShuffleNetV2_0_5', 'ShuffleNetV2_1', 'ShuffleNetV2_1_5',
        'ShuffleNetV2_2', 'SqueezeNet_1', 'SqueezeNet_1_1', 'VGG11',
        'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19',
        'VGG19_BN', 'EfficientNetB4', 'EfficientNetB0CIFAR', 'VGG16CIFAR',
        'DenseNet121CIFAR', 'ResNet18CIFAR',
        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
        'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
        'EfficientNetB6', 'EfficientNetB7', 'EfficientNetB8',
        'EfficientNetB0CIFAR', 'EfficientNetB1CIFAR', 'EfficientNetB2CIFAR',
        'EfficientNetB3CIFAR', 'EfficientNetB4CIFAR', 'EfficientNetB5CIFAR',
        'EfficientNetB6CIFAR', 'EfficientNetB7CIFAR', 'EfficientNetB8CIFAR',
    }
    if config['network'] not in valid_models:
        raise ValueError(
            f"config.yaml: unknown model {config['network']}." +
            f"Must be one of {valid_models}")

    if config['scheduler'] == 'AdaS' and config['optimizer'] not in ['SGD', 'AdamP', 'SGDP', 'SAM']:
        raise ValueError(
            "config.yaml: AdaS can't be used with this optimizer.")

    config['n_trials'] = smart_string_to_int(
        config['n_trials'],
        e='config.yaml: n_trials must be an int')
    # config['beta'] = smart_string_to_float(
    #     config['beta'],
    #     e='config.yaml: beta must be a float')
    e = 'config.yaml: init_lr must be a float or list of floats'
    if not isinstance(config['init_lr'], str):
        if isinstance(config['init_lr'], list):
            for i, lr in enumerate(config['init_lr']):
                if config['init_lr'][i] != 'auto':
                    config['init_lr'][i] = smart_string_to_float(lr, e=e)
        else:
            config['init_lr'] = smart_string_to_float(config['init_lr'], e=e)
    else:
        if config['init_lr'] != 'auto':
            raise ValueError(e)
    config['max_epochs'] = smart_string_to_int(
        config['max_epochs'],
        e='config.yaml: max_epochs must be an int')
    config['early_stop_threshold'] = smart_string_to_float(
        config['early_stop_threshold'],
        e='config.yaml: early_stop_threshold must be a float')
    config['early_stop_patience'] = smart_string_to_int(
        config['early_stop_patience'],
        e='config.yaml: early_stop_patience must be an int')
    config['mini_batch_size'] = smart_string_to_int(
        config['mini_batch_size'],
        e='config.yaml: mini_batch_size must be an int')
    # config['min_lr'] = smart_string_to_float(
    #     config['min_lr'],
    #     e='config.yaml: min_lr must be a float')
    # config['zeta'] = smart_string_to_float(
    #     config['zeta'],
    #     e='config.yaml: zeta must be a float')
    config['p'] = smart_string_to_int(
        config['p'],
        e='config.yaml: p must be an int')
    config['num_workers'] = smart_string_to_int(
        config['num_workers'],
        e='config.yaml: num_works must be an int')
    valid_losses = ['cross_entropy', 'MultiLabelSoftMarginLoss']
    if (config['loss'] not in valid_losses):
        raise ValueError(f'config.yaml: invalid loss type, must be one of {valid_losses}')
    # TODO change this to binarycrossentropy
    #if (config['dataset'] == 'ADP-Release1' and config['loss'] != 'MultiLabelSoftMarginLoss'):
    #    raise ValueError('config.yaml: loss must be MultiLabelSoftMarginLoss for ADP Dataset')
    for k, v in config['optimizer_kwargs'].items():
        if isinstance(v, list):
            for i, val in enumerate(v):
                config['optimizer_kwargs'][k][i] = smart_string_to_float(val)
        else:
            config['optimizer_kwargs'][k] = smart_string_to_float(v)
    for k, v in config['scheduler_kwargs'].items():
        if isinstance(v, list):
            for i, val in enumerate(v):
                config['scheduler_kwargs'][k][i] = smart_string_to_float(val)
        else:
            config['scheduler_kwargs'][k] = smart_string_to_float(v)
    if config['cutout']:
        if config['n_holes'] < 0 or config['cutout_length'] < 0:
            raise ValueError('N holes and length for cutout not set')
    return config


def get_mean_and_std(dataset: torch.utils.data.dataset,
                     batch_size: int = 32,
                     num_workers: int = 0,
                     shuffle: bool = False):
    """
    Slower direct implementation of mean and std calculations

    TODO Could use Welford’s method for computing variance, but not
    used here
    Args:
        dataset: torch.utils.data.dataset -> 
            dataset of images, should return
            (sample, labels)
            images are of shape (n, c, h, w)
        batch_size: int -> 
        num_workers: int ->
        shuffle: bool ->

    """
    mean = torch.zeros(size=(3,))
    std = torch.zeros(size=(3,))
    n_samples = 0.0
    n_pixels = dataset[0][0].size(-1) * dataset[0][0].size(-2)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=shuffle)

    # pass 1 to calculate the mean:
    for images, _ in loader:
        # images are of shape (n, c, h, w)
        # hopefully (they need to be passed through ToTensor)
        # images are also normalized between 0 and 1
        mean += images.mean((-2, -1)).sum(0)
        n_samples += images.size(0)
    
    mean /= n_samples

    for images, _ in loader:
        std += ((images - mean.view(1, 3, 1, 1)) ** 2).sum(dim=(0, 2, 3)) / n_pixels

    std /= n_samples - 1
    std = std ** 0.5
    
    return mean, std


class ThresholdedMetrics:

    def __init__(self, targets, predictions, level, network, epoch):

        self.target = targets.numpy()
        self.predictions = predictions.numpy()

        # class names
        self.class_names = classesADP[level]['classesNames']
        # path
        cur_path = os.path.abspath(os.path.curdir)
        self.eval_dir = os.path.join(cur_path, 'eval')
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        # sess_id
        self.sess_id = 'adp_' + str(network) + '_' + str(level) + '_Epoch_' + str(epoch + 1) + '_Release1_1um_bicubic'

        # Get optimal class thresholds
        self.class_thresholds, self.class_fprs, self.class_tprs, self.auc_measures = self.get_optimal_thresholds()

        # Get thresholded class accuracies
        self.metric_tprs, self.metric_fprs, self.metric_tnrs, self.metric_fnrs, self.metric_accs, self.metric_f1s = self.get_thresholded_metrics()

        # self.auc_measures_U = [ self.auc_measures[i] for i in self.unaugmented_class_inds]
        # self.auc_measures_U.append(self.auc_measures[-1])

        # Plot ROC curves
        self.plot_rocs()

        # Write metrics to excel
        self.write_to_excel()

    def get_optimal_thresholds(self):

        def get_opt_thresh(tprs, fprs, thresholds):
            return thresholds[np.argmin(abs(tprs - (1 - fprs)))]

        class_fprs = []
        class_tprs = []
        class_thresholds = []
        auc_measures = []
        thresh_rng = [1 / 3, 1]

        for iter_class in range(self.predictions.shape[1]):
            fprs, tprs, thresholds = \
                roc_curve(self.target[:, iter_class], self.predictions[:, iter_class])
            auc_measure = auc(fprs, tprs)
            opt_thresh = min(max(get_opt_thresh(tprs, fprs, thresholds), thresh_rng[0]), thresh_rng[1])
            class_thresholds.append(opt_thresh)
            class_fprs.append(fprs)
            class_tprs.append(tprs)
            auc_measures.append(auc_measure)
        auc_measures.append(sum(np.sum(self.target, 0) * auc_measures) / np.sum(self.target))
        return class_thresholds, class_fprs, class_tprs, auc_measures

    def get_thresholded_metrics(self):
        predictions_thresholded = self.predictions >= self.class_thresholds
        with np.errstate(divide='ignore', invalid='ignore'):
            # Obtain Metrics
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
            class_f1s = np.append(class_f1s, f1_T)

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

    



