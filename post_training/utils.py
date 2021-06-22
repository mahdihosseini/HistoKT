import torch
import torch.nn.functional as F
import numpy as np
from loader import ModelLoader

from typing import Dict, Union, List

import pstats
import sys
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracyADP(preds, targets):
    acc5 = 0
    targets_all = targets.data.int()
    acc1 = torch.sum(preds == targets_all)
    preds_cpu = preds.cpu()
    targets_all_cpu = targets_all.cpu()
    for i, pred_sample in enumerate(preds_cpu):
        labelv = targets_all_cpu[i]
        numerator = torch.sum(np.bitwise_and(pred_sample, labelv))
        denominator = torch.sum(np.bitwise_or(pred_sample, labelv))
        acc5 += (numerator.double()/denominator.double())
    return acc1, acc5


def get_accuracies(outputs, labels, batch_size, acc_tracker, config):

    if config["dataset"] == "ADP-Release1":
        preds = (F.sigmoid(torch.tensor(outputs)) > 0.5).int()

        acc1, acc5 = accuracyADP(preds, labels)
        acc_tracker.update(acc1.item() / (batch_size * labels.shape[1]), batch_size)
    else:
        acc_tracker.update(
            (outputs.argmax(1) == labels).sum().detach().cpu().item() / batch_size,
            batch_size)


def save_model(model, optimizer, scheduler, config, epoch, trial, output_filename, save_file):
    print('==> Saving...')
    state = {
        'config': config,
        'state_dict_network': model.state_dict(),
        'state_dict_optimizer': optimizer.state_dict(),
        'state_dict_scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'trial': trial,
        'output_filename': output_filename,
    }
    torch.save(state, save_file)
    del state


def get_model(config):
    if config.config_yaml["network"] == "resnet18":
        if config.pretrained_path:
            loader = ModelLoader.load_from_path(config.pretrained_path, model=config.config_yaml["network"])
            return loader.get_fine_tune_model(config.num_classes,
                                              model_type=config.config_yaml["network"],
                                              freeze_encoder=config.freeze_encoder)
    print("BAD MODEL CONFIG")
    return

def get_criterion(config):
    if config.config_yaml["loss"] == "cross_entropy":
        pass


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
    if (config['dataset'] == 'ADP-Release1' and config['loss'] != 'MultiLabelSoftMarginLoss'):
        raise ValueError('config.yaml: loss must be MultiLabelSoftMarginLoss for ADP Dataset')
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
