import torch
import torch.nn.functional as F
import numpy as np


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


def save_model(model, optimizer, scheduler, config, epoch, trial, save_file, output_filename):
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
